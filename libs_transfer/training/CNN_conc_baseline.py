import json
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
from libs_transfer.prepare_data.data_visualization import xy_plot
import time
import h5py
import pandas as pd
import joblib
import os

class CNN(nn.Module):
    def __init__(self, output_size, inshape=14996, input_channels=1, out_channels1=2, out_channels2=4, dropout_rate=0.1):
        super().__init__()
        self.input_channels = input_channels
        self.out_channels1 = out_channels1
        self.out_channels2 = out_channels2
        self.output_size = output_size
        self.kernel_size = 3
        self.pool = 2
        self.dropout_rate = dropout_rate

        self.c1 = nn.Conv1d(input_channels, out_channels1, self.kernel_size, stride=1)
        self.p = nn.AvgPool1d(self.pool, self.pool)
        self.c2 = nn.Conv1d(out_channels1, out_channels2, self.kernel_size, stride=2)

        dummy_input = torch.zeros(1, self.input_channels, inshape)
        with torch.no_grad():
            dummy_out = self.c1(dummy_input)
            dummy_out = self.p(dummy_out)
            dummy_out = self.c2(dummy_out)
            dummy_out = self.p(dummy_out)
            flattened_size = dummy_out.view(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, output_size)

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        x = x.unsqueeze(0)  # (batch, features) -> (1, batch, features)
        x = x.transpose(0, 1)  # (1, batch, features ) -> (batch, 1, features), (batch_size x channels x seq_len)
        batch_size = x.size(0)

        x = self.c1(x)
        x = self.p(x)
        x = self.c2(x)
        x = self.p(x)

        # Reshape to (batch_size x -1) for fc
        x = x.view(batch_size, -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)

        return x

class HelperSpectraDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

def train_concentration_predictors(epochs=40, batch_size=128, data_folder='./examples/processed_data', wavelength_range=(250, 850), lr=0.001, do_relu=True, do_plot=False, total_emis_norm=True):

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(torch.cuda.get_device_name(0))
        print(torch.cuda.get_device_properties(0))
    else:
        device = torch.device("cpu")
        print("using CPU, GPU is not available")

    with h5py.File(os.path.join(data_folder, 'spectra.h5'), 'r') as hf:
        wavelen = np.array(hf['calibration'][()])
        spectra_0_idx = np.where(wavelen == wavelength_range[0])[0][0]
        spectra_last_idx = np.where(wavelen == wavelength_range[1])[0][0]

        labels = np.array(hf['labels_one_hot'])
        all_spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx + 1])

        all_atm = np.array(hf['atm_one_hot'][()])
        all_ene = np.array(hf['energy_one_hot'][()])

        num_energies = all_ene.shape[1]

        conditions = np.array([np.kron(row1, row2) for row1, row2 in zip(all_atm, all_ene)])

    with open(os.path.join(data_folder, 'label_dict.json'), 'r') as fp:
        idx_to_label_dict = json.load(fp)

    with open(os.path.join(data_folder, 'atm_dict.json'), 'r') as fp:
        chosen_to_name = json.load(fp)

    with open(os.path.join(data_folder, 'perc_dict.json'), 'r') as fp:
        en_dict = json.load(fp)

    conc_df = pd.read_excel(os.path.join(data_folder, 'Concentrations.xlsx'))
    conc_df = conc_df.dropna().transpose()
    conc_df.columns = [a.upper() for a in conc_df.iloc[0].tolist()]
    conc_df = conc_df.iloc[1:]
    conc_df.index = [a.upper() for a in conc_df.index]
    elements = conc_df.columns.tolist()

    torch.cuda.empty_cache()

    if do_relu:
        all_spectra = np.maximum(all_spectra, 0)  # relu

    if total_emis_norm:
        all_spectra = all_spectra / np.sum(all_spectra, axis=1, keepdims=True)  # total emissivity normalization

    emis_std = StandardScaler().fit(all_spectra)

    all_spectra = emis_std.transform(all_spectra)

    for _chosen_cond in range(conditions.shape[1]):

        chosen_cond = np.where(np.argmax(conditions, axis=1) == _chosen_cond)[0]

        chosen_labels = labels[chosen_cond, :]
        chosen_labels = np.argmax(chosen_labels, axis=1)
        chosen_labels = np.array([idx_to_label_dict[str(i)] for i in chosen_labels])
        concentrations = np.array([conc_df.loc[name].to_list() if name in conc_df.index else [0] * conc_df.shape[1] for name in chosen_labels])
        non_zero_idx = np.where(concentrations[:, 0] != 0)[0]
        concentrations = concentrations[non_zero_idx]

        conc_std = StandardScaler().fit(concentrations)
        concentrations = conc_std.transform(concentrations)

        cond_ = _chosen_cond // num_energies
        en_ = _chosen_cond % num_energies

        condition_text = chosen_to_name[str(cond_)]
        energy_text = en_dict[str(en_)]

        joblib.dump(conc_std, os.path.join(data_folder, f'conc_std_{condition_text}_{energy_text}.joblib'))
        joblib.dump(emis_std, os.path.join(data_folder, f'spectra_std_{condition_text}_{energy_text}.joblib'))

        current_spectra = all_spectra[chosen_cond][non_zero_idx]

        x_train, x_val, y_train, y_val = train_test_split(current_spectra, concentrations, test_size=0.1, random_state=42)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

        train_dataloader = DataLoader(HelperSpectraDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(HelperSpectraDataset(x_val, y_val), batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(HelperSpectraDataset(x_test, y_test), batch_size=x_test.shape[0], shuffle=True)

        output_size = concentrations.shape[1]
        inshape = x_train.shape[1]
        print(f'Input shape for condition {condition_text} at {energy_text}: {inshape}, output shape: {output_size}')

        model = CNN(output_size=output_size, inshape=inshape).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        phases = ['train', 'valid']
        loss_train = []
        loss_valid = []
        start = time.time()

        for epoch in range(epochs):
            n_all = 0
            running_loss = 0

            for phase in phases:
                if phase == 'train':
                    model.train()
                    load_data = train_dataloader
                elif phase == 'valid':
                    model.eval()
                    load_data = val_dataloader

                for x, y in load_data:
                    x = x.to(device)
                    y = y.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        pred = model(x)
                        loss = criterion(pred, y)

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item()*y.size(0)
                    n_all += y.size(0)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / n_all

                print(f'Phase: {phase}, Epoch: [{epoch+1}/{epochs}], epoch loss: {epoch_loss:.4f}')
                if phase == 'valid':
                    tt = time.time() - start
                    remaining_epochs = epochs - (epoch + 1)
                    eta_tt = tt * remaining_epochs / (epoch + 1)
                    print(f'time elapsed: {int(tt // 60):02d}:{int(tt % 60):02d} min, ETA: {int(eta_tt // 60):02d}:{int(eta_tt % 60):02d} min\n')
                    loss_valid.append(epoch_loss)
                elif phase == 'train':
                    loss_train.append(epoch_loss)

        if do_plot:
            xy_plot([[i for i in range(1, epochs+1)], [i for i in range(1, epochs+1)]], [loss_train, loss_valid], xy_labels=['Epoch', 'Loss'], labels= ['Train', 'Valid'], title='Training and validation Loss for CNN example data')

        with torch.no_grad():
            running_loss=0
            n_all = 0
            for i, (x, y) in enumerate(test_dataloader):
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                running_loss += loss.item()*y.size(0)
                n_all += y.size(0)
            epoch_loss = running_loss / n_all
            print(f'Test loss: {epoch_loss:.4f}')

            y = conc_std.inverse_transform(y.cpu().numpy())
            pred = conc_std.inverse_transform(pred.cpu().numpy())

            rmse = np.sqrt(np.mean((y - pred)**2, axis=0))
            mae = np.mean(np.abs(y - pred), axis=0)
            print(f'Condition: {condition_text} Energy: {energy_text}\nElements:')
            print(elements)
            print(f'RMSE: {rmse}\n MAE: {mae}\n\n')

        torch.save(model, os.path.join(data_folder, f'CNN_{condition_text}_{energy_text}.pth'))
        torch.save(model.state_dict(), os.path.join(data_folder, f'CNN_{condition_text}_{energy_text}_state_dict.pth'))
