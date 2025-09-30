import json
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
from data_visualization import xy_plot
import time
import h5py
import pandas as pd
import joblib

class CNN(nn.Module):
    def __init__(self, output_size, inshape=14996, input_channels=1, out_channels1=2, out_channels2=4, dropout_rate=0.1):
        super(CNN, self).__init__()
        self.input_channels = input_channels
        self.out_channels1 = out_channels1
        self.out_channels2 = out_channels2
        self.output_size = output_size
        self.kernel_size = 3
        self.pool = 2
        self.dropout_rate = dropout_rate

        self.bn = nn.BatchNorm1d(inshape)

        self.c1 = nn.Conv1d(input_channels, out_channels1, self.kernel_size, stride=1)
        self.p = nn.AvgPool1d(self.pool, self.pool)
        self.c2 = nn.Conv1d(out_channels1, out_channels2, self.kernel_size, stride=2)

        self.fc1 = nn.Linear(inshape, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.ModuleList([nn.Linear(128, 1) for i in range(output_size)])

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        x = x.unsqueeze(0)  # (batch, features) -> (1, batch, features)
        x = x.transpose(0, 1)  # (1, batch, features ) -> (batch, 1, features), (batch_size x channels x seq_len)
        batch_size = x.size(0)

        x = self.c1(x)
        # x = self.dropout(x)
        x = self.p(x)
        x = self.c2(x)
        # x = self.dropout(x)
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
        x = torch.cat([self.fc5[i](x) for i in range(self.output_size)], dim=1)

        return x

if __name__ == '__main__':

    epochs = 40
    batch_size = 128

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(torch.cuda.get_device_name(0))
        print(torch.cuda.get_device_properties(0))
    else:
        device = torch.device("cpu")
        print("using CPU, GPU is not available")

    with h5py.File('new_data\\spectra.h5', 'r') as hf:
        wavelen = np.array(hf['calibration'][()])
        spectra_0_idx = np.where(wavelen == 250)[0][0]
        spectra_last_idx = np.where(wavelen == 850)[0][0]

        labels = np.array(hf['labels_one_hot'])
        all_spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx + 1])[:, 0::3] #take every 2nd measurement

        all_atm = np.array(hf['atm_one_hot'][()])
        all_ene = np.array(hf['energy_one_hot'][()])

        conditions = np.array([np.kron(row1, row2) for row1, row2 in zip(all_atm, all_ene)])
        # c_ = np.where(np.argmax(conditions, axis=1) == 1)[0]
        # c__ = np.where(np.argmax(conditions, axis=1) == 3)[0]
        # c_ = np.concatenate([c_, c__])
        # all_spectra = all_spectra[c_]
        # conditions = conditions[c_][:,[1,3]]

        c = np.where(np.argmax(conditions, axis=1) != 0)[0]  # no vacuum 50%
        all_spectra = all_spectra[c]
        labels = labels[c]
        conditions = conditions[c][:, [1, 2, 3]]

    with open('new_data/label_dict.json', 'r') as fp:
        idx_to_label_dict = json.load(fp)

    conc_df = pd.read_excel(r'Regoliths/Concentrations.xlsx')
    conc_df = conc_df.dropna().transpose()
    conc_df.columns = [a.upper() for a in conc_df.iloc[0].tolist()]
    conc_df = conc_df.iloc[1:]
    conc_df.index = [a.upper() for a in conc_df.index]
    elements = conc_df.columns.tolist()

    torch.cuda.empty_cache()

    all_spectra = np.maximum(all_spectra, 0)  # relu

    all_spectra = all_spectra / np.sum(all_spectra, axis=1, keepdims=True)  # total emissivity normalization
    emis_std = StandardScaler().fit(all_spectra)

    all_spectra = emis_std.transform(all_spectra)

    chosen_cond = 1

    chosen_cond = np.where(np.argmax(conditions, axis=1) == chosen_cond)[0]

    mars_labels = labels[chosen_cond, :]
    mars_labels = np.argmax(mars_labels, axis=1)
    mars_labels = np.array([idx_to_label_dict[str(i)] for i in mars_labels])
    concentrations = np.array([conc_df.loc[name].to_list() if name in conc_df.index else [0] * conc_df.shape[1] for name in mars_labels])
    non_zero_idx = np.where(concentrations[:, 0] != 0)[0]
    concentrations = concentrations[non_zero_idx]

    conc_std = StandardScaler().fit(concentrations)
    concentrations = conc_std.transform(concentrations)

    # joblib.dump(conc_std, 'CNN\\conc_std_earth100.joblib')
    # print('saved conc scaler')

    all_spectra = all_spectra[chosen_cond][non_zero_idx]

    # lbs_dummy, label_dict = to_onehot(mars_labels)

    # label_dict = {i: str(i) for i in range(lbs_dummy.shape[0])}

    #split into train-val-test
    x_train, x_val, y_train, y_val = train_test_split(all_spectra, concentrations, test_size=0.1, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    print(x_train.shape, x_val.shape, x_test.shape)

    #Used for torch DataLoader
    class SpectraDataset(Dataset):
        def __init__(self, x, y):
            self.x = torch.from_numpy(x.astype(np.float32))
            self.y = torch.from_numpy(y.astype(np.float32))
            self.n_samples = self.x.shape[0]

        def __getitem__(self, index):
            return self.x[index], self.y[index]

        def __len__(self):
            return self.n_samples

    train_dataloader = DataLoader(SpectraDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(SpectraDataset(x_val, y_val), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(SpectraDataset(x_test, y_test), batch_size=x_test.shape[0], shuffle=True)

    output_size = concentrations.shape[1]
    inshape = 4996

    model = CNN(output_size, inshape)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    phases = ['train', 'valid']
    loss_train = []
    acc_train = []
    loss_valid = []
    acc_valid = []
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

            for i, (x, y) in enumerate(load_data):
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

    # xy_plot([[i for i in range(1, epochs+1)], [i for i in range(1, epochs+1)]], [loss_train, loss_valid], xy_labels=['Epoch', 'Loss'], labels= ['Train', 'Valid'], title='Training and validation Loss for CNN example data')

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
        print(elements)
        print(f'RMSE: {rmse}\n MAE: {mae}')

        # elements conc: ['SIO2', 'AL2O3', 'CAO', 'FE2O3', 'K2O', 'MGO', 'MNO', 'TIO2']

        # vacuum 100%
        # Test loss: 0.2831
        # for transformed data:
        # RMSE: [2.0119216  2.6112318  2.4053006  2.3022451  0.3270417  3.6878777  0.0197287  0.58388054] 1.74
        # MAE: [1.492175   1.7008913  1.6676326  1.5490277  0.17735673 2.6904933  0.01048164 0.33645383]

        # earth 50%
        # Test loss: 0.0621
        # for transformed data:
        # RMSE: [0.9666956  1.2034836  1.0819154  1.0059204  0.15415709 1.592586  0.01085762 0.27204645] 0.78
        # MAE: [0.61862046 0.65612376 0.7018361  0.60216606 0.07643874 1.021763  0.00495991 0.12165295]

        # earth 100%
        # Test loss: 0.0551
        # ['SIO2', 'AL2O3', 'CAO', 'FE2O3', 'K2O', 'MGO', 'MNO', 'TIO2']
        # RMSE: [0.8979427  1.1751212  0.96777856 1.0007979  0.13131672 1.4723806  0.01037163 0.26599115] 0.75
        # MAE: [0.5243046  0.5593083  0.6112533  0.5174761  0.06065497 0.8649015  0.0046122  0.11226385]

    # torch.save(model, 'CNN/CNN_earth50_every3rd.pth')
    # torch.save(model.state_dict(), 'CNN/CNN_earth50_every3rd_state_dict.pth')