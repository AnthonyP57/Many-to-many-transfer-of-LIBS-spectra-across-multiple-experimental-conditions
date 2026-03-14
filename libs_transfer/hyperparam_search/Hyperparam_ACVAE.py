from libs_transfer.training.models import Classifier, ACVAE, Encoder, Decoder
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from libs_transfer.training.modules import SpectraDataset, ClassDataset, ACVAE_test_spectra, train_test_spectra_samples
from torch.utils.data import  DataLoader
from libs_transfer.training.CNN_conc_baseline import CNN
from libs_transfer.hyperparam_search.PolyHyper import PolyHyper_search, Config
import json
import joblib
import pandas as pd
import copy

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0))
else:
    device = torch.device("cpu")
    print("using CPU, GPU is not available")

with h5py.File('new_data\\spectra.h5', 'r') as hf:
    wavelen = np.array(hf['calibration'][()])
    spectra_0_idx = np.where(wavelen == 250)[0][0] - 25
    spectra_last_idx = np.where(wavelen == 850)[0][0] + 25

    labels = np.array(hf['labels_one_hot'])
    all_spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx+1])

    all_atm = np.array(hf['atm_one_hot'][()])
    all_ene = np.array(hf['energy_one_hot'][()])

    conditions = np.array([np.kron(row1, row2) for row1, row2 in zip(all_atm, all_ene)])

    c = np.where(np.argmax(conditions, axis=1) != 0)[0] #no vacuum 50%
    all_spectra = all_spectra[c]
    labels = labels[c]
    conditions = conditions[c][:, [1,2,3]]

print(all_spectra.shape, conditions.shape, labels.shape)

torch.cuda.empty_cache()

all_spectra = np.maximum(all_spectra, 0) #relu

emis_norm = all_spectra / np.sum(all_spectra, axis=1, keepdims=True) # total emissivity normalization
emis_std = StandardScaler().fit(emis_norm)
del emis_norm

total_emis = np.expand_dims(np.sum(all_spectra, 1),1)

# zero_baseline = np.zeros((1,all_spectra.shape[1]))

std = StandardScaler().fit(all_spectra)
all_spectra = std.transform(all_spectra)

std_emis = StandardScaler().fit(total_emis)
total_emis = std_emis.transform(total_emis)

mean = torch.tensor(std.mean_).to(device)
std_ = torch.tensor(std.scale_).to(device)

train_idx, test_idx = train_test_spectra_samples(conditions, labels) # divide within samples -> test samples idx: [1, 7, 17, 40]
# train_idx, test_idx = train_test_spectra_idx(conditions, labels) #leave out samples

ds = SpectraDataset(all_spectra[train_idx], conditions[train_idx], labels[train_idx], total_emis[train_idx], batch_size=25)
ds.random_select_samples(shuffle_data_pairs=True)

class_dataset = DataLoader(ClassDataset(all_spectra[train_idx], conditions[train_idx]), batch_size=400, shuffle=True)

classifier = Classifier(n_classes=conditions.shape[1]).to(device)
clas_optim = torch.optim.Adam(classifier.parameters(), lr=5e-3)

# clas_sched = torch.optim.lr_scheduler.StepLR(clas_optim, step_size=2, gamma=0.2)
clas_sched = torch.optim.lr_scheduler.CosineAnnealingLR(clas_optim, T_max=5)

for i in range(0):
    epoch_loss = 0
    n_all = 0
    for x, y in class_dataset:
        x = x[:, 25:-25]
        x = x.unsqueeze(1)
        x = x.to(device)
        pred = classifier(x)
        pred = pred.permute(0, 2, 1).reshape(-1, pred.shape[1])
        y = torch.argmax(y, dim=1).cpu().numpy()[0]
        y = torch.LongTensor(y * np.ones(len(pred))).to(device)

        loss = F.cross_entropy(pred, y)
        clas_optim.zero_grad()
        loss.backward()
        clas_optim.step()

        n_all += x.shape[0]
        epoch_loss += loss * x.shape[0]
    clas_sched.step()
    print(f'Classifier pretraining epoch: {i+1} loss: {epoch_loss/n_all:.4f}')

# that clustering comes from optimize_k_means_concentration

with open('new_data/label_dict.json', 'r') as fp:
    reg_conditions_dict = json.load(fp)

conc_df = pd.read_excel(r'Regoliths/Concentrations.xlsx')
conc_df = conc_df.dropna().transpose()
conc_df.columns = [a.upper() for a in conc_df.iloc[0].tolist()]
conc_df = conc_df.iloc[1:]
conc_df.index = [a.upper() for a in conc_df.index]
elements = conc_df.columns.tolist()

std_v100 = joblib.load('CNN/conc_std_vacuum100.joblib')
std_e50 = joblib.load('CNN/conc_std_earth50.joblib')
std_e100 = joblib.load('CNN/conc_std_earth100.joblib')

vacuum100 = np.where(conditions[[test_idx], 0] == 1)[0]
earth50 = np.where(conditions[[test_idx], 1] == 1)[0]
earth100 = np.where(conditions[[test_idx], 2] == 1)[0]

vac100_to_earth_50, vacuum100_conc = ACVAE_test_spectra(reg_conditions_dict, conc_df, std_v100, labels[test_idx], all_spectra[test_idx], total_emis[test_idx], vacuum100, earth50, 0, 1)

vac100_to_earth_100 = ACVAE_test_spectra(reg_conditions_dict, conc_df, std_v100, labels[test_idx], all_spectra[test_idx], total_emis[test_idx], vacuum100, earth100, 0, 2)[0]

earth50_to_vacuum100, earth50_conc = ACVAE_test_spectra(reg_conditions_dict, conc_df, std_e50, labels[test_idx], all_spectra[test_idx], total_emis[test_idx], earth50, vacuum100, 1, 0)

earth50_to_earth100 = ACVAE_test_spectra(reg_conditions_dict, conc_df, std_e50, labels[test_idx], all_spectra[test_idx], total_emis[test_idx], earth50, earth100, 1, 2)[0]

earth100_to_vacuum100, earth100_conc = ACVAE_test_spectra(reg_conditions_dict, conc_df, std_e100, labels[test_idx], all_spectra[test_idx], total_emis[test_idx], earth100, vacuum100, 2, 0)

earth100_to_earth50 = ACVAE_test_spectra(reg_conditions_dict, conc_df, std_e100, labels[test_idx], all_spectra[test_idx], total_emis[test_idx], earth100, earth50, 2, 1)[0]

to_earth50_transform_in_data = [vac100_to_earth_50, earth100_to_earth50]
to_earth_50_conc = [vacuum100_conc, earth100_conc]

to_earth100_transform_in_data = [vac100_to_earth_100, earth50_to_earth100]
to_earth100_conc = [vacuum100_conc, earth50_conc]

to_vacuum100_transform_in_data = [earth100_to_vacuum100, earth50_to_vacuum100]
to_vacuum100_conc = [earth100_conc, earth50_conc]

# cnn_model = CNN(vacuum100_conc.shape[1])
# print(torch.load(r'CNN/CNN_vacuum100_state_dict.pth', weights_only=True))
model_v100 = CNN(vacuum100_conc.shape[1], 4996)
model_v100.load_state_dict(torch.load(r'CNN/CNN_vacuum100_every3rd_state_dict.pth', weights_only=True))
model_v100.to(device)

model_e50 = CNN(earth50_conc.shape[1], 4996)
model_e50.load_state_dict(torch.load(r'CNN/CNN_earth50_every3rd_state_dict.pth', weights_only=True))
model_e50.to(device)

model_e100 = CNN(earth100_conc.shape[1], 4996)
model_e100.load_state_dict(torch.load(r'CNN/CNN_earth100_every3rd_state_dict.pth', weights_only=True))
model_e100.to(device)

class Model:
    def __init__(self, config):
        self.config = config

        encoder = Encoder(n_classes_channels=1, in_channels=1, ks_lst=config['kernel_and_pad'][0], out_channels_lst=config['channels'], stride_lst=[1, 1, 1, 1, 1, 1, 1],
                            skip_pad=config['kernel_and_pad'][1])
        decoder = Decoder(n_classes_channels=1, in_channels=config['channels'][-1], ks_lst=config['kernel_and_pad'][0][::-1], out_channels_lst=config['channels'][::-1][1:] + [1],
                            stride_lst=[1, 1, 1, 1, 1, 1, 1], skip_pad=config['kernel_and_pad'][1])

        for p in encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.model = ACVAE(encoder, decoder, copy.deepcopy(classifier), mean, std_).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'], weight_decay=config['wd'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 10)

    def train(self):

        for x_0, x_1, y_0, y_1, e_0, e_1 in ds:

            x_0 = x_0.to(device)
            x_1 = x_1.to(device)
            y_0 = y_0.to(device)
            y_1 = y_1.to(device)
            e_0 = e_0.to(device)
            e_1 = e_1.to(device)

            kl_loss, gaus_neg_log_like, ClsLoss_class, ClsLoss_recon, ClsLoss_convert, loss_d = self.model.calc_loss(x_0, x_1, y_0, y_1, e_0, e_1, crop_by=25)

            vae_loss = gaus_neg_log_like + kl_loss + ClsLoss_class + ClsLoss_recon + ClsLoss_convert + loss_d * 1

            self.optimizer.zero_grad()
            vae_loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        ds.random_select_samples(shuffle_data_pairs=True)

        earth_50_transformed = []
        earth100_transformed = []
        vacuum100_transformed = []

        with torch.no_grad():
            for d in to_earth50_transform_in_data:
                a=[]
                for x_0, y_0, y_1, e_0, e_1 in d:
                    x_0 = x_0.to(device)
                    y_0 = y_0.to(device)
                    y_1 = y_1.to(device)
                    e_0 = e_0.to(device)
                    e_1 = e_1.to(device)
                    x_01 = self.model(x_0, y_0, y_1, e_0, e_1, crop_by=25)
                    x_01 = x_01.cpu().detach().numpy().reshape(x_01.shape[0], -1)
                    x_01 = np.maximum(std.inverse_transform(np.pad(x_01, ((0, 0), (25, 25)), mode='constant')), 0)
                    a.append(emis_std.transform(x_01 / np.sum(x_01, axis=1, keepdims=True))[:, 9:-9])

                earth_50_transformed.append(np.concatenate(a, axis=0))

            for d in to_earth100_transform_in_data:
                a = []
                for x_0, y_0, y_1, e_0, e_1 in d:
                    x_0 = x_0.to(device)
                    y_0 = y_0.to(device)
                    y_1 = y_1.to(device)
                    e_0 = e_0.to(device)
                    e_1 = e_1.to(device)
                    x_01 = self.model(x_0, y_0, y_1, e_0, e_1, crop_by=25)
                    x_01 = x_01.cpu().detach().numpy().reshape(x_01.shape[0], -1)
                    x_01 = np.maximum(std.inverse_transform(np.pad(x_01, ((0, 0), (25, 25)), mode='constant')), 0)
                    a.append(emis_std.transform(x_01 / np.sum(x_01, axis=1, keepdims=True))[:, 9:-9])

                earth100_transformed.append(np.concatenate(a, axis=0))

            for d in to_vacuum100_transform_in_data:
                a = []
                for x_0, y_0, y_1, e_0, e_1 in d:
                    x_0 = x_0.to(device)
                    y_0 = y_0.to(device)
                    y_1 = y_1.to(device)
                    e_0 = e_0.to(device)
                    e_1 = e_1.to(device)
                    x_01 = self.model(x_0, y_0, y_1, e_0, e_1, crop_by=25)
                    x_01 = x_01.cpu().detach().numpy().reshape(x_01.shape[0], -1)
                    x_01 = np.maximum(std.inverse_transform(np.pad(x_01, ((0, 0), (25, 25)), mode='constant')), 0)
                    a.append(emis_std.transform(x_01 / np.sum(x_01, axis=1, keepdims=True))[:, 9:-9])

                vacuum100_transformed.append(np.concatenate(a, axis=0))

        running_loss_ = 0
        n_all_ = 0

        with torch.no_grad():
            preds = []
            ys = []
            for d, l, m, s in zip([earth_50_transformed, earth100_transformed, vacuum100_transformed],
                               [to_earth_50_conc, to_earth100_conc, to_earth100_conc], [model_e50, model_e100, model_v100], [std_e50, std_e100, std_v100]):

                for d_, l_ in zip(d,l):
                    test_dataloader = DataLoader(ClassDataset(d_, l_), batch_size=25)


                    for i, (x, y) in enumerate(test_dataloader):
                        x = x.to(device)
                        y = y.to(device)
                        pred = m(x)
                        loss = F.mse_loss(pred, y)

                        running_loss_ += loss.item()*y.size(0)
                        n_all_ += y.size(0)

                        preds.append(s.inverse_transform(pred.cpu().detach().numpy()))
                        ys.append(s.inverse_transform(y.cpu().detach().numpy()))

        rmse = np.round(np.mean(np.sqrt(np.mean((np.array(ys) - np.array(preds))**2, axis=0)), axis=0), 2)
        print(f'\n\tRMSE: {rmse}')

        conc_loss = running_loss_ / n_all_

        total_loss = conc_loss

        return total_loss

kernel_and_pad = Config([[[16,16,16,16,16,16,16], 6], [[8,8,8,8,8,8,8], 2]], 'choice', 'kernel_and_pad')

channels = Config([[8,8,8,8,8,8,8], [8,8,8,8,4,4,4], [4,4,4,4,8,8,8]], 'choice', 'channels')

lr = Config([1e-6, 1e-3], 'continuous', 'lr')

w_decay = Config([1e-7, 1e-4], 'continuous', 'wd')

PolyHyper_search(Model, kernel_and_pad, channels, lr, w_decay, initial_search=10, n_probes=10, tolerance=3, max_epoch=10)