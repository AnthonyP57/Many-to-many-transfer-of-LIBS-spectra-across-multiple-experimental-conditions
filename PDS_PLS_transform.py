import pycaltransfer.caltransfer as caltransfer
import h5py
import numpy as np
from modules import SpectraDataset, ClassDataset, PDS_test_spectra, train_test_spectra_idx, train_test_spectra_samples, PDSDataset
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import joblib
from torch.utils.data import  DataLoader
from CNN_conc_baseline import CNN
import json
from tqdm import tqdm
from sklearn.metrics import r2_score

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0))
else:
    device = torch.device("cpu")
    print("using CPU, GPU is not available")

with h5py.File('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/new_data/spectra.h5', 'r') as hf:
    wavelen = np.array(hf['calibration'][()])
    spectra_0_idx = np.where(wavelen == 250)[0][0]
    spectra_last_idx = np.where(wavelen == 850)[0][0]

    labels = np.array(hf['labels_one_hot'])
    all_spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx+1])

    all_atm = np.array(hf['atm_one_hot'][()])
    all_ene = np.array(hf['energy_one_hot'][()])

    conditions = np.array([np.kron(row1, row2) for row1, row2 in zip(all_atm, all_ene)])

    c = np.where(np.argmax(conditions, axis=1) != 0)[0] #no vacuum 50%
    all_spectra = all_spectra[c]
    labels = labels[c]
    conditions = conditions[c][:, [1,2,3]]

region_dict = {str(i) : v for i, v in enumerate(['vacuum 100%', 'earth 50%', 'earth 100%'])}

with open('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/new_data/label_dict.json', 'r') as fp:
    reg_conditions_dict = json.load(fp)

print(all_spectra.shape, conditions.shape, labels.shape)

emis_norm = all_spectra / np.sum(all_spectra, axis=1, keepdims=True) # total emissivity normalization
emis_std = StandardScaler().fit(emis_norm)
del emis_norm

torch.cuda.empty_cache()

# all_spectra = all_spectra / np.sum(all_spectra, axis=1, keepdims=True) # total emissivity normalization

all_spectra = np.maximum(all_spectra, 0) #relu
total_emis = np.expand_dims(np.sum(all_spectra, 1),1)

std = StandardScaler().fit(all_spectra)
all_spectra = std.transform(all_spectra)

std_emis = MinMaxScaler().fit(total_emis)
total_emis = std_emis.transform(total_emis)

# zero_baseline = torch.tensor(std.transform(zero_baseline), dtype=torch.float32)[:, 25:-25]

train_idx, test_idx = train_test_spectra_samples(conditions, labels) # divide within samples -> test samples idx: [1, 7, 17, 40]
# train_idx, test_idx = train_test_spectra_idx(conditions, labels) #leave out samples

batch_size=50
segment_size=20

# ds = SpectraDataset(all_spectra[train_idx], conditions[train_idx], labels[train_idx], total_emis[train_idx], batch_size=batch_size)
# ds.random_select_samples(shuffle_data_pairs=True)

ds = PDSDataset(all_spectra[train_idx], conditions[train_idx], labels[train_idx])

conc_df = pd.read_excel(r'/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/Regoliths/Concentrations.xlsx')
conc_df = conc_df.dropna().transpose()
conc_df.columns = [a.upper() for a in conc_df.iloc[0].tolist()]
conc_df = conc_df.iloc[1:]
conc_df.index = [a.upper() for a in conc_df.index]
elements = conc_df.columns.tolist()

std_v100 = joblib.load('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/CNN/conc_std_vacuum100.joblib')
std_e50 = joblib.load('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/CNN/conc_std_earth50.joblib')
std_e100 = joblib.load('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/CNN/conc_std_earth100.joblib')

vacuum100 = np.where(np.argmax(conditions[test_idx], axis=1) == 0)[0]
earth50 = np.where(np.argmax(conditions[test_idx], axis=1) == 1)[0]
earth100 = np.where(np.argmax(conditions[test_idx], axis=1) == 2)[0]

vacuum100_test, vacuum100_conc = PDS_test_spectra(reg_conditions_dict, conc_df, std_v100, labels[test_idx], all_spectra[test_idx], vacuum100)

earth50_test, earth50_conc = PDS_test_spectra(reg_conditions_dict, conc_df, std_e50, labels[test_idx], all_spectra[test_idx], earth50)

earth100_test, earth100_conc = PDS_test_spectra(reg_conditions_dict, conc_df, std_e100, labels[test_idx], all_spectra[test_idx], earth100)

model_v100 = CNN(vacuum100_conc.shape[1])
model_v100.load_state_dict(torch.load(r'/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/CNN/CNN_vacuum100_state_dict.pth', weights_only=True))
model_v100.to(device)

model_e50 = CNN(earth50_conc.shape[1])
model_e50.load_state_dict(torch.load(r'/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/CNN/CNN_earth50_state_dict.pth', weights_only=True))
model_e50.to(device)

model_e100 = CNN(earth100_conc.shape[1])
model_e100.load_state_dict(torch.load(r'/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/CNN/CNN_earth100_state_dict.pth', weights_only=True))
model_e100.to(device)

preds_01=[]
ys_01=[]
preds_01_cnn=[]
preds_10=[]
ys_10=[]
preds_10_cnn=[]

iterator = tqdm(ds, desc='PDS')
# for x_0, x_1, y_0, y_1, e_0, e_1 in iterator:
for x_0, x_1, y_0, y_1 in iterator:

    x_0 = x_0.detach().numpy()
    x_1 = x_1.detach().numpy()

    # if torch.argmax(y_0, axis=1)[0] == 0:
    if y_0 == 0:
        _0 = 0
        x_0_valid = vacuum100_test
        valid_10 = model_v100
        std_10 = std_v100
        conc_10 = vacuum100_conc
    # elif torch.argmax(y_0, axis=1)[0] == 1:
    elif y_0 == 1:
        _0 = 1
        x_0_valid = earth50_test
        valid_10 = model_e50
        std_10 = std_e50
        conc_10 = earth50_conc
    # elif torch.argmax(y_0, axis=1)[0] == 2:
    elif y_0 == 2:
        _0 = 2
        x_0_valid = earth100_test
        valid_10 = model_e100
        std_10 = std_e100
        conc_10 = earth100_conc

    # if torch.argmax(y_1, axis=1)[0] == 0:
    if y_1 == 0:
        _1 = 0
        x_1_valid = vacuum100_test
        valid_01 = model_v100
        std_01 = std_v100
        conc_01 = vacuum100_conc
    # elif torch.argmax(y_1, axis=1)[0] == 1:
    elif y_1 == 1:
        _1 = 1
        x_1_valid = earth50_test
        valid_01 = model_e50
        std_01 = std_e50
        conc_01 = earth50_conc
    # elif torch.argmax(y_1, axis=1)[0] == 2:
    elif y_1 == 2:
        _1 = 2
        x_1_valid = earth100_test
        valid_01 = model_e100
        std_01 = std_e100
        conc_01 = earth100_conc

    F, a = caltransfer.pds_pls_transfer_fit(x_1, x_0, max_ncp = 10, ww = 10)
    x_01 = x_0_valid.dot(F) + a
    del F,a
    x_01 = np.maximum(std.inverse_transform(x_01), 0) #relu
    x_01 = emis_std.transform(x_01 / np.sum(x_01, axis=1, keepdims=True))

    F, a = caltransfer.pds_pls_transfer_fit(x_0, x_1, max_ncp = 10, ww = 10)
    x_10 = x_1_valid.dot(F) + a
    del F,a
    x_10 = np.maximum(std.inverse_transform(x_10), 0) #relu
    x_10 = emis_std.transform(x_10 / np.sum(x_10, axis=1, keepdims=True))

    with torch.no_grad():
        preds = []
        ys = []
        test_dataloader = DataLoader(ClassDataset(x_01, conc_01), batch_size=25)

        cnn_pred_spectra = std.inverse_transform(x_0_valid)
        cnn_pred_spectra = emis_std.transform(cnn_pred_spectra / np.sum(cnn_pred_spectra, axis=1, keepdims=True))

        cnn_preds = std_10.inverse_transform(valid_10(torch.tensor(cnn_pred_spectra).to(device)).cpu().detach().numpy())

        for i, (x, y) in enumerate(test_dataloader):
            x = x.to(device)
            y = y.to(device)
            pred = valid_01(x)
            preds.append(std_01.inverse_transform(pred.cpu().detach().numpy()))
            ys.append(std_01.inverse_transform(y.cpu().detach().numpy()))

    rmse = np.sqrt(np.mean((np.concatenate(ys, axis=0) - np.concatenate(preds, axis=0)) ** 2, axis=0))
    preds_01.append(np.concatenate(preds, axis=0))
    ys_01.append(np.concatenate(ys, axis=0))
    rmse_c = np.sqrt(np.mean((cnn_preds - np.concatenate(preds, axis=0)) ** 2, axis=0))
    preds_01_cnn.append(cnn_preds)

    del preds, ys, cnn_preds

    with torch.no_grad():
        preds = []
        ys = []
        test_dataloader = DataLoader(ClassDataset(x_10, conc_10), batch_size=25)

        cnn_pred_spectra = std.inverse_transform(x_1_valid)
        cnn_pred_spectra = emis_std.transform(cnn_pred_spectra / np.sum(cnn_pred_spectra, axis=1, keepdims=True))

        cnn_preds = std_01.inverse_transform(valid_01(torch.tensor(cnn_pred_spectra).to(device)).cpu().detach().numpy())

        for i, (x, y) in enumerate(test_dataloader):
            x = x.to(device)
            y = y.to(device)
            pred = valid_10(x)
            preds.append(std_10.inverse_transform(pred.cpu().detach().numpy()))
            ys.append(std_10.inverse_transform(y.cpu().detach().numpy()))

    rmse_ = np.sqrt(np.mean((np.concatenate(ys, axis=0) - np.concatenate(preds, axis=0)) ** 2, axis=0))
    preds_10.append(np.concatenate(preds, axis=0))
    ys_10.append(np.concatenate(ys, axis=0))
    rmse_c_ = np.sqrt(np.mean((cnn_preds - np.concatenate(preds, axis=0)) ** 2, axis=0))
    preds_10_cnn.append(cnn_preds)

    # iterator.write(f'\n0 : {"v100" if _0 == 0 else "e50" if _0 == 1 else "e100"}\t\t1 : {"v100" if _1 == 0 else "e50" if _1 == 1 else "e100"}')

    iterator.set_postfix({'ave 0->1 RMSE': str(np.mean(rmse)),
                          'ave 0->1 RMSE CNN': str(np.mean(rmse_c)),
                          'ave 1->0 RMSE': str(np.mean(rmse_)),
                          'ave 1->0 RMSE CNN': str(np.mean(rmse_c_))})

print(f'ave total 0->1 RMSE: {np.round(np.sqrt(np.mean((np.concatenate(ys_01, axis=0) - np.concatenate(preds_01, axis=0)) ** 2, axis=0)), 2)}')
print(f'ave total 1->0 RMSE: {np.round(np.sqrt(np.mean((np.concatenate(ys_10, axis=0) - np.concatenate(preds_10, axis=0)) ** 2, axis=0)), 2)}')
print(f'total: {np.round(np.sqrt(np.mean((np.concatenate([
                                                        np.concatenate(ys_01, axis=0),
                                                        np.concatenate(ys_10, axis=0)], axis=0)
                                                        -
                                                        np.concatenate([
                                                        np.concatenate(preds_01, axis=0),
                                                        np.concatenate(preds_10, axis=0)], axis=0)) ** 2, axis=0)), 2)}')

print(f'total ave: {np.round(np.mean(np.sqrt(np.mean((np.concatenate([
                                                        np.concatenate(ys_01, axis=0),
                                                        np.concatenate(ys_10, axis=0)], axis=0)
                                                        -
                                                        np.concatenate([
                                                        np.concatenate(preds_01, axis=0),
                                                        np.concatenate(preds_10, axis=0)], axis=0)) ** 2, axis=0))), 2)}')

print(f'ave total 0->1 RMSE CNN: {np.round(np.sqrt(np.mean((np.concatenate(ys_01, axis=0) - np.concatenate(preds_01_cnn, axis=0)) ** 2, axis=0)), 2)}')
print(f'ave total 1->0 RMSE: {np.round(np.sqrt(np.mean((np.concatenate(ys_10, axis=0) - np.concatenate(preds_10_cnn, axis=0)) ** 2, axis=0)), 2)}')
print(f'total: {np.round(np.sqrt(np.mean((np.concatenate([
                                                        np.concatenate(ys_01, axis=0),
                                                        np.concatenate(ys_10, axis=0)], axis=0)
                                                        -
                                                        np.concatenate([
                                                        np.concatenate(preds_01_cnn, axis=0),
                                                        np.concatenate(preds_10_cnn, axis=0)], axis=0)) ** 2, axis=0)), 2)}')

print(f'total ave: {np.round(np.mean(np.sqrt(np.mean((np.concatenate([
                                                        np.concatenate(ys_01, axis=0),
                                                        np.concatenate(ys_10, axis=0)], axis=0)
                                                        -
                                                        np.concatenate([
                                                        np.concatenate(preds_01_cnn, axis=0),
                                                        np.concatenate(preds_10_cnn, axis=0)], axis=0)) ** 2, axis=0))), 2)}')

print(f'total R2 true: {r2_score(np.concatenate([
    np.concatenate(ys_01, axis=0),
    np.concatenate(ys_10, axis=0)], axis=0).reshape(-1),
    np.concatenate([
    np.concatenate(preds_01, axis=0),
    np.concatenate(preds_10, axis=0)], axis=0).reshape(-1))}')

print(f'total R2 CNN: {r2_score(np.concatenate([
    np.concatenate(ys_01, axis=0),
    np.concatenate(ys_10, axis=0)], axis=0).reshape(-1),    
    np.concatenate([
    np.concatenate(preds_01_cnn, axis=0),
    np.concatenate(preds_10_cnn, axis=0)], axis=0).reshape(-1))}')