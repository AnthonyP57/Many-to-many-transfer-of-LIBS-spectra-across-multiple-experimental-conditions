from models import Classifier, ACVAE, Encoder, Decoder#, CNN_d
import torch
from modules import SpectraDataset, ClassDataset, train_test_spectra_idx, train_test_spectra_samples, ACVAE_test_spectra, ModelConfig, divide_into_sublists, create_random_combined_list
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import torch.nn as nn
from torch.utils.data import  DataLoader
import torch.nn.functional as F
import pandas as pd
import joblib
from libs_transfer.training.CNN_conc_baseline import CNN
import json
from tqdm import tqdm
import warnings
import copy
from kmeans_loss_mean import calc_kmeans_loss_mean_spectra
from kmeans_loss import calc_kmeans_loss
from consine_loss_mean import cosine_similarity_mean_spectra
from consine_loss import cosine_similarity
import argparse
from sklearn.metrics import r2_score
# from preprocessing_LIBS import correct_baseline, apply_multiprocessing_along_axis

def int_or_list(val):
    if isinstance(val, int):
        return [val]
    else:
        try:
            return list(map(int, val.split(',')))
        except:
            raise argparse.ArgumentTypeError("Expected an integer or comma separated list of integers e.g. 1,2,3")

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--random_seed', type=int_or_list, default=[123456, 42, 123, 835, 297])
parser.add_argument('-d', '--device', type=str, default='cuda:0')
parser.add_argument('-e', '--epochs', type=int, default=5)
parser.add_argument('-p', '--pretrain', type=bool, default=True)
parser.add_argument('--path', type=str, default='/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/new_data/metrics')
args = parser.parse_args()

path_metrics = args.path
epochs_train = args.epochs
pretrain_ = args.pretrain
seeds = args.random_seed

warnings.filterwarnings("ignore")

if args.device == 'cuda:0':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(torch.cuda.get_device_name(0))
        print(torch.cuda.get_device_properties(0))
    else:
        device = torch.device("cpu")
        print("using CPU, GPU is not available")

# device = torch.device("cpu")

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

"""After PCA distance filtering"""

# with h5py.File('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/new_data/pca_inliers_spectra.h5', 'r') as hf:
#     wavelen = np.array(hf['calibration'][()])
#     spectra_0_idx = np.where(wavelen == 250)[0][0] - 25
#     spectra_last_idx = np.where(wavelen == 850)[0][0] + 25

#     labels = np.array(hf['labels'])
#     all_spectra = np.array(hf['spectra']) # already in the correct range

#     # all_spectra = apply_multiprocessing_along_axis(correct_baseline, all_spectra) #baseline correction

#     conditions = np.array(hf['conditions'])

"""No preprocessing"""

with h5py.File('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/new_data/spectra.h5', 'r') as hf:
    wavelen = np.array(hf['calibration'][()])
    spectra_0_idx = np.where(wavelen == 250)[0][0] - 25
    spectra_last_idx = np.where(wavelen == 850)[0][0] + 25

    labels = np.array(hf['labels_one_hot'])
    all_spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx+1])

    # all_spectra = apply_multiprocessing_along_axis(correct_baseline, all_spectra) #baseline correction

    all_atm = np.array(hf['atm_one_hot'][()])
    all_ene = np.array(hf['energy_one_hot'][()])

    conditions = np.array([np.kron(row1, row2) for row1, row2 in zip(all_atm, all_ene)])

    c = np.where(np.argmax(conditions, axis=1) != 0)[0] #no vacuum 50%
    all_spectra = all_spectra[c]
    labels = labels[c]
    conditions = conditions[c][:, [1,2,3]]

print(all_spectra.shape, conditions.shape, labels.shape)

torch.cuda.empty_cache()

emis_norm = all_spectra / np.sum(all_spectra, axis=1, keepdims=True) # total emissivity normalization
emis_std = StandardScaler().fit(emis_norm)
del emis_norm

# all_spectra = all_spectra / np.sum(all_spectra, axis=1, keepdims=True) # total emissivity normalization

all_spectra = np.maximum(all_spectra, 0) #relu
total_emis = np.expand_dims(np.sum(all_spectra, 1),1)

# zero_baseline = np.zeros((1,all_spectra.shape[1]))

std = StandardScaler().fit(all_spectra)
all_spectra = std.transform(all_spectra)

std_emis = MinMaxScaler().fit(total_emis)
total_emis = std_emis.transform(total_emis)

# zero_baseline = torch.tensor(std.transform(zero_baseline), dtype=torch.float32)[:, 25:-25]

mean = torch.tensor(std.mean_).to(device)
std_ = torch.tensor(std.scale_).to(device)

train_idx, test_idx = train_test_spectra_samples(conditions, labels) # divide within samples -> test samples idx: [1, 7, 17, 40]
# train_idx, test_idx = train_test_spectra_idx(conditions, labels) #leave out samples

ds = SpectraDataset(all_spectra[train_idx], conditions[train_idx], labels[train_idx], total_emis[train_idx], batch_size=70)
ds.random_select_samples(shuffle_data_pairs=True)

with open('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/new_data/label_dict.json', 'r') as fp:
    reg_conditions_dict = json.load(fp)

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

vac100_to_earth_50, vacuum100_conc, vac100_labels = ACVAE_test_spectra(reg_conditions_dict, conc_df, std_v100, labels[test_idx], all_spectra[test_idx], total_emis[test_idx], vacuum100, earth50, 0, 1)
vac100_to_earth_100 = ACVAE_test_spectra(reg_conditions_dict, conc_df, std_v100, labels[test_idx], all_spectra[test_idx], total_emis[test_idx], vacuum100, earth100, 0, 2)[0]
earth50_to_vacuum100, earth50_conc, earth50_labels = ACVAE_test_spectra(reg_conditions_dict, conc_df, std_e50, labels[test_idx], all_spectra[test_idx], total_emis[test_idx], earth50, vacuum100, 1, 0)
earth50_to_earth100 = ACVAE_test_spectra(reg_conditions_dict, conc_df, std_e50, labels[test_idx], all_spectra[test_idx], total_emis[test_idx], earth50, earth100, 1, 2)[0]
earth100_to_vacuum100, earth100_conc, earth100_labels = ACVAE_test_spectra(reg_conditions_dict, conc_df, std_e100, labels[test_idx], all_spectra[test_idx], total_emis[test_idx], earth100, vacuum100, 2, 0)
earth100_to_earth50 = ACVAE_test_spectra(reg_conditions_dict, conc_df, std_e100, labels[test_idx], all_spectra[test_idx], total_emis[test_idx], earth100, earth50, 2, 1)[0]

to_earth50_transform_in_data = [vac100_to_earth_50, earth100_to_earth50]
to_earth_50_conc = [vacuum100_conc, earth100_conc]

to_earth100_transform_in_data = [vac100_to_earth_100, earth50_to_earth100]
to_earth100_conc = [vacuum100_conc, earth50_conc]

to_vacuum100_transform_in_data = [earth100_to_vacuum100, earth50_to_vacuum100]
to_vacuum100_conc = [earth100_conc, earth50_conc]

i_dict = {
    0: 'v100 to e50',
    1: 'e100 to e50',
    2: 'v100 to e100',
    3: 'e50 to e100',
    4: 'e100 to v100',
    5: 'e50 to v100'
}

model_v100 = CNN(vacuum100_conc.shape[1])
model_v100.load_state_dict(torch.load(r'/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/CNN/CNN_vacuum100_state_dict.pth', weights_only=True))
model_v100.to(device)

model_e50 = CNN(earth50_conc.shape[1])
model_e50.load_state_dict(torch.load(r'/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/CNN/CNN_earth50_state_dict.pth', weights_only=True))
model_e50.to(device)

model_e100 = CNN(earth100_conc.shape[1])
model_e100.load_state_dict(torch.load(r'/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/CNN/CNN_earth100_state_dict.pth', weights_only=True))
model_e100.to(device)

config = ModelConfig(checkpoint_path='/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/models/checkpoints/',
                     lr=5e-2, wd=1e-6, ks_list=[16]*7, stride_list=[1]*7, channel_list=[8]*7, skip_pad=6, resume=False, fc=False, pretrain=pretrain_)

class_dataset = DataLoader(ClassDataset(all_spectra[train_idx], conditions[train_idx]), batch_size=250, shuffle=True)

elements_ = ['SIO2','AL2O3','CAO','FE2O3','K2O','MGO','MNO','TIO2']

metrics = pd.DataFrame(index=[f'seed {s} epoch {e}' for s in seeds for e in range(epochs_train)],
                       columns=['concentration RMSE'] + [f'{e} RMSE' for e in elements_] + ['conc CNN RMSE'] + 
                       [f'{e} CNN RMSE' for e in elements_] + [f'{m} {i}' for i in i_dict.values() for m in ['RMSE', 'CNN RMSE']] + # wrong names for the transfers
                       ['clustering acc mean spectra', 'cos sim mean spectra', 'clustering acc', 'cos sim', 'r2', 'r2 cnn'])

for _seed in seeds:
    torch.cuda.empty_cache()
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    print(f'seed: {_seed}')

    classifier = Classifier(n_classes=conditions.shape[1]).to(device)
    clas_optim = torch.optim.Adam(classifier.parameters(), lr=5e-3)
    clas_sched = torch.optim.lr_scheduler.CosineAnnealingLR(clas_optim, T_max=5)


    if not hasattr(config, 'loaded_checkpoint') and config.pretrain:
        for i in range(5):
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

        del x,pred,y,loss

    torch.cuda.empty_cache()

    encoder = Encoder(n_classes_channels=1, in_channels=1, ks_lst=config.ks_list, out_channels_lst=config.channel_list, stride_lst=config.stride_list, skip_pad=config.skip_pad, add_fc=config.fc)
    decoder = Decoder(n_classes_channels=1, in_channels=config.channel_list[-1], ks_lst=config.ks_list[::-1], out_channels_lst=config.channel_list[:-1][1:]+[1], stride_lst=config.stride_list[::-1], skip_pad=config.skip_pad, add_fc=config.fc)

    for p in encoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    acvae = ACVAE(encoder, decoder, classifier, mean, std_, True).to(device)
    acvae = torch.compile(acvae)

    optimizer = torch.optim.Adam(acvae.parameters(), lr=config.lr, weight_decay=config.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3)

    start = time.time()
    epochs=epochs_train
    iterator = tqdm(range(epochs), desc='model training')

    for epoch in iterator:
        metrics_epoch=[]
        torch.cuda.empty_cache()

        running_kl_loss = 0
        running_gaus_loss = 0
        running_clas_loss = 0
        running_recon_loss = 0
        running_convert_loss = 0
        running_d_loss = 0
        n_all = 0
        running_fake_loss = 0
        running_real_loss = 0
        fake_n = 0
        real_n = 0
        running_acvae_fake_loss = 0
        n_ = 0

        for x_0, x_1, y_0, y_1, e_0, e_1 in ds:
            acvae.train()

            if epoch == epochs-1:
                x_0__ = copy.deepcopy(x_0)
                x_1__ = copy.deepcopy(x_1)
                y_0__ = copy.deepcopy(y_0)
                y_1__ = copy.deepcopy(y_1)
                e_0__ = copy.deepcopy(e_0)
                e_1__ = copy.deepcopy(e_1)

            with torch.autocast(device_type="cuda:0", dtype=torch.bfloat16):

                x_0 = x_0.to(device)
                x_1 = x_1.to(device)
                y_0 = y_0.to(device)
                y_1 = y_1.to(device)
                e_0 = e_0.to(device)
                e_1 = e_1.to(device)

                kl_loss, gaus_neg_log_like, ClsLoss_class, ClsLoss_recon, ClsLoss_convert, loss_d = acvae.calc_loss(x_0, x_1, y_0, y_1, e_0, e_1, crop_by=25)

                vae_loss = gaus_neg_log_like + kl_loss + ClsLoss_class \
                    + ClsLoss_recon + ClsLoss_convert + loss_d
                

                a = x_0.shape[0]
                running_kl_loss += kl_loss*a
                running_gaus_loss += gaus_neg_log_like*a
                running_clas_loss += ClsLoss_class*a
                running_recon_loss += ClsLoss_recon*a
                running_convert_loss += ClsLoss_convert*a
                running_d_loss += loss_d*a
                n_all += a


            optimizer.zero_grad()
            vae_loss.backward()
            optimizer.step()

        scheduler.step()

        losses= {'kl_loss': f'{running_kl_loss/n_all:.5f}',
            'gauss_neg_log_like': f'{running_gaus_loss/n_all:.5f}',
            'ClsLoss_recon': f'{running_recon_loss/n_all:.5f}',
            'ClsLoss_convert': f'{running_convert_loss/n_all:.5f}',
            'ClsLoss_class': f'{running_clas_loss/n_all/3:.5f}',
            'loss_d': f'{running_d_loss/n_all:.5f}',
        }

        ds.random_select_samples(shuffle_data_pairs=True)
        iterator.set_postfix(losses)

        earth_50_transformed = []
        earth50_preds = []
        c_earth_50_transformed = []
        earth100_transformed = []
        earth100_preds = []
        c_earth100_transformed = []
        vacuum100_transformed = []
        vacuum100_preds = []
        c_vacuum100_transformed = []
        i_s=[]

        acvae.eval()
        with torch.no_grad():
            for i, (d, m, s) in enumerate(zip(to_earth50_transform_in_data, [model_v100, model_e100], [std_v100, std_e100])):
                a = []
                b = []
                for x_0, y_0, y_1, e_0, e_1 in d:
                    x_0 = x_0.to(device)
                    y_0 = y_0.to(device)
                    y_1 = y_1.to(device)
                    e_0 = e_0.to(device)
                    e_1 = e_1.to(device)
                    x_01 = acvae(x_0, y_0, y_1, e_0, e_1, crop_by=25)
                    x_01 = x_01.cpu().detach().numpy().reshape(x_01.shape[0], -1)
                    x_01 = np.maximum(std.inverse_transform(np.pad(x_01, ((0, 0), (25, 25)), mode='constant')), 0)
                    b.append(x_01[:, 25:-25])
                    x_01 = emis_std.transform(x_01 / np.sum(x_01, axis=1, keepdims=True))[:, 25:-25]
                    a.append(x_01.reshape(-1, x_01.shape[-1]))

                    x_0 = std.inverse_transform(x_0.cpu().detach())
                    x_0 = emis_std.transform(x_0 / np.sum(x_0, axis=1, keepdims=True))[:, 25:-25]
                    earth50_preds.append(s.inverse_transform(m(torch.tensor(x_0, dtype=torch.float).to(device)).cpu().detach().numpy()))

                    for _ in range(x_0.shape[0]):
                        i_s.append(i)

                earth_50_transformed.append(np.concatenate(a, axis=0))
                c_earth_50_transformed.append(np.concatenate(b, axis=0))

            for i, (d, m, s) in enumerate(zip(to_earth100_transform_in_data, [model_v100, model_e100], [std_v100, std_e100])):
                a = []
                b = []
                for x_0, y_0, y_1, e_0, e_1 in d:
                    x_0 = x_0.to(device)
                    y_0 = y_0.to(device)
                    y_1 = y_1.to(device)
                    e_0 = e_0.to(device)
                    e_1 = e_1.to(device)
                    x_01 = acvae(x_0, y_0, y_1, e_0, e_1, crop_by=25)
                    x_01 = x_01.cpu().detach().numpy().reshape(x_01.shape[0], -1)
                    x_01 = np.maximum(std.inverse_transform(np.pad(x_01, ((0, 0), (25, 25)), mode='constant')), 0)
                    b.append(x_01[:, 25:-25])
                    x_01 = emis_std.transform(x_01 / np.sum(x_01, axis=1, keepdims=True))[:, 25:-25]
                    a.append(x_01.reshape(-1, x_01.shape[-1]))

                    x_0 = std.inverse_transform(x_0.cpu().detach())
                    x_0 = emis_std.transform(x_0 / np.sum(x_0, axis=1, keepdims=True))[:, 25:-25]
                    earth100_preds.append(s.inverse_transform(m(torch.tensor(x_0, dtype=torch.float).to(device)).cpu().detach().numpy()))

                    for _ in range(x_0.shape[0]):
                        i_s.append(i + 2)

                earth100_transformed.append(np.concatenate(a, axis=0))
                c_earth100_transformed.append(np.concatenate(b, axis=0))

            for i, (d, m, s) in enumerate(zip(to_vacuum100_transform_in_data, [model_e100, model_e50], [std_e100, std_e50])):
                a = []
                b = []
                for x_0, y_0, y_1, e_0, e_1 in d:
                    x_0 = x_0.to(device)
                    y_0 = y_0.to(device)
                    y_1 = y_1.to(device)
                    e_0 = e_0.to(device)
                    e_1 = e_1.to(device)
                    x_01 = acvae(x_0, y_0, y_1, e_0, e_1, crop_by=25)
                    x_01 = x_01.cpu().detach().numpy().reshape(x_01.shape[0], -1)
                    x_01 = np.maximum(std.inverse_transform(np.pad(x_01, ((0, 0), (25, 25)), mode='constant')), 0)
                    b.append(x_01[:, 25:-25])
                    x_01 = emis_std.transform(x_01 / np.sum(x_01, axis=1, keepdims=True))[:, 25:-25]
                    a.append(x_01.reshape(-1, x_01.shape[-1]))

                    x_0 = std.inverse_transform(x_0.cpu().detach())
                    x_0 = emis_std.transform(x_0 / np.sum(x_0, axis=1, keepdims=True))[:, 25:-25]
                    vacuum100_preds.append(s.inverse_transform(m(torch.tensor(x_0, dtype=torch.float).to(device)).cpu().detach().numpy()))

                    for _ in range(x_0.shape[0]):
                        i_s.append(i + 4)

                vacuum100_transformed.append(np.concatenate(a, axis=0))
                c_vacuum100_transformed.append(np.concatenate(b, axis=0))

            running_loss_ = 0
            n_all_ = 0

            preds = []
            ys = []
            for d, l, m, s in zip([earth_50_transformed, earth100_transformed, vacuum100_transformed],
                                [to_earth_50_conc, to_earth100_conc, to_earth100_conc], [model_e50, model_e100, model_v100], [std_e50, std_e100, std_v100]):

                for d_, l_ in zip(d, l):
                    test_dataloader = DataLoader(ClassDataset(d_, l_), batch_size=25)

                    for i, (x, y) in enumerate(test_dataloader):
                        x = x.to(device)
                        y = y.to(device)
                        pred = m(x)
                        loss = F.mse_loss(pred, y)

                        running_loss_ += loss.item() * y.size(0)
                        n_all_ += y.size(0)

                        preds.append(s.inverse_transform(pred.cpu().detach().numpy()))
                        ys.append(s.inverse_transform(y.cpu().detach().numpy()))


        ys = np.concatenate(ys, axis=0)
        preds = np.concatenate(preds, axis=0)
        cnn_preds = np.concatenate([
            np.concatenate(earth50_preds, axis=0),
            np.concatenate(earth100_preds, axis=0),
            np.concatenate(vacuum100_preds, axis=0)
        ], axis=0)

        rmse = np.round(np.sqrt(np.mean((ys - preds) ** 2, axis=0)), 3)
        rmse_cnn = np.round(np.sqrt(np.mean((cnn_preds - preds) ** 2, axis=0)), 3)
        iterator.write(f'\nRMSE altogether: {rmse}')
        iterator.write(f'RMSE mean: {round(np.mean(rmse), 3):.3f}\n')
        iterator.write(f'\nRMSE altogether CNN: {rmse_cnn}')
        iterator.write(f'RMSE mean CNN: {round(np.mean(rmse_cnn), 3):.3f}\n')

        metrics_epoch.append(np.mean(rmse).tolist())
        metrics_epoch.extend(rmse.tolist())
        metrics_epoch.append(np.mean(rmse_cnn).tolist())
        metrics_epoch.extend(rmse_cnn.tolist())

        sublists, vals = divide_into_sublists(i_s)
        s_=0
        for s, v in zip(sublists, vals):
            rmse = np.round(np.mean(np.sqrt(np.mean((ys[s_:s_+s] - preds[s_:s_+s]) ** 2, axis=0)), axis=0), 3)
            rmse_cnn = np.round(np.mean(np.sqrt(np.mean((cnn_preds[s_:s_+s] - preds[s_:s_+s]) ** 2, axis=0)), axis=0), 3)
            s_+=s
            iterator.write(f'\tRMSE: {rmse:.3f}\tRMSE CNN: {rmse_cnn:.3f}')
            iterator.write(f'\t{i_dict[v]}\n')

            metrics_epoch.append(rmse.tolist())
            metrics_epoch.append(rmse_cnn.tolist())


        test_spectra_mean = np.vstack([ #not too elegant but works and it is clear what happens
                                    np.mean(vacuum100_transformed[0][np.where(earth100_labels == 1)[0]], axis=0),
                                    np.mean(vacuum100_transformed[0][np.where(earth100_labels == 7)[0]], axis=0),
                                    np.mean(vacuum100_transformed[0][np.where(earth100_labels == 17)[0]], axis=0),
                                    np.mean(vacuum100_transformed[0][np.where(earth100_labels == 40)[0]], axis=0),

                                    np.mean(vacuum100_transformed[1][np.where(earth50_labels == 1)[0]], axis=0),
                                    np.mean(vacuum100_transformed[1][np.where(earth50_labels == 7)[0]], axis=0),
                                    np.mean(vacuum100_transformed[1][np.where(earth50_labels == 17)[0]], axis=0),
                                    np.mean(vacuum100_transformed[1][np.where(earth50_labels == 40)[0]], axis=0),

                                    np.mean(earth_50_transformed[0][np.where(vac100_labels == 1)[0]], axis=0),
                                    np.mean(earth_50_transformed[0][np.where(vac100_labels == 7)[0]], axis=0),
                                    np.mean(earth_50_transformed[0][np.where(vac100_labels == 17)[0]], axis=0),
                                    np.mean(earth_50_transformed[0][np.where(vac100_labels == 40)[0]], axis=0),

                                    np.mean(earth_50_transformed[1][np.where(earth100_labels == 1)[0]], axis=0),
                                    np.mean(earth_50_transformed[1][np.where(earth100_labels == 7)[0]], axis=0),
                                    np.mean(earth_50_transformed[1][np.where(earth100_labels == 17)[0]], axis=0),
                                    np.mean(earth_50_transformed[1][np.where(earth100_labels == 40)[0]], axis=0),

                                    np.mean(earth100_transformed[0][np.where(vac100_labels == 1)[0]], axis=0),
                                    np.mean(earth100_transformed[0][np.where(vac100_labels == 7)[0]], axis=0),
                                    np.mean(earth100_transformed[0][np.where(vac100_labels == 17)[0]], axis=0),
                                    np.mean(earth100_transformed[0][np.where(vac100_labels == 40)[0]], axis=0),

                                    np.mean(earth100_transformed[1][np.where(earth50_labels == 1)[0]], axis=0),
                                    np.mean(earth100_transformed[1][np.where(earth50_labels == 7)[0]], axis=0),
                                    np.mean(earth100_transformed[1][np.where(earth50_labels == 17)[0]], axis=0),
                                    np.mean(earth100_transformed[1][np.where(earth50_labels == 40)[0]], axis=0),
                                ])
        
        test_spectra = np.concatenate([np.concatenate(vacuum100_transformed, axis=0),
                                    np.concatenate(earth_50_transformed, axis=0),
                                    np.concatenate(earth100_transformed, axis=0)],
                                    axis=0)

        clus_acc, clus_loss = calc_kmeans_loss_mean_spectra(test_spectra_mean)
        clus_acc_, clus_loss_ = calc_kmeans_loss(test_spectra)

        cosine_sim = cosine_similarity_mean_spectra(test_spectra_mean)
        cosine_sim_ = cosine_similarity(test_spectra)

        iterator.write(f'clustering accuracy for mean spectra: {clus_acc:.3f}')
        iterator.write(f'cosine similarity for mean spectra: {cosine_sim:.3f}')
        iterator.write(f'clustering accuracy: {clus_acc_:.3f}')
        iterator.write(f'cosine similarity: {cosine_sim_:.3f}\n')

        metrics_epoch.extend([clus_acc, cosine_sim])
        metrics_epoch.extend([clus_acc_, cosine_sim_])

        ys, preds, cnn_preds = ys.reshape(-1), preds.reshape(-1), cnn_preds.reshape(-1)
        metrics_epoch.append(r2_score(ys, preds))
        metrics_epoch.append(r2_score(cnn_preds, preds))

        metrics.loc[f'seed {_seed} epoch {epoch}'] = metrics_epoch
        metrics.to_csv(f'{path_metrics}/{_seed}-{epoch}.csv', index=True)

        out_spectra = np.concatenate([np.concatenate(c_vacuum100_transformed, axis=0),
                                    np.concatenate(c_earth_50_transformed, axis=0),
                                    np.concatenate(c_earth100_transformed, axis=0)],
                                    axis=0)

        with h5py.File(f'{path_metrics}/{_seed}-{epoch}.h5', 'w') as hf:
            hf.create_dataset('labels', data=[f'{i}-{s}' for i in ['e100_vac100', 'e50_vac100', 'v100_e50', 'e100_e50', 'vac100_e100', 'e50_e100'] for s in [1,7,17,40] for _ in range(100)])
            hf.create_dataset('data', data=out_spectra)

metrics.to_csv(f'{path_metrics}/final.csv', index=True)