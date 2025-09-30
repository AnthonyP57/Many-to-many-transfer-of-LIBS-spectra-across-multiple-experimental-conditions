from models import Classifier, ACVAE, Encoder, Decoder, CNN_d
import torch
from modules import SpectraDataset, ClassDataset
import h5py
import numpy as np
from data_visualization import xy_plot
from sklearn.preprocessing import StandardScaler
import time
import torch.nn as nn
from torch.utils.data import  DataLoader, TensorDataset
import torch.nn.functional as F
import os
import datetime
import json

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0))
else:
    device = torch.device("cpu")
    print("using CPU, GPU is not available")

# device = torch.device("cpu")

with h5py.File('C:\\Users\\Antoni\\PycharmProjects\\Spectra_transfer\\PCA_inliers_mean.h5', 'r') as hf:
    wavelen = np.array(hf['calibration'][()])
    spectra_0_idx = np.where(wavelen == 250)[0][0] - 25
    spectra_last_idx = np.where(wavelen == 850)[0][0] + 25

    all_labels = np.array(hf['labels'])
    indices = list(np.where(all_labels[:, 15] == 1)[0])
    indices_excluded_15 = sorted(list(set(indices)))
    indices.extend(list(np.where(all_labels[:, 32] == 1)[0]))
    indices_excluded_32 = sorted(list(set(indices) - set(indices_excluded_15)))

    indices_ = list(filter(lambda x: x not in indices, list(range(all_labels.shape[0]))))
    print(f'{len(indices_)} spectra included')

    spectra = np.array(hf['spectra'][indices_, spectra_0_idx:spectra_last_idx+1])
    all_spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx+1])

    spectra_15 = all_spectra[indices_excluded_15]
    spectra_32 = all_spectra[indices_excluded_32]

    all_conditions = np.array(hf['conditions'][()])
    conditions = all_conditions[()]
    excluded_conditions_15 = all_conditions[indices_excluded_15]
    excluded_conditions_32 = all_conditions[indices_excluded_32]

    cond_15_earth = np.where(excluded_conditions_15[:, 0] == 1)[0]
    cond_15_mars = np.where(excluded_conditions_15[:, 1] == 1)[0]
    cond_15_vacuum = np.where(excluded_conditions_15[:, 2] == 1)[0]

    cond_32_earth = np.where(excluded_conditions_32[:, 0] == 1)[0]
    cond_32_mars = np.where(excluded_conditions_32[:, 1] == 1)[0]
    cond_32_vacuum = np.where(excluded_conditions_32[:, 2] == 1)[0]

    earth_15 = excluded_conditions_15[cond_15_earth]
    mars_15 = excluded_conditions_15[cond_15_mars]
    vacuum_15 = excluded_conditions_15[cond_15_vacuum]

    earth_32 = excluded_conditions_32[cond_32_earth]
    mars_32 = excluded_conditions_32[cond_32_mars]
    vacuum_32 = excluded_conditions_32[cond_32_vacuum]

    earth_15_spectra = spectra_15[cond_15_earth]
    mars_15_spectra = spectra_15[cond_15_mars]
    vacuum_15_spectra = spectra_15[cond_15_vacuum]

    earth_32_spectra = spectra_32[cond_32_earth]
    mars_32_spectra = spectra_32[cond_32_mars]
    vacuum_32_spectra = spectra_32[cond_32_vacuum]

    labels = all_labels

torch.cuda.empty_cache()

std = StandardScaler().fit(all_spectra)
spectra = std.transform(all_spectra)

mean = torch.tensor(std.mean_).to(device)
std_ = torch.tensor(std.scale_).to(device)

ds = SpectraDataset(spectra, conditions, labels, batch_size=25)
ds.random_select_samples(shuffle_data_pairs=True)

encoder = Encoder(n_classes_channels=1, in_channels=1, ks_lst=[16,16,16,16], out_channels_lst=[3,3,3,3], stride_lst=[1,1,1,1])
decoder = Decoder(n_classes_channels=1, in_channels=3, ks_lst=[16,16,16,16][::-1], out_channels_lst=[3,3,3,1], stride_lst=[1,1,1,1])

class_dataset = DataLoader(ClassDataset(all_spectra, conditions), batch_size=64, shuffle=True)
# classifier = MLP(spectra.shape[1]-2*20, [2048, 1024, 512, 256]).to(device)
classifier = Classifier().to(device)
# classifier = CNN(spectra.shape[1]-2*20, 1, 4, 3).to(device)
clas_optim = torch.optim.Adam(classifier.parameters(), lr=0.001)

for i in range(0):
    epoch_loss = 0
    n_all = 0
    for x, y in class_dataset:
        x = x[:, 25:-25]
        x = x.unsqueeze(1)
        x = x.to(device)
        pred = classifier(x)
        pred = pred.permute(0,2,1).reshape(-1, pred.shape[1])
        y = torch.argmax(y, dim=1).cpu().numpy()[0]
        y = torch.LongTensor(y*np.ones(len(pred))).to(device)
        # pred = pred.reshape(-1, pred.shape[2])
        loss = F.cross_entropy(pred, y)
        clas_optim.zero_grad()
        loss.backward()
        clas_optim.step()

        n_all += x.shape[0]
        epoch_loss += loss*x.shape[0]

    print(f'Classifier pretraining epoch: {i+1} loss: {epoch_loss/n_all:.4f}')
    ds.random_select_samples(True)

# for param in classifier.parameters():
#     param.requires_grad = False

# discriminator = Discriminator(spectra.shape[1] - 2*20).to(device)
discriminator = CNN_d(spectra.shape[1]-2*25, 1, 4).to(device)
# discriminator = Discriminator_c().to(device)
# discriminator = MLP(spectra.shape[1]-2*25, [256, 128, 64], drop=0.1).to(device)
acvae = ACVAE(encoder, decoder, classifier, mean, std_).to(device)
optimizer = torch.optim.Adam(acvae.parameters(), lr=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)

disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
disc_scheduler = torch.optim.lr_scheduler.StepLR(disc_optimizer, step_size=10, gamma=0.5)
disc_criterion = nn.BCELoss()

start = time.time()

epochs=5

for epoch in range(epochs):
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

    for x_0, x_1, y_0, y_1 in ds:
        # x_0_ave = x_0.mean(dim=0)
        # x_1_ave = x_1.mean(dim=0)
        #
        # x_0 = x_0_ave.unsqueeze(0).expand_as(x_0)
        # x_1 = x_1_ave.unsqueeze(0).expand_as(x_1)

        x_0 = x_0.to(device)
        x_1 = x_1.to(device)
        y_0 = y_0.to(device)
        y_1 = y_1.to(device)

        kl_loss, gaus_neg_log_like, ClsLoss_class, ClsLoss_recon, ClsLoss_convert, loss_d = acvae.calc_loss(x_0, x_1, y_0, y_1, crop_by=25)

        vae_loss = gaus_neg_log_like + kl_loss + ClsLoss_class + ClsLoss_recon + ClsLoss_convert + loss_d*1

        a = x_0.shape[0]
        running_kl_loss += kl_loss*a
        running_gaus_loss += gaus_neg_log_like*a
        running_clas_loss += ClsLoss_class*a
        running_recon_loss += ClsLoss_recon*a
        running_convert_loss += ClsLoss_convert*a
        running_d_loss += loss_d*a
        n_all += a

        x_01 = acvae(x_0, y_0, y_1, crop_by=25)
        x_00 = acvae(x_0, y_0, y_0, crop_by=25)
        x_11 = acvae(x_1, y_1, y_1, crop_by=25)
        x_10 = acvae(x_1, y_1, y_0, crop_by=25)

        fake = torch.cat([x_00, x_01, x_10, x_11], dim=0).squeeze(1)
        real = torch.cat([x_0, x_1], dim=0)[:, 25:-25]

        fake_labels = torch.zeros(fake.shape[0],1)
        real_labels = torch.ones(real.shape[0],1)

        all_tensors = torch.cat([fake.detach(), real], dim=0)
        all_labels = torch.cat([fake_labels, real_labels], dim=0)

        dataset = TensorDataset(all_tensors, all_labels)
        data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

        for batch_tensors, batch_labels in data_loader:
            batch_tensors = batch_tensors.to(device)
            batch_labels = batch_labels.to(device)
            output = discriminator(batch_tensors)
            loss = disc_criterion(output, batch_labels)
            disc_loss = (loss)
            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()

            running_fake_loss += loss*batch_labels.shape[0]
            n_ += batch_labels.shape[0]

        real_n += real.shape[0]
        fake_n += fake.shape[0]

        fake_output_ = discriminator(fake)
        fake_loss_ = disc_criterion(fake_output_, torch.ones_like(fake_output_))
        vae_loss += fake_loss_/5

        running_acvae_fake_loss += fake_loss_*fake.shape[0]

        optimizer.zero_grad()
        vae_loss.backward()
        optimizer.step()

    scheduler.step()
    disc_scheduler.step()

    print(f'kl_loss: {running_kl_loss/n_all:.5f}'
          f'\ngauss_neg_log_like: {running_gaus_loss/n_all:.5f}'
          f'\nClsLoss_recon: {running_recon_loss/n_all:.5f}'
          f'\nClsLoss_convert: {running_convert_loss/n_all:.5f}'
          f'\nClsLoss_class: {running_clas_loss/n_all/3:.5f}'
          f'\nloss_d: {running_d_loss/n_all:.5f}'
          f'\ndisc_fake: {running_fake_loss/n_:.5f}'
          f'\ndisc_real: {running_real_loss/real_n:.5f}'
          f'\nacvae_disc: {running_acvae_fake_loss/fake_n:.5f}')

    print(f'epoch: {epoch + 1}/{epochs}')
    tt = time.time() - start
    remaining_epochs = epochs - (epoch + 1)
    eta_tt = tt * remaining_epochs / (epoch + 1)
    print(f'time elapsed: {int(tt // 60):02d}:{int(tt % 60):02d} min, ETA: {int(eta_tt // 60):02d}:{int(eta_tt % 60):02d} min\n')
    ds.random_select_samples(shuffle_data_pairs=True)

now = datetime.datetime.now()
folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
folder_name = os.path.join('img', folder_name)
os.makedirs(folder_name, exist_ok=True)

with open('reg_conditions_dict.json', 'r') as j:
    reg_conditions_dict = json.loads(j.read())

x_01 = acvae(x_0.to(device), y_0.to(device), y_1.to(device), crop_by=25)
x_01 = std.inverse_transform(np.pad(x_01.view(-1, x_01.shape[2]).cpu().detach().numpy(), ((0,0), (25,25)), mode='constant'))[:, 25:-25]
x_0_ = std.inverse_transform(x_0.cpu().detach())[:,25:-25]
x_1_ = std.inverse_transform(x_1.cpu().detach())[:,25:-25]

for i in range(5 if x_1.shape[0] > 5 else x_1.shape[0]):
    spectra_ = [x_0_[i], x_1_[i], x_01[i]]
    xy_plot([wavelen[spectra_0_idx:spectra_last_idx + 1][25:-25] for i in range(len(spectra_))], spectra_,
            labels=['original', 'how transformed should look', 'transformed'],
            xy_labels=['wavelength [nm]', 'intensity [a.u.]'], min_max_xlim=[wavelen[spectra_0_idx+25], wavelen[spectra_last_idx+1-25]],
            title=f'Random reconstructed spectrum visualization from {reg_conditions_dict[str(np.argmax(y_0.cpu().detach().numpy()[0]))]} '
                  f'to {reg_conditions_dict[str(np.argmax(y_1.cpu().detach().numpy()[0]))]}', save_to=folder_name+f'/no_disc_01_{i}.png')

x_10 = acvae(x_1.to(device), y_1.to(device), y_0.to(device), crop_by=25)
x_10 = std.inverse_transform(np.pad(x_10.view(-1, x_10.shape[2]).cpu().detach().numpy(), ((0,0), (25,25)), mode='constant'))[:, 25:-25]

for i in range(5 if x_10.shape[0] > 5 else x_10.shape[0]):
    spectra_ = [x_1_[i], x_0_[i], x_10[i]]
    xy_plot([wavelen[spectra_0_idx:spectra_last_idx + 1][25:-25] for i in range(len(spectra_))], spectra_,
            labels=['original', 'how transformed should look', 'transformed'],
            xy_labels=['wavelength [nm]', 'intensity [a.u.]'], min_max_xlim=[wavelen[spectra_0_idx + 25],wavelen[spectra_last_idx + 1 - 25]],
            title=f'Random reconstructed spectrum visualization from {reg_conditions_dict[str(np.argmax(y_1.cpu().detach().numpy()[0]))]} '
                  f'to {reg_conditions_dict[str(np.argmax(y_0.cpu().detach().numpy()[0]))]}', save_to=folder_name+f'/no_disc_10_{i}.png')

earth_15_spectra = np.array([earth_15_spectra.mean(axis=0)])

earth_15_spectra_ = std.transform(earth_15_spectra)
earth_15_ = acvae(torch.from_numpy(earth_15_spectra_.astype(np.float32)).to(device),
                  torch.from_numpy(earth_15.astype(np.float32)).to(device), torch.from_numpy(mars_15.astype(np.float32)).to(device), crop_by=25)
earth_15_ = std.inverse_transform(np.pad(earth_15_.view(-1, earth_15_.shape[2]).cpu().detach().numpy(), ((0,0), (25,25)), mode='constant'))[:, 25:-25]

earth_15_eg = earth_15_spectra[:, 25:-25]
mars_15_eg = mars_15_spectra[:, 25:-25]

for i in range(5 if earth_15_eg.shape[0] > 5 else earth_15_eg.shape[0]):
    spectra_ = [earth_15_eg[i], mars_15_eg[i], earth_15_[i]]
    xy_plot([wavelen[spectra_0_idx:spectra_last_idx + 1][25:-25] for i in range(len(spectra_))], spectra_,
            labels=['original', 'how transformed should look', 'transformed'],
            xy_labels=['wavelength [nm]', 'intensity [a.u.]'], min_max_xlim=[wavelen[spectra_0_idx + 25], wavelen[spectra_last_idx + 1 - 25]],
            title=f'Random LHS-1E reconstructed spectrum visualization from {reg_conditions_dict[str(np.argmax(earth_15[0]))]} '
                  f'to {reg_conditions_dict[str(np.argmax(mars_15[0]))]}', save_to=folder_name+f'/no_disc_LHS-1E_EtM_{i}.png')

earth_32_spectra = np.array([earth_32_spectra.mean(axis=0)])

earth_32_spectra_ = std.transform(earth_32_spectra)
earth_32_ = acvae(torch.from_numpy(earth_32_spectra_.astype(np.float32)).to(device),
                  torch.from_numpy(earth_32.astype(np.float32)).to(device), torch.from_numpy(mars_32.astype(np.float32)).to(device), crop_by=25)
earth_32_ = std.inverse_transform(np.pad(earth_32_.view(-1, earth_32_.shape[2]).cpu().detach().numpy(), ((0,0), (25,25)), mode='constant'))[:, 25:-25]

earth_32_eg = earth_32_spectra[:, 25:-25]
mars_32_eg = mars_32_spectra[:, 25:-25]

for i in range(5 if earth_32_eg.shape[0] > 5 else earth_32_eg.shape[0]):
    spectra_ = [earth_32_eg[i], mars_32_eg[i], earth_32_[i]]
    xy_plot([wavelen[spectra_0_idx:spectra_last_idx + 1][25:-25] for i in range(len(spectra_))], spectra_,
            labels=['original', 'how transformed should look', 'transformed'],
            xy_labels=['wavelength [nm]', 'intensity [a.u.]'], min_max_xlim=[wavelen[spectra_0_idx + 25], wavelen[spectra_last_idx + 1 - 25]],
            title=f'Random SiO2-CL reconstructed spectrum visualization from {reg_conditions_dict[str(np.argmax(earth_32[0]))]} '
                  f'to {reg_conditions_dict[str(np.argmax(mars_32[0]))]}', save_to=folder_name+f'/no_disc_SiO2-CL_EtM_{i}.png')

# torch.save(acvae, 'models/ACVAE_baseline_all_spectra_all_range.pth')
# torch.save(acvae.state_dict(), 'models/ACVAE_baseline_state_dict_all_spectra_all_range.pth')