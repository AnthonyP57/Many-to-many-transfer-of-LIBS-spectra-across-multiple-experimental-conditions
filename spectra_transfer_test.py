from models import Classifier, ACVAE, Encoder, Decoder, Classifier2, MLP, Discriminator, CNN, CNN_d
import torch
from modules import SpectraDataset, calculate_model_size, ClassDataset
import h5py
import numpy as np
from data_visualization import xy_plot
from sklearn.preprocessing import StandardScaler
import time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0))
else:
    device = torch.device("cpu")
    print("using CPU, GPU is not available")

with h5py.File('/home/antonipa57/PycharmProjects/Spectra_transfer/spectra_transfer_code/CAVE_alike/PCA_inliers.h5', 'r') as hf:
    wavelen = np.array(hf['calibration'][()])
    spectra_0_idx = np.where(wavelen == 390)[0][0]
    spectra_last_idx = np.where(wavelen == 400)[0][0]

    all_labels = np.array(hf['labels'])
    indices = list(np.where(all_labels[:, 13] == 1)[0])
    indices.extend(list(np.where(all_labels[:, 27] == 1)[0]))
    indices_ = list(filter(lambda x: x not in indices, list(range(all_labels.shape[0]))))
    print(f'{len(indices_)} spectra included')

    spectra = np.array(hf['spectra'][indices_, spectra_0_idx:spectra_last_idx+1])
    all_spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx+1])
    excluded_spectra = all_spectra[indices_, spectra_0_idx:spectra_last_idx+1]

    all_conditions = np.array(hf['conditions'][()])
    conditions = all_conditions[indices_]
    excluded_conditions = all_conditions[indices]

    labels = all_labels[indices_]
    excluded_labels = all_labels[indices]

torch.cuda.empty_cache()

std = StandardScaler().fit(all_spectra)
spectra = std.transform(spectra)

ds = SpectraDataset(spectra, conditions, labels, batch_size=50)
ds.random_select_samples(shuffle_data_pairs=True)

encoder = Encoder(n_classes_channels=1, in_channels=1, ks_lst=[12,6,6,3], out_channels_lst=[8,8,8,8], stride_lst=[1,1,1,1])
decoder = Decoder(n_classes_channels=1, in_channels=8, ks_lst=[3,6,6,12], out_channels_lst=[8,8,8,1], stride_lst=[1,1,1,1])

class_dataset = DataLoader(ClassDataset(spectra, conditions), batch_size=64, shuffle=True)
classifier = MLP(spectra.shape[1]-2*20, [2048, 1024, 512, 256]).to(device)
# classifier = CNN(spectra.shape[1]-2*20, 1, 4, 3).to(device)
clas_optim = torch.optim.Adam(classifier.parameters(), lr=0.00026)
pred_loss = nn.CrossEntropyLoss()

for i in range(25):
    epoch_loss = 0
    n_all = 0
    for x, y in class_dataset:
        x = x.to(device)[:, 20:-20]#.unsqueeze(1)
        y = y.to(device)
        pred = classifier(x)
        loss = pred_loss(pred, y)
        clas_optim.zero_grad()
        loss.backward()
        clas_optim.step()

        n_all += x.shape[0]
        epoch_loss += loss*x.shape[0]

    print(f'Classifier pretraining epoch: {i+1} loss: {epoch_loss/n_all:.4f}')
    ds.random_select_samples(True)

for param in classifier.parameters():
    param.requires_grad = False

acvae = ACVAE(encoder, decoder, classifier).to(device)
optimizer = torch.optim.Adam(acvae.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
calculate_model_size(acvae)
start = time.time()

epochs=30

for epoch in range(epochs):
    running_kl_loss = 0
    running_gaus_loss = 0
    running_clas_loss = 0
    running_recon_loss = 0
    running_convert_loss = 0
    n_all = 0

    for x_0, x_1, y_0, y_1 in ds:
        x_0 = x_0.to(device)
        x_1 = x_1.to(device)
        y_0 = y_0.to(device)
        y_1 = y_1.to(device)

        kl_loss, gaus_neg_log_like, ClsLoss_class, ClsLoss_recon, ClsLoss_convert = acvae.calc_loss(x_0, x_1, y_0, y_1, crop_by=20)

        optimizer.zero_grad()
        vae_loss = gaus_neg_log_like + kl_loss + ClsLoss_class*0 + ClsLoss_recon + ClsLoss_convert
        vae_loss.backward()
        optimizer.step()

        a = x_0.shape[0]
        running_kl_loss += kl_loss*a
        running_gaus_loss += gaus_neg_log_like*a
        running_clas_loss += ClsLoss_class*a
        running_recon_loss += ClsLoss_recon*a
        running_convert_loss += ClsLoss_convert*a
        n_all += a


    scheduler.step()

    print(f'kl_loss: {running_kl_loss/n_all:.5f}'
          f'\ngauss_neg_log_like: {running_gaus_loss/n_all:.5f}'
          f'\nClsLoss_recon: {running_recon_loss/n_all:.5f}'
          f'\nClsLoss_convert: {running_convert_loss/n_all:.5f}'
          f'\nClsLoss_class: {running_clas_loss/n_all:.5f}')

    print(f'epoch: {epoch + 1}/{epochs}')
    tt = time.time() - start
    remaining_epochs = epochs - (epoch + 1)
    eta_tt = tt * remaining_epochs / (epoch + 1)
    print(f'time elapsed: {int(tt // 60):02d}:{int(tt % 60):02d} min, ETA: {int(eta_tt // 60):02d}:{int(eta_tt % 60):02d} min\n')
    ds.random_select_samples(shuffle_data_pairs=True)


x_01 = acvae(x_0.to(device), y_0.to(device), y_1.to(device), crop_by=20)
x_01 = std.inverse_transform(np.pad(x_01.view(-1, x_01.shape[2]).cpu().detach().numpy(), ((0,0), (20,20)), mode='constant'))[:, 20:-20]
x_0_ = std.inverse_transform(x_0.cpu().detach())[:,20:-20]
x_1_ = std.inverse_transform(x_1.cpu().detach())[:,20:-20]

for i in range(5 if x_1.shape[0] > 5 else x_1.shape[0]):
    spectra_ = [x_0_[i], x_1_[i], x_01[i]]
    xy_plot([wavelen[spectra_0_idx:spectra_last_idx + 1][20:-20] for i in range(len(spectra_))], spectra_, labels=['original', 'how transformed should look', 'transformed'], xy_labels=['wavelength [nm]', 'intensity [a.u.]'], min_max_xlim=[wavelen[spectra_0_idx+20], wavelen[spectra_last_idx+1-20]], title='Random reconstructed spectrum visualization')

x_10 = acvae(x_1.to(device), y_1.to(device), y_0.to(device), crop_by=20)
x_10 = std.inverse_transform(np.pad(x_10.view(-1, x_10.shape[2]).cpu().detach().numpy(), ((0,0), (20,20)), mode='constant'))[:, 20:-20]

for i in range(5 if x_10.shape[0] > 5 else x_10.shape[0]):
    spectra_ = [x_1_[i], x_0_[i], x_10[i]]
    xy_plot([wavelen[spectra_0_idx:spectra_last_idx + 1][20:-20] for i in range(len(spectra_))], spectra_, labels=['original', 'how transformed should look', 'transformed'], xy_labels=['wavelength [nm]', 'intensity [a.u.]'], min_max_xlim=[wavelen[spectra_0_idx + 20], wavelen[spectra_last_idx + 1 - 20]], title='Random reconstructed spectrum visualization')


epochs_discriminator = 3
discriminator = Discriminator(spectra.shape[1] - 2*20).to(device)
discriminator = CNN_d(spectra.shape[1]-2*20, 1, 4).to(device)
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.001)
criterion_d = nn.BCELoss()
scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_discriminator, step_size=3, gamma=0.5)
optimizer = torch.optim.Adam(acvae.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


cow = 0
frog = True
while frog:

    data_list = []
    label_list = []

    with torch.no_grad():
        for x_0, x_1, y_0, y_1 in ds:
            x_0 = x_0.to(device)
            x_1 = x_1.to(device)
            y_0 = y_0.to(device)
            y_1 = y_1.to(device)

            a = acvae(x_0, y_0, y_1, crop_by=20)

            data_list.append(a.view(-1, a.shape[2]).cpu().detach())
            label_list.append([0]*a.shape[0])
            data_list.append(x_1.cpu().detach()[:,20:-20])
            label_list.append([1]*x_1.shape[0])

            a = acvae(x_1, y_1, y_0, crop_by=20)

            data_list.append(a.view(-1, a.shape[2]).cpu().detach())
            label_list.append([0]*a.shape[0])
            data_list.append(x_0.cpu().detach()[:,20:-20])
            label_list.append([1]*x_0.shape[0])

            a = acvae(x_0, y_0, y_0, crop_by=20)

            data_list.append(a.view(-1, a.shape[2]).cpu().detach())
            label_list.append([0]*a.shape[0])
            data_list.append(x_0.cpu().detach()[:,20:-20])
            label_list.append([1]*x_0.shape[0])

            a = acvae(x_1, y_1, y_1, crop_by=20)

            data_list.append(a.view(-1, a.shape[2]).cpu().detach())
            label_list.append([0]*a.shape[0])
            data_list.append(x_1.cpu().detach()[:,20:-20])
            label_list.append([1]*x_1.shape[0])

    data_list = np.array(data_list).reshape(-1, 501-2*20)
    label_list = np.array(label_list).reshape(-1, 1)

    disc_dataloader = DataLoader(ClassDataset(data_list, label_list), batch_size=64, shuffle=True)
    del data_list, label_list

    for param in discriminator.parameters():
        param.requires_grad = True

    for epoch in range(epochs_discriminator):
        n_all = 0
        running_loss = 0
        for data, label in disc_dataloader:

            data = data.to(device)
            label = label.to(device)

            optimizer_discriminator.zero_grad()
            output = discriminator(data)
            loss = criterion_d(output, label)
            loss.backward()
            optimizer_discriminator.step()

            running_loss += loss*label.shape[0]
            n_all += label.shape[0]

        scheduler_d.step()
        print(f'epoch: {epoch + 1}/{epochs_discriminator}, loss: {running_loss/n_all:.3f}')
        if epoch == epochs_discriminator - 1:
            last_epoch_loss = running_loss/n_all
            if last_epoch_loss > 0.4:
                frog = False

    for param in discriminator.parameters():
        param.requires_grad = False


    start = time.time()
    with torch.set_grad_enabled(True):
        for epoch in range(round(epochs)):
            running_kl_loss = 0
            running_gaus_loss = 0
            running_clas_loss = 0
            running_recon_loss = 0
            running_convert_loss = 0
            running_disc_loss = 0
            n_all = 0

            for x_0, x_1, y_0, y_1 in ds:
                x_0 = x_0.to(device)
                x_1 = x_1.to(device)
                y_0 = y_0.to(device)
                y_1 = y_1.to(device)

                kl_loss, gaus_neg_log_like, ClsLoss_class, ClsLoss_recon, ClsLoss_convert = acvae.calc_loss(x_0, x_1, y_0, y_1, crop_by=20)

                with torch.no_grad():
                    a = acvae(x_0, y_0, y_1, crop_by=20)
                    b = acvae(x_1, y_1, y_0, crop_by=20)
                    c = acvae(x_0, y_0, y_0, crop_by=20)
                    d = acvae(x_1, y_1, y_1, crop_by=20)
                convert_loss = torch.cat([a.view(-1, a.shape[2]), b.view(-1, b.shape[2])], dim=0)
                recon_loss = torch.cat([c.view(-1, c.shape[2]), d.view(-1, d.shape[2])], dim=0)
                disc_pred_c = discriminator(convert_loss)
                disc_pred_r = discriminator(recon_loss)
                disc_pred = torch.cat([disc_pred_c, 0.1*disc_pred_r], dim=0)

                loss_d = criterion_d(disc_pred_c, torch.ones_like(disc_pred_c))/2
                loss_d += 0.2*criterion_d(disc_pred_r, torch.zeros_like(disc_pred_r))/2

                loss = kl_loss + gaus_neg_log_like + ClsLoss_class*0 + ClsLoss_recon + ClsLoss_convert + loss_d*0.5

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_disc_loss += loss*disc_pred.shape[0]
                running_kl_loss += kl_loss*disc_pred.shape[0]
                running_gaus_loss += gaus_neg_log_like*disc_pred.shape[0]
                running_recon_loss += ClsLoss_recon*disc_pred.shape[0]
                running_convert_loss += ClsLoss_convert*disc_pred.shape[0]
                running_clas_loss += ClsLoss_class*disc_pred.shape[0]
                n_all += disc_pred.shape[0]

            print(f'kl_loss: {running_kl_loss / n_all:.5f}'
                  f'\ngauss_neg_log_like: {running_gaus_loss / n_all:.5f}'
                  f'\nClsLoss_recon: {running_recon_loss / n_all:.5f}'
                  f'\nClsLoss_convert: {running_convert_loss / n_all:.5f}'
                  f'\nClsLoss_class: {running_clas_loss / n_all:.5f}'
                  f'\ndisc_loss: {running_disc_loss/n_all/4:.5f}')

            print(f'epoch: {epoch + 1}/{epochs}')
            tt = time.time() - start
            remaining_epochs = epochs - (epoch + 1)
            eta_tt = tt * remaining_epochs / (epoch + 1)
            print(f'time elapsed: {int(tt // 60):02d}:{int(tt % 60):02d} min, ETA: {int(eta_tt // 60):02d}:{int(eta_tt % 60):02d} min\n')
            ds.random_select_samples(shuffle_data_pairs=True)
            scheduler.step()


    # x_01 = acvae(x_0.to(device), y_0.to(device), y_1.to(device), crop_by=20)
    # x_01 = std.inverse_transform(np.pad(x_01.view(-1, x_01.shape[2]).cpu().detach().numpy(), ((0,0), (20,20)), mode='constant'))[:, 20:-20]
    # x_0 = std.inverse_transform(x_0.cpu().detach())[:,20:-20]
    # x_1 = std.inverse_transform(x_1.cpu().detach())[:,20:-20]

    # for i in range(5 if x_01.shape[0] > 5 else x_01.shape[0]):
    #     spectra = [x_0[i], x_1[i], x_01[i]]
    #     xy_plot([wavelen[spectra_0_idx:spectra_last_idx + 1][20:-20] for i in range(len(spectra))], spectra, labels=['original', 'how transformed should look', 'transformed'], xy_labels=['wavelength [nm]', 'intensity [a.u.]'], min_max_xlim=[wavelen[spectra_0_idx+20], wavelen[spectra_last_idx+1-20]], title='Random reconstructed spectrum visualization')

    cow += 1
    if cow == 2:
        frog = False

    x_01 = acvae(x_0.to(device), y_0.to(device), y_1.to(device), crop_by=20)
    x_01 = std.inverse_transform(np.pad(x_01.view(-1, x_01.shape[2]).cpu().detach().numpy(), ((0,0), (20,20)), mode='constant'))[:, 20:-20]
    x_0_ = std.inverse_transform(x_0.cpu().detach())[:,20:-20]
    x_1_ = std.inverse_transform(x_1.cpu().detach())[:,20:-20]

    for i in range(5 if x_01.shape[0] > 5 else x_01.shape[0]):
        spectra_ = [x_0_[i], x_1_[i], x_01[i]]
        xy_plot([wavelen[spectra_0_idx:spectra_last_idx + 1][20:-20] for i in range(len(spectra_))], spectra_, labels=['original', 'how transformed should look', 'transformed'], xy_labels=['wavelength [nm]', 'intensity [a.u.]'], min_max_xlim=[wavelen[spectra_0_idx+20], wavelen[spectra_last_idx+1-20]], title='Random reconstructed spectrum visualization')

    x_10 = acvae(x_1.to(device), y_1.to(device), y_0.to(device), crop_by=20)
    x_10 = std.inverse_transform(np.pad(x_10.view(-1, x_10.shape[2]).cpu().detach().numpy(), ((0,0), (20,20)), mode='constant'))[:, 20:-20]

    for i in range(5 if x_10.shape[0] > 5 else x_10.shape[0]):
        spectra_ = [x_1_[i], x_0_[i], x_10[i]]
        xy_plot([wavelen[spectra_0_idx:spectra_last_idx + 1][20:-20] for i in range(len(spectra_))], spectra_, labels=['original', 'how transformed should look', 'transformed'], xy_labels=['wavelength [nm]', 'intensity [a.u.]'], min_max_xlim=[wavelen[spectra_0_idx + 20], wavelen[spectra_last_idx + 1 - 20]], title='Random reconstructed spectrum visualization')
