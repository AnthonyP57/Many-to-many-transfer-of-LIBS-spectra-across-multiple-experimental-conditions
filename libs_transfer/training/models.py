import numpy as np
from libs_transfer.training.modules import ConvBatchNormGLU1D, DeConvBatchNormGLU1D, concat_dim1, gaussian_repar, kl_loss, gauss_negative_log_like, add_total_emis
import torch.nn as nn
import torch
import torch.nn.functional as F
from libs_transfer.prepare_data.spectra_normalization import calc_padding, find_tensor_highest_peaks

class Encoder(nn.Module):
    def __init__(self, total_emis_channels=1, ks_lst=[3,4,4,9], in_channels=1, n_classes_channels=1, out_channels_lst=[8,16,16,10], pd_lst=[True, True, True, True, True, True], dilation=1, stride_lst=[1,2,2,1], skip_pad=6, add_fc=False):
        super(Encoder, self).__init__()

        if add_fc:
            self.fc = nn.GRU(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
            self.fc_ = nn.Linear(16,1)
            self.bn = nn.BatchNorm1d(1)

        self.add_fc = add_fc

        self.l1 = ConvBatchNormGLU1D(ks=ks_lst[0],
                                       in_channels=in_channels + n_classes_channels + total_emis_channels,
                                       out_channels=out_channels_lst[0],
                                       pd=pd_lst[0],
                                       dilation=dilation,
                                       stride=stride_lst[0])

        self.l2 = ConvBatchNormGLU1D(ks=ks_lst[1],
                                       in_channels=out_channels_lst[0] + n_classes_channels + total_emis_channels,
                                       out_channels=out_channels_lst[1],
                                       pd=pd_lst[1],
                                       dilation=dilation,
                                       stride=stride_lst[1])

        self.l3 = ConvBatchNormGLU1D(ks=ks_lst[2],
                                       in_channels=out_channels_lst[1] + n_classes_channels + total_emis_channels,
                                       out_channels=out_channels_lst[2],
                                       pd=pd_lst[2],
                                       dilation=dilation,
                                       stride=stride_lst[2])

        self.l4 = ConvBatchNormGLU1D(ks=ks_lst[3],
                                       in_channels=out_channels_lst[2] + n_classes_channels + total_emis_channels,
                                       out_channels=out_channels_lst[3],
                                       pd=pd_lst[3],
                                       dilation=dilation,
                                       stride=stride_lst[3])

        self.l5 = ConvBatchNormGLU1D(ks=ks_lst[4],
                                       in_channels=out_channels_lst[3] + n_classes_channels + total_emis_channels,
                                       out_channels=out_channels_lst[4],
                                       pd=pd_lst[4],
                                       dilation=dilation,
                                       stride=stride_lst[4])

        self.l6 = ConvBatchNormGLU1D(ks=ks_lst[5],
                                       in_channels=out_channels_lst[4] + n_classes_channels + total_emis_channels,
                                       out_channels=out_channels_lst[5],
                                       pd=pd_lst[5],
                                       dilation=dilation,
                                       stride=stride_lst[5])

        self.conv = nn.Conv1d(in_channels=out_channels_lst[5] + n_classes_channels + total_emis_channels,
                                       out_channels=out_channels_lst[-1] * 2,  # because we divide into mu and ln_var
                                       kernel_size=ks_lst[-1],
                                       stride=stride_lst[-1],
                                       padding=0)

        self.skip = ConvBatchNormGLU1D(ks=ks_lst[3],
                                                    in_channels=out_channels_lst[2],
                                                   out_channels=out_channels_lst[5],
                                                   pd=skip_pad,
                                                   dilation=dilation,
                                                   stride=stride_lst[3])

    def forward(self, x, y, e, concat=concat_dim1):
        batch, channels, data = x.shape

        if self.add_fc:
            hidden = torch.zeros(1, batch, 16).to(x.device)

            x = F.leaky_relu(self.fc(x.permute(0,2,1), hidden)[0])
            x = self.fc_(x).permute(0,2,1)
            x = self.bn(x)

        x = concat(x, y)
        x = add_total_emis(x, e)
        x = self.l1(x)

        x = concat(x, y)
        x = add_total_emis(x, e)
        x = self.l2(x)

        x = concat(x, y)
        x = add_total_emis(x, e)
        x = self.l3(x)

        merge = self.skip(x)


        x = concat(x, y)
        x = add_total_emis(x, e)
        x = self.l4(x)

        x = concat(x, y)
        x = add_total_emis(x, e)
        x = self.l5(x)

        x = concat(x, y)
        x = add_total_emis(x, e)
        x = self.l6(x)


        x += merge

        x = concat(x, y)
        x = add_total_emis(x, e)
        x = self.conv(x)

        mu, ln_var = torch.split(x, x.shape[1] // 2, dim=1)

        ln_var = torch.clamp(ln_var, min=-50.0, max=0.0)  # used for variance
        return mu, ln_var

class Decoder(nn.Module):
    def __init__(self, total_emis_channels=1, ks_lst=[9,4,4,3], in_channels=10, n_classes_channels=1, out_channels_lst=[16,18,8,1], pd_lst=[True, True, True, True, True, True], dilation=1, stride_lst=[1,2,2,1], skip_pad=6, add_fc=False):
        super(Decoder, self).__init__()

        if add_fc:
            self.fc1 = nn.GRU(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
            self.fc_1 = nn.Linear(16,1)
            self.bn1 = nn.BatchNorm1d(1)
            self.fc2 = nn.GRU(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
            self.fc_2 = nn.Linear(16,1)
            self.bn2 = nn.BatchNorm1d(1)

        self.add_fc = add_fc

        self.l1 = DeConvBatchNormGLU1D(ks=ks_lst[0],
                                   in_channels=in_channels + n_classes_channels + total_emis_channels,
                                   out_channels=out_channels_lst[0],
                                   pd=pd_lst[0],
                                   dilation=dilation,
                                   stride=stride_lst[0])

        self.l2 = DeConvBatchNormGLU1D(ks=ks_lst[1],
                                       in_channels=out_channels_lst[0] + n_classes_channels + total_emis_channels,
                                       out_channels=out_channels_lst[1],
                                       pd=pd_lst[1],
                                       dilation=dilation,
                                       stride=stride_lst[1])

        self.l3 = DeConvBatchNormGLU1D(ks=ks_lst[2],
                                       in_channels=out_channels_lst[1] + n_classes_channels + total_emis_channels,
                                       out_channels=out_channels_lst[2],
                                       pd=pd_lst[2],
                                       dilation=dilation,
                                       stride=stride_lst[2])

        self.l4 = DeConvBatchNormGLU1D(ks=ks_lst[3],
                                       in_channels=out_channels_lst[2] + n_classes_channels + total_emis_channels,
                                       out_channels=out_channels_lst[3],
                                       pd=pd_lst[3],
                                       dilation=dilation,
                                       stride=stride_lst[3])

        self.l5 = DeConvBatchNormGLU1D(ks=ks_lst[4],
                                       in_channels=out_channels_lst[3] + n_classes_channels + total_emis_channels,
                                       out_channels=out_channels_lst[4],
                                       pd=pd_lst[4],
                                       dilation=dilation,
                                       stride=stride_lst[4])

        self.l6 = DeConvBatchNormGLU1D(ks=ks_lst[5],
                                       in_channels=out_channels_lst[4] + n_classes_channels + total_emis_channels,
                                       out_channels=out_channels_lst[5],
                                       pd=pd_lst[5],
                                       dilation=dilation,
                                       stride=stride_lst[5])

        self.conv = nn.ConvTranspose1d(in_channels=out_channels_lst[5] + n_classes_channels + total_emis_channels,
                              out_channels=out_channels_lst[-1] * 2,  # because we divide into mu and ln_var
                              kernel_size=ks_lst[-1],
                              stride=stride_lst[-1],
                              padding=0)

        self.skip = DeConvBatchNormGLU1D(ks=ks_lst[3],
                                       in_channels=out_channels_lst[2],
                                       out_channels=out_channels_lst[5],
                                       pd=skip_pad,
                                       dilation=dilation,
                                       stride=stride_lst[3])

    def forward(self, x, y, e, concat=concat_dim1):
        batch, channels, data = x.shape

        x = concat(x, y)
        x = add_total_emis(x,e)
        x = self.l1(x)

        x = concat(x, y)
        x = add_total_emis(x, e)
        x = self.l2(x)

        x = concat(x, y)
        x = add_total_emis(x, e)
        x = self.l3(x)

        merge = self.skip(x)

        x = concat(x, y)
        x = add_total_emis(x, e)
        x = self.l4(x)

        x = concat(x, y)
        x = add_total_emis(x, e)
        x = self.l5(x)

        x = concat(x, y)
        x = add_total_emis(x, e)
        x = self.l6(x)

        x += merge

        x = concat(x,y)
        x = add_total_emis(x,e)
        x = self.conv(x)

        # x = x.clone()[:, :, 0:data]
        mu, ln_var = torch.split(x, x.shape[1] // 2, dim=1)

        if self.add_fc:
            hidden1 = torch.zeros(1, batch, 16).to(x.device)
            hidden2 = torch.zeros(1, batch, 16).to(x.device)

            mu = F.leaky_relu(self.fc1(mu.permute(0,2,1), hidden1)[0])
            mu = self.fc_1(mu).permute(0,2,1)
            mu = self.bn1(mu)

            ln_var = F.leaky_relu(self.fc2(ln_var.permute(0,2,1), hidden2)[0])
            ln_var = self.fc_2(ln_var).permute(0,2,1)
            ln_var = self.bn2(ln_var)

        ln_var = torch.clamp(ln_var, min=-50.0, max=0.0)  # used for variance
        return mu, ln_var

class Classifier(nn.Module):
    def __init__(self, ks_lst=[5,4,4,4,5], in_channels=1, n_classes=3, out_channels_lst=[8,8,8,8], pd_lst=[True, True, True, True], dilation=1, stride_lst=[1,2,2,2,1], drop=0.2):
        super(Classifier, self).__init__()
        self.l1 = ConvBatchNormGLU1D(ks=ks_lst[0],
                                     in_channels=in_channels,
                                     out_channels=out_channels_lst[0],
                                     pd=pd_lst[0],
                                     dilation=dilation,
                                     stride=stride_lst[0],)

        self.l2 = ConvBatchNormGLU1D(ks=ks_lst[1],
                                     in_channels=out_channels_lst[0],
                                     out_channels=out_channels_lst[1],
                                     pd=pd_lst[1],
                                     dilation=dilation,
                                     stride=stride_lst[1])

        self.l3 = ConvBatchNormGLU1D(ks=ks_lst[2],
                                     in_channels=out_channels_lst[1],
                                     out_channels=out_channels_lst[2],
                                     pd=pd_lst[2],
                                     dilation=dilation,
                                     stride=stride_lst[2])

        self.l4 = ConvBatchNormGLU1D(ks=ks_lst[3],
                                     in_channels=out_channels_lst[2],
                                     out_channels=out_channels_lst[3],
                                     pd=pd_lst[3],
                                     dilation=dilation,
                                     stride=stride_lst[3])

        self.l5 = nn.Conv1d(in_channels=out_channels_lst[3],
                            out_channels=out_channels_lst[2],
                            kernel_size=ks_lst[4],
                            stride=stride_lst[4],
                            padding=calc_padding(ks_lst[4], dilation, False, stride_lst[4]))

        self.skip = ConvBatchNormGLU1D(ks=4,
                                     in_channels=out_channels_lst[0],
                                     out_channels=out_channels_lst[3],
                                     pd=0,
                                     dilation=dilation,
                                     stride=1)

        self.drop1 = nn.Dropout(p=drop)
        self.drop2 = nn.Dropout(p=drop)
        self.drop3 = nn.Dropout(p=drop)
        self.drop4 = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.drop1(self.l1(x))
        skip = self.skip(x)
        # print(skip.shape)
        x = self.drop2(self.l2(x))
        x = self.drop3(self.l3(x))
        x = self.drop4(self.l4(x))
        x += skip
        x = self.l5(x)
        return x

class ACVAE(nn.Module):
    def __init__(self, enc, dec, classifier, mean=None, std=None, class_embedding=False, n_embeddings=3):
        super(ACVAE, self).__init__()
        self.enc = enc
        self.dec = dec
        self.classifier = classifier
        self.mean = mean
        self.std_ = std
        if class_embedding:
            self.class_emb = nn.Embedding(n_embeddings, 1)

    def forward(self, x_0, y_e, y_d, e_e, e_d, crop_by=False):
        """
        x_0: input spectra
        y_e: encoder labels
        y_d: decoder labels
        e_e: encoder total emissivity
        e_d: encoder total emissivity
        """
        x_0 = x_0.unsqueeze(1)
        a = self.dec(self.enc(x_0, y_e, e_e)[0], y_d, e_d)[0]
        if crop_by:
            return a[:,:,crop_by:-crop_by]
        else:
            return a

    def calc_loss(self, x_0, x_1, y_0, y_1, e_0, e_1, crop_by=False):

        if hasattr(self, 'class_emb'):
            y_0 = torch.argmax(y_0, axis=1)
            y_1 = torch.argmax(y_1, axis=1)
            y_0 = self.class_emb(y_0)
            y_1 = self.class_emb(y_1)

        """
        x_0: input spectra
        x_1: output spectra
        y_0: encoder labels one-hot
        y_1: decoder labels one-hot
        n_classes: number of classes
        """
        x_0 = x_0.unsqueeze(1) #(batch,data) -> (batch, 1 [channel], data)
        x_1 = x_1.unsqueeze(1)
        device = x_0.device

        """ ---- spectra 0 ---- """

        z_mu_s, z_ln_var_s = self.enc(x_0, y_0, e_0, add_total_emis if hasattr(self, 'class_emb') else concat_dim1)
        z_s = gaussian_repar(z_mu_s, z_ln_var_s)

        # reconstruct spectra 0 -> 0
        x_mu_ss, x_ln_var_ss = self.dec(z_s, y_0, e_0)
        x_00 = gaussian_repar(x_mu_ss, x_ln_var_ss)

        # convert spectra 0 -> 1
        x_mu_st, x_ln_var_st = self.dec(z_s, y_1, e_1)
        x_01 = gaussian_repar(x_mu_st, x_ln_var_st)

        """ ---- spectra 1 ---- """

        z_mu_t, z_ln_var_t = self.enc(x_1, y_1, e_1, add_total_emis if hasattr(self, 'class_emb') else concat_dim1)
        z_t = gaussian_repar(z_mu_t, z_ln_var_t)

        # convert spectra 1 -> 0
        x_mu_ts, x_ln_var_ts = self.dec(z_t, y_0, e_0)
        x_10 = gaussian_repar(x_mu_ts, x_ln_var_ts)

        # reconstruct spectra 1 -> 1
        x_mu_tt, x_ln_var_tt = self.dec(z_t, y_1, e_1)
        x_11 = gaussian_repar(x_mu_tt, x_ln_var_tt)

        """ ---- loss ---- """

        vae_loss_prior_s = kl_loss(z_mu_s, z_ln_var_s)
        vae_loss_like_s = gauss_negative_log_like(x_0, x_mu_ss, x_ln_var_ss)

        vae_loss_prior_t = kl_loss(z_mu_t, z_ln_var_t)
        vae_loss_like_t = gauss_negative_log_like(x_1, x_mu_tt, x_ln_var_tt)

        kl_loss_val = (vae_loss_prior_s + vae_loss_prior_t) / 2.0
        gaus_neg_log_like = torch.clamp((vae_loss_like_s + vae_loss_like_t) / 2.0, min=0, max=50)

        """ ---- classification of all reconstructed and converted spectra ---- """
        if crop_by:
            x_1 = x_1[:, :, crop_by:-crop_by]
            x_00 = x_00[:, :, crop_by:-crop_by]
            x_01 = x_01[:, :, crop_by:-crop_by]
            x_10 = x_10[:, :, crop_by:-crop_by]
            x_11 = x_11[:, :, crop_by:-crop_by]
            x_0 = x_0[:, :, crop_by:-crop_by]

        clas_0 = self.classifier(x_0) # (batch, n_class, values)
        clas_1 = self.classifier(x_1)
        clas_00 = self.classifier(x_00)
        clas_01 = self.classifier(x_01)
        clas_10 = self.classifier(x_10)
        clas_11 = self.classifier(x_11)

        clas_0 = clas_0.permute(0,2,1).reshape(-1, clas_0.shape[1])
        clas_1 = clas_1.permute(0,2,1).reshape(-1, clas_1.shape[1])
        clas_00 = clas_00.permute(0,2,1).reshape(-1, clas_00.shape[1])
        clas_01 = clas_01.permute(0,2,1).reshape(-1, clas_01.shape[1])
        clas_10 = clas_10.permute(0,2,1).reshape(-1, clas_10.shape[1])
        clas_11 = clas_11.permute(0,2,1).reshape(-1, clas_11.shape[1])

        clas_0_correct = torch.argmax(y_0, dim=1).cpu().numpy()[0]
        clas_1_correct = torch.argmax(y_1, dim=1).cpu().numpy()[0]

        clas_0_correct = torch.LongTensor(clas_0_correct*np.ones(len(clas_0))).to(device)
        clas_1_correct = torch.LongTensor(clas_1_correct*np.ones(len(clas_0))).to(device)

        clas_0_correct = clas_0_correct.reshape(-1)
        clas_1_correct = clas_1_correct.reshape(-1)

        ClsLoss_class = ((F.cross_entropy(clas_0, clas_0_correct) * clas_0.shape[0] +
                      F.cross_entropy(clas_1, clas_1_correct) * clas_0.shape[0])
                     /
                     (clas_0.shape[0] + clas_1.shape[0]))

        ClsLoss_recon = ((F.cross_entropy(clas_00, clas_0_correct) * clas_00.shape[0] +
                        F.cross_entropy(clas_11, clas_1_correct) * clas_11.shape[0])
                         /
                         (clas_00.shape[0] + clas_11.shape[0]))


        ClsLoss_convert = ((F.cross_entropy(clas_01, clas_1_correct) * clas_01.shape[0] +
                           F.cross_entropy(clas_10, clas_0_correct) * clas_10.shape[0])
                           /
                           (clas_01.shape[0] + clas_10.shape[0]))

        """ --- reconstruct the original spectra --- """

        n_0 = (F.pad(x_0.view(-1, x_0.shape[2]), (crop_by, crop_by, 0, 0), mode='constant', value=0)*self.mean+self.std_)[:, crop_by: -crop_by]
        n_1 = (F.pad(x_1.view(-1, x_1.shape[2]), (crop_by, crop_by, 0, 0), mode='constant', value=0)*self.mean+self.std_)[:, crop_by: -crop_by]

        """ --- find peaks --- """

        n_0_peaks, n_0_mask = find_tensor_highest_peaks(n_0)
        n_1_peaks, n_1_mask = find_tensor_highest_peaks(n_1)

        """ --- reconstruct the "reconstructed internally" spectra --- """

        # loss explanation: z - comparing total emmisivity , y - comparing peaks, n_XX_mse - penalty for negative values

        n_00 = (F.pad(x_00.view(-1, x_00.shape[2]), (crop_by, crop_by, 0, 0), mode='constant', value=0)*self.mean+self.std_)[:, crop_by: -crop_by]
        z_00 = torch.sum(n_00, dim=1)/torch.sum(n_0, dim=1)
        z_00_mse = F.l1_loss(z_00, torch.ones_like(z_00), reduction='mean')

        y_00 = torch.sum(F.relu(n_00)*n_0_mask, dim=1) / torch.sum(n_0_peaks, dim=1)
        y_00_mse = F.l1_loss(y_00, torch.ones_like(y_00), reduction='mean')

        n_00 = torch.sum(torch.relu(-n_00), dim=1) / torch.sum(torch.abs(n_00), dim=1)
        n_00_mse = F.l1_loss(n_00, torch.zeros_like(n_00), reduction='mean')/n_00.shape[0]

        n_11 = (F.pad(x_11.view(-1, x_11.shape[2]), (crop_by, crop_by, 0, 0), mode='constant', value=0)*self.mean+self.std_)[:, crop_by: -crop_by]
        z_11 = torch.sum(n_11, dim=1)/torch.sum(n_1, dim=1)
        z_11_mse = F.l1_loss(z_11, torch.ones_like(z_11), reduction='mean')

        y_11 = torch.sum(F.relu(n_11)*n_1_mask, dim=1) / torch.sum(n_1_peaks, dim=1)
        y_11_mse = F.l1_loss(y_11, torch.ones_like(y_11), reduction='mean')

        n_11 = torch.sum(torch.relu(-n_11), dim=1) / torch.sum(torch.abs(n_11), dim=1)
        n_11_mse = F.l1_loss(n_11, torch.zeros_like(n_11), reduction='mean')/n_11.shape[0]

        n_01 = (F.pad(x_01.view(-1, x_01.shape[2]), (crop_by, crop_by, 0, 0), mode='constant', value=0)*self.mean+self.std_)[:, crop_by: -crop_by]
        z_01 = torch.sum(n_01, dim=1)/torch.sum(n_1, dim=1)
        z_01_mse = F.l1_loss(z_01, torch.ones_like(z_01), reduction='mean')

        y_01 = torch.sum(F.relu(n_01)*n_1_mask, dim=1) / torch.sum(n_1_peaks, dim=1)
        y_01_mse = F.l1_loss(y_01, torch.ones_like(y_01), reduction='mean')

        n_01 = torch.sum(torch.relu(-n_01), dim=1) / torch.sum(torch.abs(n_01), dim=1)
        n_01_mse = F.l1_loss(n_01, torch.zeros_like(n_01), reduction='mean')/n_01.shape[0]

        n_10 = (F.pad(x_10.view(-1, x_10.shape[2]), (crop_by, crop_by, 0, 0), mode='constant', value=0)*self.mean+self.std_)[:, crop_by: -crop_by]
        z_10 = torch.sum(n_10, dim=1)/torch.sum(n_0, dim=1)
        z_10_mse = F.l1_loss(z_10, torch.ones_like(z_10), reduction='mean')

        y_10 = torch.sum(F.relu(n_10)*n_0_mask, dim=1) / torch.sum(n_0_peaks, dim=1)
        y_10_mse = F.l1_loss(y_10, torch.ones_like(y_10), reduction='mean')

        n_10 = torch.sum(torch.relu(-n_10), dim=1) / torch.sum(torch.abs(n_10), dim=1)
        n_10_mse = F.l1_loss(n_10, torch.zeros_like(n_10), reduction='mean')/n_10.shape[0]


        recon_dev_mse = ((z_00_mse + z_11_mse)/2)#*x_0.shape[2]
        convert_dev_mse = ((z_01_mse + z_10_mse)/2)#*x_0.shape[2]

        recon_mse = ((n_00_mse + n_11_mse)/2)#*x_0.shape[2]
        convert_mse = ((n_01_mse + n_10_mse)/2)#*x_0.shape[2]

        recon_p = (y_00_mse + y_11_mse)/2
        convert_p = (y_01_mse + y_10_mse)/2

        loss_d = (recon_mse + convert_mse)*2 + (recon_dev_mse + convert_dev_mse)*1 + (recon_p + convert_p)*1

        return kl_loss_val, gaus_neg_log_like, ClsLoss_class*3, ClsLoss_recon, ClsLoss_convert, loss_d
