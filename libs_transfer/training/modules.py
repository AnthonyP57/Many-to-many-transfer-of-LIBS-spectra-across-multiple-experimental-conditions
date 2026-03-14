import torch.nn as nn
from libs_transfer.prepare_data.spectra_normalization import calc_padding
import torch
import math
from torch.utils.data import Dataset
import itertools
from torch.utils.data import  DataLoader
import numpy as np
import random
import os

def create_random_combined_list(list1, list2):
    random_mask = np.random.randint(2, size=list1.shape)
    combined_array = np.where(random_mask, list2, list1)
    return combined_array

class PDSDataset(Dataset):
    def __init__(self, spectra, conditions, labels):
        super().__init__()
        self.spectra = spectra
        self.conditions = conditions
        self.labels = labels

        self.n_cond = set(np.argmax(conditions, axis=1))
        self.n_labels = set(np.argmax(labels, axis=1))

        self._prepare_data()

    def _prepare_data(self):
        data = [[] for _ in self.n_cond]
        for c in self.n_cond:
            c_data=[]
            cond_idx = np.where(np.argmax(self.conditions, axis=1) == c)[0]
            for l in self.n_labels:
                label_idx = np.where(np.argmax(self.labels, axis=1) == l)[0]
                label_idx = [i for i in label_idx if i in cond_idx]
                c_data.append(self.spectra[label_idx])
            c_data = np.concatenate(c_data)
            data[c] = c_data

        data = itertools.combinations(((c, data) for c, data in zip(self.n_cond, data)), 2)
        data = list(data)
        data_pairs=[]
        for pair in data:
            data_pairs.append([torch.tensor(pair[0][1]), torch.tensor(pair[1][1]), torch.tensor(pair[0][0]), torch.tensor(pair[1][0])])

        self.data_pairs = data_pairs

    def __getitem__(self, idx):
        return self.data_pairs[idx]
    
def divide_into_sublists(input_list):
    sublists = {}

    for item in input_list:
        if item not in sublists:
            sublists[item] = []
        sublists[item].append(item)

    lens = list(sublists.values())
    vals = [i[0] for i in lens]
    lens = [len(i) for i in lens]
    return lens, vals

class ModelConfig:
    def __init__(self, checkpoint_path, lr, wd, ks_list, stride_list, channel_list, skip_pad, resume=False, fc=False, pretrain=True):
        self.lr = lr
        self.wd = wd
        self.ks_list = ks_list
        self.stride_list = stride_list
        self.channel_list = channel_list
        self.skip_pad = skip_pad
        self.resume = resume
        self.fc = fc
        self.pretrain = pretrain
        self.checkpoint_path = checkpoint_path
        self.epoch = 0

        if resume and os.path.exists(self.checkpoint_path + f'{self.lr}_{self.wd}_{self.ks_list}_{self.stride_list}_{self.channel_list}_{self.skip_pad}_{self.fc}.pth'):
            config = torch.load(self.checkpoint_path + f'{self.lr}_{self.wd}_{self.ks_list}_{self.stride_list}_{self.channel_list}_{self.skip_pad}_{self.fc}.pth')
            self.loaded_checkpoint = config
            config = config['config']
            self.lr = config['lr']
            self.wd = config['wd']
            self.ks_list = config['ks_list']
            self.stride_list = config['stride_list']
            self.channel_list = config['channel_list']
            self.skip_pad = config['skip_pad']
            self.fc = config['fc']
            self.epoch = config['epoch']
            self.pretrain = config['pretrain']

            print(f'resuming from epoch: {self.epoch}')

    def get_config(self, epoch):
        return {
            'lr': self.lr,
            'wd': self.wd,
            'ks_list': self.ks_list,
            'stride_list': self.stride_list,
            'channel_list': self.channel_list,
            'skip_pad': self.skip_pad,
            'fc': self.fc,
            'pretrain': self.pretrain,
            'epoch': epoch
        }

    def checkpoint(self, model, optimizer, scheduler, epoch):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': self.get_config(epoch+1)},
                   self.checkpoint_path+f'{self.lr}_{self.wd}_{self.ks_list}_{self.stride_list}_{self.channel_list}_{self.skip_pad}_{self.fc}.pth')

def train_test_spectra_samples(conditions, labels, test_split=0.1, seed=42):
    random.seed(seed)
    all_conditions_idx = set(np.argmax(conditions, axis=1))
    all_labels_idx = set(np.argmax(labels, axis=1))
    test_samples = random.sample(list(all_labels_idx), int(len(all_labels_idx) * test_split))
    print(f'test samples idx: {sorted(test_samples)}')

    test_ = []
    train_ = []

    for c in all_conditions_idx:
        condition_idx = set(np.where(np.argmax(conditions, axis=1) == c)[0])

        for l in all_labels_idx:
            label_idx = set(np.where(np.argmax(labels, axis=1) == l)[0])
            label_idx = [i for i in label_idx if i in condition_idx]
            if l in test_samples:
                test_.extend(label_idx)
            else:
                train_.extend(label_idx)

    train_.sort()
    test_.sort()

    return train_, test_

def train_test_spectra_idx(conditions, labels, test_split=0.1, seed=42):
    random.seed(seed)
    all_conditions_idx = set(np.argmax(conditions, axis=1))
    all_labels_idx = set(np.argmax(labels, axis=1))

    test_=[]
    train_=[]

    for c in all_conditions_idx:
        condition_idx = set(np.where(np.argmax(conditions, axis=1) == c)[0])

        for l in all_labels_idx:
            label_idx = set(np.where(np.argmax(labels, axis=1) == l)[0])
            label_idx = [i for i in label_idx if i in condition_idx]
            test_idx = random.sample(label_idx, int(len(label_idx) * test_split))
            train_idx = [i for i in label_idx if i not in test_idx]
            test_.extend(test_idx)
            train_.extend(train_idx)

    return train_, test_
def ACVAE_test_spectra(reg_conditions_dict, conc_df, std, all_labels, all_spectra, total_emis, condition_index_in, condition_index_out, in_idx, out_idx):
    y_in = torch.zeros((len(condition_index_in), 3))
    y_in[:, in_idx] = 1
    y_out = torch.zeros((len(condition_index_in), 3))
    y_out[:, out_idx] = 1

    condition_labels = all_labels[condition_index_in]
    condition_labels_ = np.argmax(condition_labels, axis=1)
    condition_labels = np.array([reg_conditions_dict[str(i)] for i in condition_labels_])
    cond_conc = np.array([conc_df.loc[name].to_list() if name in conc_df.index else [0] * conc_df.shape[1] for name in condition_labels])
    non_zero_idx = np.where(cond_conc[:, 0] != 0)[0]
    cond_conc = cond_conc[non_zero_idx]
    cond_conc = std.transform(cond_conc)
    non_zero_idx_out = list(filter(lambda c: c < len(condition_index_out), non_zero_idx))
    non_zero_idx = non_zero_idx if len(non_zero_idx) < len(non_zero_idx_out) else non_zero_idx_out

    in_to_out = FullDataset(all_spectra[condition_index_in][non_zero_idx], y_in[non_zero_idx], y_out[non_zero_idx]
                                     , total_emis[condition_index_in][non_zero_idx], total_emis[condition_index_out][non_zero_idx])
    in_to_out = DataLoader(in_to_out, batch_size=100, shuffle=False)

    return in_to_out, cond_conc, condition_labels_[non_zero_idx]

def PDS_test_spectra(reg_conditions_dict, conc_df, std, all_labels, all_spectra, condition_index):
    condition_labels = all_labels[condition_index]
    condition_labels = np.argmax(condition_labels, axis=1)
    condition_labels = np.array([reg_conditions_dict[str(i)] for i in condition_labels])
    cond_conc = np.array([conc_df.loc[name].to_list() if name in conc_df.index else [0] * conc_df.shape[1] for name in condition_labels])
    non_zero_idx = np.where(cond_conc[:, 0] != 0)[0]
    cond_conc = cond_conc[non_zero_idx]
    cond_conc = std.transform(cond_conc)

    vacuum100_test = all_spectra[condition_index][non_zero_idx]

    return vacuum100_test, cond_conc

class FullDataset(Dataset):
    def __init__(self, spectra, y_in, y_out, e_in, e_out):
        self.spectra = torch.from_numpy(spectra.astype(np.float32))
        self.y_in = y_in
        self.y_out = y_out
        self.e_in = e_in
        self.e_out = e_out

    def __len__(self):
        return self.spectra.shape[0]

    def __getitem__(self, idx):
        return self.spectra[idx], self.y_in[idx], self.y_out[idx], self.e_in[idx], self.e_out[idx]

class StandardScalerTensor(nn.Module):
    def __init__(self, scaler):
        super(StandardScalerTensor, self).__init__()
        self.register_buffer('mean', torch.tensor(scaler.mean_))
        self.register_buffer('std', torch.tensor(scaler.scale_))

    def forward(self, x):
        return (x - self.mean) / self.std

class ClassDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.x.shape[0]

class SpectraDataset(Dataset):
    def __init__(self, spectra, conditions, labels, total_emissivity, batch_size=64, min_spectra_in_sample=10):
        self.data_samples = None

        all_conditions_idx = set(np.argmax(conditions, axis=1))
        all_labels_idx = set(np.argmax(labels, axis=1))
        n_cond = conditions.shape[1]
        self.n_cond = n_cond

        data=[]
        for i in range(n_cond):
            data.append([0]*(max(all_labels_idx)+1))
        minimum_spectra_n = float('inf')

        total_emis=[]
        for i in range(n_cond):
            total_emis.append([0]*(max(all_labels_idx)+1))

        for c in all_conditions_idx:
            condition_idx = set(np.where(np.argmax(conditions, axis=1) == c)[0])

            for l in all_labels_idx:
                label_idx = set(np.where(np.argmax(labels, axis=1) == l)[0])
                label_idx = [i for i in label_idx if i in condition_idx]
                spc_label = spectra[label_idx]
                emis = total_emissivity[label_idx]

                if spc_label.shape[0] > min_spectra_in_sample:

                    if spc_label.shape[0] < minimum_spectra_n:
                        minimum_spectra_n = spc_label.shape[0]

                    data[c][l] = spc_label
                    total_emis[c][l] = emis

        if batch_size > minimum_spectra_n:
            batch_size = minimum_spectra_n
            print(f'Batch size is changed to {batch_size} due to insufficient number of spectra')

        del spectra, conditions, labels, all_conditions_idx
        empty=[]
        for d in data:
            empty.extend([i for i, x in enumerate(d) if not isinstance(x, np.ndarray)])

        empty_idx = list(set(empty))
        if len(empty_idx) > 0:
            print(f'Samples of idx: {sorted(empty_idx)} dont have sufficient number of spectra')

        data = [(index, d, e) for index, (d, e) in enumerate(zip(data, total_emis))]

        data_pairs_ = list(itertools.combinations(data, 2))
        data_pairs=[]
        for pair in data_pairs_:
            data_pairs.append([pair[0][1], pair[1][1], pair[0][0], pair[1][0], pair[0][2], pair[1][2]]) #data, cond, total_emis

        del data, data_pairs_
        self.data_pairs = data_pairs
        self.all_labels_idx = all_labels_idx
        self.batch_size = batch_size
        self.empty_idx = empty_idx

    def random_select_samples(self, shuffle_data_pairs = False):
        data_samples = []
        if shuffle_data_pairs:
            random.shuffle(self.data_pairs)
        for pair in self.data_pairs:
            for s in self.all_labels_idx:
                if s not in self.empty_idx:
                    rand_idx_0 = random.sample(range(0, pair[0][s].shape[0]), self.batch_size)
                    rand_idx_1 = random.sample(range(0, pair[1][s].shape[0]), self.batch_size)

                    cond_0 = np.zeros((self.batch_size, self.n_cond))
                    cond_0[:, pair[2]] = 1
                    cond_1 = np.zeros((self.batch_size, self.n_cond))
                    cond_1[:, pair[3]] = 1

                    data_samples.append(
                                        (torch.from_numpy(pair[0][s][rand_idx_0].astype(np.float32)), #data0
                                         torch.from_numpy(pair[1][s][rand_idx_1].astype(np.float32)), #data1
                                         torch.from_numpy(cond_0.astype(np.float32)), #y0
                                         torch.from_numpy(cond_1.astype(np.float32)), #y1
                                         torch.from_numpy(pair[4][s][rand_idx_0].astype(np.float32)), #total emis0
                                         torch.from_numpy(pair[5][s][rand_idx_1].astype(np.float32))) #total emis1
                                        )

        if shuffle_data_pairs:
            random.shuffle(data_samples)

        self.data_samples = data_samples

    def __getitem__(self, idx):
        if self.data_samples is None:
            self.random_select_samples()

        return self.data_samples[idx]

    def __len__(self):
        if self.data_samples is None:
            self.random_select_samples()
        return len(self.data_samples)

def gaussian_repar(z_mu, z_lnvar): # Copyright 2021 Hirokazu Kameoka
    device = z_mu.device
    epsilon = torch.randn(z_mu.shape, dtype=torch.float).to(device)
    return z_mu + torch.sqrt(torch.exp(z_lnvar)) * epsilon

def kl_loss(z_mean, z_log_sigma):
    kl_loss = -0.5 * torch.mean(1 + z_log_sigma - z_mean.pow(2) - z_log_sigma.exp())
    return kl_loss

def gauss_negative_log_like(x, x_mu, x_ln_var): # Copyright 2021 Hirokazu Kameoka
    x_prec = torch.exp(-x_ln_var)
    x_diff = x - x_mu
    x_power = (x_diff * x_diff) * x_prec * -0.5
    like_loss = torch.mean((x_ln_var + math.log(2 * math.pi)) / 2 - x_power)
    return like_loss

def calculate_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))

class ConvBatchNormGLU1D(nn.Module):
    def __init__(self, ks, in_channels=1, out_channels=4, pd=0, dilation=1, stride=1):
        super(ConvBatchNormGLU1D, self).__init__()
        self.kernel_size = ks
        self.dilation = dilation
        self.stride = stride
        self.out_channels = out_channels*2
        self.padding = calc_padding(self.kernel_size, self.dilation, False, self.stride) if pd is True else pd

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=ks, padding=self.padding)

        self.bn1 = nn.BatchNorm1d(num_features=self.out_channels)
        self.glu1 = nn.GLU(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.glu1(x)
        return x

class DeConvBatchNormGLU1D(nn.Module):
    def __init__(self, ks, in_channels=1, out_channels=4, pd=0, dilation=1, stride=1):
        super(DeConvBatchNormGLU1D, self).__init__()
        self.kernel_size = ks
        self.dilation = dilation
        self.stride = stride
        self.out_channels = out_channels*2
        self.padding = calc_padding(self.kernel_size, self.dilation, False, self.stride) if pd is True else pd

        self.deconv1 = nn.ConvTranspose1d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=ks, padding=self.padding)

        self.bn1 = nn.BatchNorm1d(num_features=self.out_channels)
        self.glu1 = nn.GLU(dim=1)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.glu1(x)
        return x

def concat_dim1(x, y): # Copyright 2021 Hirokazu Kameoka
    y = y.argmax(dim=1)
    if not torch.all(y == y[0]):
        ValueError('y must represent the same class')
    y = y[0].unsqueeze(0)
    y0 = torch.unsqueeze(torch.unsqueeze(y,0),2)
    N, n_ch, n_t = x.shape
    yy = y0.repeat(N,1,n_t)
    h = torch.cat((x,yy), dim=1)
    return h

def add_total_emis(x,y):
    batch, chan, feat = x.shape
    y = y.unsqueeze(1)
    y = y.repeat(1, 1, feat)
    x = torch.concat((x, y), dim=1)
    return x