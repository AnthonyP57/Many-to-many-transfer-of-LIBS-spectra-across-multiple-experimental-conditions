from models import Classifier, ACVAE, Encoder, Decoder, Classifier2, MLP, Discriminator, CNN, CNN_d, Discriminator_c
import torch
import h5py
import numpy as np
from data_visualization import xy_plot
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, Dataset, DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0))
else:
    device = torch.device("cpu")
    print("using CPU, GPU is not available")

with h5py.File('C:\\Users\\Antoni\\PycharmProjects\\Spectra_transfer\\PCA_inliers_mean.h5', 'r') as hf:
    wavelen = np.array(hf['calibration'][()])
    spectra_0_idx = np.where(wavelen == 250)[0][0] - 25
    spectra_last_idx = np.where(wavelen == 850)[0][0] + 25
    conditions = np.array(hf['conditions'][()])
    earth_cond = np.where(conditions[:, 0] == 1)[0]
    earth_spectra = np.array(hf['spectra'][earth_cond, spectra_0_idx:spectra_last_idx+1])
    all_spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx+1])
    labels = np.array(hf['labels'][earth_cond])

torch.cuda.empty_cache()

std = StandardScaler().fit(all_spectra)
spectra = std.transform(earth_spectra)

print(earth_spectra.shape)

mean = torch.tensor(std.mean_)
std_ = torch.tensor(std.scale_)


# encoder = Encoder(n_classes_channels=1, in_channels=1, ks_lst=[3,3,3,3], out_channels_lst=[8,16,16,10], stride_lst=[1,1,1,1])
# decoder = Decoder(n_classes_channels=1, in_channels=10, ks_lst=[3,3,3,3][::-1], out_channels_lst=[16,16,8,1], stride_lst=[1,1,1,1])
#
# classifier = Classifier()
# discriminator = CNN_d(spectra.shape[1]-2*25, 1, 4)
# acvae = ACVAE(encoder, decoder, classifier, mean, std_).to(device)

acvae = torch.load('models\\ACVAE_baseline_all_spectra_all_range.pth')

y_in = torch.zeros((spectra.shape[0], 3))
y_in[:, 0] = 1

y_out = torch.zeros((spectra.shape[0], 3))
y_out[:, 1] = 1

class FullDataset(Dataset):
    def __init__(self, spectra, y_in, y_out):
        self.spectra = torch.from_numpy(spectra.astype(np.float32))
        self.y_in = y_in
        self.y_out = y_out
    def __len__(self):
        return self.spectra.shape[0]
    def __getitem__(self, idx):
        return self.spectra[idx], self.y_in[idx], self.y_out[idx]

ds = FullDataset(spectra, y_in, y_out)
ds = DataLoader(ds, batch_size=100, shuffle=False)

all_spectra = []

with torch.no_grad():

    for x_0, y_0, y_1 in ds:
        x_0 = x_0.to(device)
        y_0 = y_0.to(device)
        y_1 = y_1.to(device)
        x_01 = acvae(x_0, y_0, y_1, crop_by=25)
        x_01_ = x_01.cpu().detach().numpy()
        all_spectra.append(x_01_)

vis = std.inverse_transform(np.pad(x_01.view(-1, x_01.shape[2]).cpu().detach().numpy(), ((0,0), (25,25)), mode='constant'))[:, 25:-25]
org = std.inverse_transform(x_0.cpu().detach().numpy())
xy_plot([wavelen[spectra_0_idx:spectra_last_idx+1][25:-25] for i in range(4)], np.concatenate((vis[:2], org[:, 25:-25][:2]), axis=0))

all_spectra_np = np.concatenate(all_spectra, axis=0)
all_spectra_np = all_spectra_np.reshape(all_spectra_np.shape[0], -1)
all_spectra_np = std.inverse_transform(np.pad(all_spectra_np, ((0,0), (25,25)), mode='constant'))[:, 25:-25]

print(all_spectra_np.shape, labels.shape)
xy_plot([wavelen[spectra_0_idx:spectra_last_idx+1][25:-25] for i in range(4)], all_spectra_np[:4])

with h5py.File('transformed_data_EtM.h5', 'w') as hf:
    hf.create_dataset('spectra', data=all_spectra_np)
    hf.create_dataset('labels', data=labels)

print(11, all_spectra_np.shape, labels.shape, 11)