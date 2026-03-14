import json
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
from data_visualization import xy_plot, confusion_matrix_plot
from libs_transfer.prepare_data.spectra_normalization import to_onehot
import time
from libs_transfer.prepare_data.spectra_normalization import size_after_conv1d, size_after_avgpool1d
import h5py

epochs = 100
batch_size = 128

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0))
else:
    device = torch.device("cpu")
    print("using CPU, GPU is not available")

device = torch.device("cpu")

with h5py.File('C:\\Users\\Antoni\\PycharmProjects\\Spectra_transfer\\PCA_inliers_mean.h5', 'r') as hf:
    wavelen = np.array(hf['calibration'][()])
    spectra_0_idx = np.where(wavelen == 250)[0][0]
    spectra_last_idx = np.where(wavelen == 850)[0][0]
    conditions = np.array(hf['conditions'][()])
    mars_cond = np.where(conditions[:, 1] == 1)[0]
    mars_spectra = np.array(hf['spectra'][mars_cond, spectra_0_idx:spectra_last_idx+1])#[:, 25: -25]
    # all_spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx+1])[:, 25: -25]
    labels = np.array(hf['labels'][()])

with h5py.File('transformed_data_EtM.h5', 'r') as hf:
    tr_spectra = np.array(hf['spectra'][()])
    tr_labels = np.array(hf['labels'][()])


with open('kmeans_conc_sample_name_to_cluster.json', 'r') as fp:
    sample_name_to_cluster = json.load(fp)

with open('reg_labels_dict.json', 'r') as fp:
    reg_conditions_dict = json.load(fp)

mars_labels = tr_labels
mars_labels = np.argmax(mars_labels, axis=1)
mars_labels = np.array([reg_conditions_dict[str(i)] for i in mars_labels])
mars_ = mars_labels
mars_labels = np.array([sample_name_to_cluster[str(i)] for i in mars_labels])

std = StandardScaler().fit(mars_spectra)
all_spectra = std.transform(tr_spectra)

lbs_dummy, label_dict = to_onehot(mars_labels)

label_dict = {i: str(i) for i in range(lbs_dummy.shape[0])}

#Used for torch DataLoader
class SpectraDataset(Dataset):
    def __init__(self, x, y, z):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.z = z.astype(np.str_)
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index]

    def __len__(self):
        return self.n_samples

test_dataloader = DataLoader(SpectraDataset(all_spectra, lbs_dummy, mars_), batch_size=all_spectra.shape[0], shuffle=True)

class CNN(nn.Module):
    def __init__(self, input_channels, out_channels1, out_channels2, output_size):
        super(CNN, self).__init__()
        self.input_channels = input_channels
        self.out_channels1 = out_channels1
        self.out_channels2 = out_channels2
        self.output_size = output_size
        self.kernel_size = 3
        self.pool = 2

        #feature size after convolutions and pools
        self.size_after_conv1 = size_after_conv1d(mars_spectra.shape[1], out_channels1, self.kernel_size)
        self.size_after_pool1 = size_after_avgpool1d(self.size_after_conv1, self.pool, self.pool)
        self.size_after_conv2 = size_after_conv1d(self.size_after_pool1, out_channels2, self.kernel_size)
        self.size_after_pool2 = size_after_avgpool1d(self.size_after_conv2, self.pool, self.pool)

        self.c1 = nn.Conv1d(input_channels, out_channels1, self.kernel_size, stride=2)
        self.p = nn.AvgPool1d(self.pool, self.pool)
        self.c2 = nn.Conv1d(out_channels1, out_channels2, self.kernel_size, stride=2)

        self.fc1 = nn.Linear(4948, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, output_size)

    def forward(self, input):
        input = input.unsqueeze(0) # (batch, features) -> (1, batch, features)
        input = input.transpose(0, 1) # (1, batch, features ) -> (batch, 1, features), (batch_size x channels x seq_len)
        batch_size = input.size(0)

        x = self.c1(input)
        # x = self.p(x)
        x = self.c2(x)
        # x = self.p(x)

        # Reshape to (batch_size x -1) for fc
        x = x.view(batch_size, -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        output = self.fc5(x)

        return output

model = torch.load('CNN\\CNN_baseline.pth')

pred_list = []
label_list = []
wrong_pred = []
wrong_lab = []
correct_lab = []
with torch.no_grad():
    n_correct = 0
    n_all = 0
    model.to(device)
    model.eval()
    for i, (x, y, lab) in enumerate(test_dataloader):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        predicted = torch.argmax(pred.data, 1) #idx of pred
        sorted_pred = torch.argsort(pred.data, 1, descending=True)
        pred_list.extend(predicted.flatten().tolist())
        labels = torch.argmax(y.data, 1) #idx of y
        label_list.extend(labels.flatten().tolist())
        n_correct += (predicted == labels).sum().item()
        n_all += y.size(0)

        _, top2_indices = torch.topk(pred, 2, dim=1)
        correct_top2 = (top2_indices == labels.unsqueeze(1)).any(dim=1)

        #wrongly predicted idx
        idx_ = predicted != labels
        idx_ = idx_.cpu().detach().numpy()
        idx_ = np.where(idx_ == True)[0]
        lab = list(lab)

        wrong_pred.extend([lab[i] for i in idx_])
        correct_lab.extend(labels[idx_].tolist())
        wrong_lab.extend(predicted[idx_].tolist())

    acc = 100*n_correct/n_all
    print(f'Accuracy of the network: {acc:.3f} %')
    # print(pred_list, label_list)
    accuracy = correct_top2.float().mean()*100
    print(f'top 2 accuracy: {accuracy:.3f} %')

classes = [label_dict[i] for i in range(len(list(set(pred_list))))]
print(pred_list, label_list, classes)
confusion_matrix_plot(pred_list, label_list, classes)

common_wrong_pred = max(set(wrong_pred), key=wrong_pred.count)
common_idx = wrong_pred.index(common_wrong_pred)
correct = correct_lab[common_idx]
all_common = [i for i,a in enumerate(wrong_pred) if a == common_wrong_pred]
wrong = [wrong_lab[i] for i in all_common]
common_wrong = sorted(set(wrong), key=wrong.count)

print(f'most commonly missclassified: {common_wrong_pred}, correct cluster: {correct}, two most common missclassification clusters for the sample: {common_wrong[:2]}')
# Accuracy of the network: 63.784 %
