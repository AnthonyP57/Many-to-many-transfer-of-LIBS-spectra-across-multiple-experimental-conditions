import numpy as np
from math import ceil
from scipy.signal import find_peaks
from multiprocessing import Pool
import torch
import torch.nn.functional as F
from .data_visualization import xy_plot

def find_peaks_pytorch(tensor, prominence, height, visualize=False):
    batch_size, features = tensor.shape
    peaks_tensor = torch.zeros_like(tensor)

    for i in range(batch_size):
        gradient = tensor[i, 1:] - tensor[i, :-1]
        peak_candidates = torch.nonzero((gradient[:-1] * gradient[1:]) < 0).squeeze(1) + 1
        del gradient
        peaks = peak_candidates[(tensor[i, peak_candidates] > height) &
                                (tensor[i, peak_candidates] > tensor[i, peak_candidates - 1] + prominence) &
                                (tensor[i, peak_candidates] > tensor[i, peak_candidates + 1] + prominence)]

        peaks_tensor[i, peaks] = tensor[i, peaks]

    if visualize:
        xy_plot([[i for i in range(peaks_tensor.shape[1])], [i for i in range(peaks_tensor.shape[1])]], [peaks_tensor.cpu().detach().numpy()[0], tensor.cpu().detach().numpy()[0]])
    
    return peaks_tensor

def find_tensor_highest_peaks(spectra_tensor, n_highest_peaks=10):
    device = spectra_tensor.device
    sorted_tensor, _ = torch.sort(spectra_tensor, descending=True, dim=1)
    threshold = sorted_tensor[:, n_highest_peaks]
    # spectra_tensor = spectra_tensor.unsqueeze(1)
    mean_of_tensor = spectra_tensor.mean().item()

    spectra_tensor_ = find_peaks_pytorch(spectra_tensor, prominence=100, height=mean_of_tensor*2).float()

    # spectra_tensor = torch.where(spectra_tensor <= threshold.unsqueeze(1), torch.tensor(0., dtype=torch.float, device=device), spectra_tensor).float()
    tensor_mask = torch.where(spectra_tensor_ > 0, torch.tensor(1., dtype=torch.float, device=device), torch.tensor(0., dtype=torch.float, device=device))
    del spectra_tensor_, threshold, mean_of_tensor

    kernel = torch.tensor([1,2,3,4,3,2,1], dtype=torch.float, device=device).view(1, 1, -1)
    kernel = kernel / kernel.max()

    spectra_tensor = spectra_tensor.unsqueeze(1)
    tensor_mask = tensor_mask.unsqueeze(1)

    # spectra_tensor_blurred = F.conv1d(spectra_tensor, kernel, padding=2)
    tensor_mask_blurred = F.conv1d(tensor_mask, kernel, padding=kernel.size(-1) // 2)
    spectra_tensor_blurred = spectra_tensor * tensor_mask_blurred
    del spectra_tensor, tensor_mask

    spectra_tensor_blurred = spectra_tensor_blurred.squeeze(1)
    tensor_mask_blurred = tensor_mask_blurred.squeeze(1)

    # xy_plot([[i for i in range(spectra_tensor_blurred.shape[1])], [i for i in range(spectra_tensor_blurred.shape[1])], [i for i in range(spectra_tensor_blurred.shape[1])]],
    #         [spectra_tensor_blurred.cpu().detach().numpy()[0], spectra_tensor.cpu().detach().numpy()[0,0], spectra_tensor_[0].cpu().detach().numpy()])

    return spectra_tensor_blurred, tensor_mask_blurred


def find_noise_mean(idx_start, idx_stop, spectra_as_np, prominence=0.008, height=0.01):
    spectra_as_np = spectra_as_np[idx_start:idx_stop]
    peaks, _ = find_peaks(spectra_as_np, prominence=prominence, height=height)
    noise = [spectra_as_np[i] for i in range(len(spectra_as_np)) if i not in peaks]
    mean_of_noise = np.mean(np.array(noise))

    return mean_of_noise

def normalize_by_total_emissivity(spectra):
    out_spectra = []
    for spc in spectra:
        total_emis = np.sum(spc)
        out_spectra.append(spc / total_emis)

    out_spectra = np.array(out_spectra)
    return out_spectra

def to_onehot(arr):
    """
    Convert a NumPy array with str labels to a one-hot encoded array.

    :return:
    onehot_arr (numpy array): One-hot encoded array with shape (n_samples, n_classes)
    label_dict (dict): Dictionary with indices as keys and label names as values
    """
    arr = arr.flatten()
    n_samples, = arr.shape
    unique_labels = np.unique(arr)
    n_classes = len(unique_labels)

    try:
        label_dict = {i: str(label, 'utf-8') for i, label in enumerate(unique_labels)}
    except:
        label_dict = {i: str(label) for i, label in enumerate(unique_labels)}

    index_dict = {label: i for i, label in enumerate(unique_labels)}

    onehot_arr = np.zeros((n_samples, n_classes))

    for i, label in enumerate(arr):
        onehot_arr[i, index_dict[label]] = 1

    return onehot_arr, label_dict

def calc_padding(kernel_size, dilation, causal=False, stride=1):
    if causal:
        padding = (kernel_size-1)*dilation+1-stride
    else:
        padding = ((kernel_size-1)*dilation+1-stride)//2
    return padding

def find_pixels_idx(dimx = 961, dimy = 994, divide_by = 4):
    """
    the spectral 'picture' is done in a snake pattern, top left to down left, one right then top ... -> |_|-|_ ...
    the 'picture' is divided and the indices of the middle part are found
    :return: list of the indices in the middle of the 'picture', cropped 'picture' xdim, cropped 'picture' ydim
    """
    a = dimx/divide_by
    b = dimy/divide_by
    amin = ceil(a*(divide_by - 1)/2)
    amax = dimx - ceil(a*(divide_by - 1)/2)
    bmin = ceil(b*(divide_by - 1)/2)
    bmax = dimy - ceil(b*(divide_by - 1)/2)

    if amin%2 == 0:
        start_passes = amin
    else:
        start_passes = amin - 1
        amin -= 1

    if amax%2 == 0:
        end_passes = amax + 1
        amax += 1
    else:
        end_passes = amax

    starting_point = start_passes*dimy
    ending_point = end_passes*dimy - 1

    idx = [i for i in range(starting_point, ending_point+1)]
    y_dif_top = dimy - bmax
    y_dif_bot = bmin
    y_dim_difference = bmax - bmin
    x_dim_difference = amax - amin
    idx = idx[y_dif_bot:-y_dif_top]

    delete_indices = []
    len_delete = 2*y_dif_top
    for i in range(0, len(idx) + 1, y_dim_difference+len_delete):
        delete_indices.extend(range(idx[i] + y_dim_difference, idx[i] + y_dim_difference+len_delete))

    delete_indices_set = set(delete_indices)
    idx = [i for i in idx if i not in delete_indices_set]

    return idx, x_dim_difference, y_dim_difference

class FixSpectra:
    def __init__(self, lstart_nm_idx, lstop_np_idx, rstart_nm_idx, rstop_np_idx, bump_start_idx, bump_stop_idx):
        self.lstart_nm_idx = lstart_nm_idx
        self.lstop_nm_idx = lstop_np_idx
        self.rstart_nm_idx = rstart_nm_idx
        self.rstop_nm_idx = rstop_np_idx
        self.bump_start_idx = bump_start_idx
        self.bump_stop_idx = bump_stop_idx

    def __call__(self, sample):
        if len(sample) == 2:
            x, y = sample
        else:
            x = sample

        left_noise_mean = find_noise_mean(self.lstart_nm_idx, self.lstop_nm_idx, x)
        right_noise_mean = find_noise_mean(self.rstart_nm_idx, self.rstop_nm_idx, x)
        bump_noise_mean = find_noise_mean(self.bump_start_idx, self.bump_stop_idx+1, x)

        higher_noise = max(left_noise_mean, right_noise_mean)
        bump_high_dif = bump_noise_mean - higher_noise
        dif = abs(left_noise_mean - right_noise_mean)
        rg = range(self.bump_start_idx, self.bump_stop_idx+1)
        a = dif/len(rg) #gradient step value

        if higher_noise == left_noise_mean:
            correct_gradient = np.array([i*a for i in range(len(rg))])
        else:
            correct_gradient = np.array([dif - i*a for i in range(len(rg))])

        x[self.bump_start_idx:self.bump_stop_idx+1] = x[self.bump_start_idx:self.bump_stop_idx+1] - np.array([bump_high_dif for i in range(len(rg))]) - correct_gradient
        x[self.bump_start_idx- 1:self.bump_start_idx + 1] = x[self.bump_start_idx - 4:self.bump_start_idx - 2]

        if len(sample) == 2:
            return x, y
        else:
            return x

def process_row(process, row):
    return process(row)

def apply_multiprocessing_along_axis(process, data):
    with Pool() as pool:
        result = pool.starmap(process_row, [(process, row) for row in data])

    return np.array(result)