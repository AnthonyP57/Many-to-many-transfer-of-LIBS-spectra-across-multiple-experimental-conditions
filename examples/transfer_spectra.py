import os
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from libs_transfer.prepare_data.data_visualization import xy_plot
from libs_transfer.training.models import Classifier, ACVAE, Encoder, Decoder
from libs_transfer.training.modules import ModelConfig

class InferenceDataset(Dataset):
    def __init__(self, spectra, y_in, y_out, e_in, e_out):
        self.spectra = torch.from_numpy(spectra.astype(np.float32))
        self.y_in = torch.from_numpy(y_in.astype(np.float32))
        self.y_out = torch.from_numpy(y_out.astype(np.float32))
        self.e_in = torch.from_numpy(e_in.astype(np.float32))
        self.e_out = torch.from_numpy(e_out.astype(np.float32))

    def __len__(self):
        return self.spectra.shape[0]

    def __getitem__(self, idx):
        return self.spectra[idx], self.y_in[idx], self.y_out[idx], self.e_in[idx], self.e_out[idx]


def setup_inference_environment(data_path, metadata_dir, exclude_id=None):
    with h5py.File(data_path, 'r') as hf:
        wavelen = np.array(hf['calibration'][()])
        spectra_0_idx = np.where(wavelen == 250)[0][0] - 25
        spectra_last_idx = np.where(wavelen == 850)[0][0] + 25

        labels = np.array(hf['labels_one_hot'])
        all_spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx+1])
        all_atm = np.array(hf['atm_one_hot'][()])
        all_ene = np.array(hf['energy_one_hot'][()])
        conditions = np.array([np.kron(row1, row2) for row1, row2 in zip(all_atm, all_ene)])

        cols_to_keep = list(range(conditions.shape[1]))

        if exclude_id is not None:
            c = np.where(np.argmax(conditions, axis=1) != exclude_id)[0]
            all_spectra = all_spectra[c]
            labels = labels[c]
            cols_to_keep = [i for i in range(conditions.shape[1]) if i != exclude_id]
            conditions = conditions[c][:, cols_to_keep]

    all_spectra = np.maximum(all_spectra, 0)
    total_emis = np.expand_dims(np.sum(all_spectra, 1), 1)

    std = StandardScaler().fit(all_spectra)
    std_emis = MinMaxScaler().fit(total_emis)

    mean = torch.tensor(std.mean_)
    std_ = torch.tensor(std.scale_)

    with open(os.path.join(metadata_dir, 'atm_dict.json'), 'r') as fp:
        atm_dict = json.load(fp)
    with open(os.path.join(metadata_dir, 'perc_dict.json'), 'r') as fp:
        en_dict = json.load(fp)
        
    atm_lookup = {v: int(k) for k, v in atm_dict.items()}
    en_lookup = {v: int(k) for k, v in en_dict.items()}

    return all_spectra, conditions, labels, wavelen, spectra_0_idx, spectra_last_idx, std, std_emis, mean, std_, atm_lookup, en_lookup, cols_to_keep


def get_condition_index(atm_name, ene_name, atm_lookup, en_lookup, num_energies):
    atm_idx = atm_lookup[atm_name]
    ene_idx = en_lookup[ene_name]
    return (atm_idx * num_energies) + ene_idx

def transfer_spectra(model, source_data, source_emis, src_idx, tgt_idx, num_conditions, std, device, batch_size=100):
    model.eval()
    
    y_in = np.zeros((source_data.shape[0], num_conditions))
    y_in[:, src_idx] = 1
    y_out = np.zeros((source_data.shape[0], num_conditions))
    y_out[:, tgt_idx] = 1

    ds = InferenceDataset(source_data, y_in, y_out, source_emis, source_emis)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    transferred_spectra = []
    
    with torch.no_grad():
        for x_0, y_0, y_1, e_0, e_1 in dataloader:
            x_0 = x_0.to(device)
            y_0 = y_0.to(device)
            y_1 = y_1.to(device)
            e_0 = e_0.to(device)
            e_1 = e_1.to(device)
            
            x_01 = model(x_0, y_0, y_1, e_0, e_1, crop_by=25)
            
            x_01_np = x_01.cpu().detach().numpy().reshape(x_01.shape[0], -1)
            x_01_np = np.maximum(std.inverse_transform(np.pad(x_01_np, ((0, 0), (25, 25)), mode='constant')), 0)
            
            transferred_spectra.append(x_01_np[:, 25:-25])

    return np.concatenate(transferred_spectra, axis=0)


if __name__ == "__main__":
    DATA_PATH = './examples/processed_data/spectra.h5'
    META_DIR = './examples/processed_data/'
    MODEL_PATH = './examples/model/checkpoints/0.05_1e-06_[16, 16, 16, 16, 16, 16, 16]_[1, 1, 1, 1, 1, 1, 1]_[8, 8, 8, 8, 8, 8, 8]_6_False.pth'
    OUTPUT_H5 = './transformed_data.h5'
    
    SOURCE_ATM = 'ATM'
    SOURCE_ENE = '50%'
    TARGET_ATM = '700_PA'
    TARGET_ENE = '100%'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    (all_spectra, conditions, labels, wavelen, 
    s0_idx, slast_idx, std, std_emis, mean, std_, 
    atm_lookup, en_lookup, cols_to_keep) = setup_inference_environment(DATA_PATH, META_DIR)

    num_conditions = conditions.shape[1]
    num_energies = len(en_lookup)

    raw_src_idx = get_condition_index(SOURCE_ATM, SOURCE_ENE, atm_lookup, en_lookup, num_energies)
    raw_tgt_idx = get_condition_index(TARGET_ATM, TARGET_ENE, atm_lookup, en_lookup, num_energies)

    if raw_src_idx not in cols_to_keep or raw_tgt_idx not in cols_to_keep:
        raise ValueError(f"Cannot transfer: Source or Target condition was excluded (exclude_id={0})!")
        
    src_idx = cols_to_keep.index(raw_src_idx)
    tgt_idx = cols_to_keep.index(raw_tgt_idx)

    print(f"Transferring {SOURCE_ATM}_{SOURCE_ENE} (Raw: {raw_src_idx} -> Shifted: {src_idx}) "
            f"to {TARGET_ATM}_{TARGET_ENE} (Raw: {raw_tgt_idx} -> Shifted: {tgt_idx})")

    source_mask = np.where(np.argmax(conditions, axis=1) == src_idx)[0]
    source_spectra_scaled = all_spectra[source_mask]
    source_labels = labels[source_mask]
    
    raw_source_spectra = std.inverse_transform(source_spectra_scaled)
    raw_source_spectra = np.maximum(raw_source_spectra, 0)
    source_total_emis = np.expand_dims(np.sum(raw_source_spectra, 1), 1)
    source_total_emis_scaled = std_emis.transform(source_total_emis)

    try:
        acvae = torch.load(MODEL_PATH, map_location=device)
        if isinstance(acvae, dict):
            raise TypeError("Loaded a dictionary, falling back to state_dict load.")
        acvae.eval()
    except Exception as e:
        print(f"Direct load failed, attempting state_dict load... ({e})")
        
        ckpt_dir = os.path.dirname(MODEL_PATH)
        config = ModelConfig(checkpoint_path=ckpt_dir, lr=5e-2, wd=1e-6, ks_list=[16]*7, stride_list=[1]*7, channel_list=[8]*7, skip_pad=6)
        
        classifier = Classifier(n_classes=num_conditions).to(device)
        encoder = Encoder(n_classes_channels=1, in_channels=1, ks_lst=config.ks_list, out_channels_lst=config.channel_list, stride_lst=config.stride_list, skip_pad=config.skip_pad, add_fc=config.fc)
        decoder = Decoder(n_classes_channels=1, in_channels=config.channel_list[-1], ks_lst=config.ks_list[::-1], out_channels_lst=config.channel_list[:-1][1:]+[1], stride_lst=config.stride_list[::-1], skip_pad=config.skip_pad, add_fc=config.fc)
        
        acvae = ACVAE(encoder, decoder, classifier, mean.to(device), std_.to(device), True, num_conditions).to(device)
        
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        acvae.load_state_dict(clean_state_dict)
        acvae.eval()

    transferred_spectra = transfer_spectra(
        acvae, source_spectra_scaled, source_total_emis_scaled, 
        src_idx, tgt_idx, num_conditions, std, device
    )

    print("Generating plots...")
    original_physical = raw_source_spectra[:, 25:-25]
    wavelength_axis = wavelen[s0_idx:slast_idx+1][25:-25]
    
    plot_data = np.concatenate((transferred_spectra[:2], original_physical[:2]), axis=0)
    xy_plot(
        [wavelength_axis for _ in range(4)], 
        plot_data,
        size = (21, 14),
        labels=[f'Trans.{TARGET_ATM}', f'Trans.{TARGET_ATM}', f'Org.{SOURCE_ATM}', f'Org.{SOURCE_ATM}'],
        xy_labels=['Wavelength [nm]', 'Intensity [a.u.]'],
        title=f'Spectra Transfer: {SOURCE_ATM}_{SOURCE_ENE} to {TARGET_ATM}_{TARGET_ENE}',
        save_to=f'./{SOURCE_ATM}_{SOURCE_ENE}_to_{TARGET_ATM}_{TARGET_ENE}.png'
    )

    print(f"Saving transferred spectra to {OUTPUT_H5}")
    with h5py.File(OUTPUT_H5, 'w') as hf:
        hf.create_dataset('spectra', data=transferred_spectra)
        hf.create_dataset('labels_one_hot', data=source_labels)
