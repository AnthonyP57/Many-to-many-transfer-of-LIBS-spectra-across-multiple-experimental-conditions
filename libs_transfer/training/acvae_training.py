import os
import json
import itertools

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

from libs_transfer.training.models import Classifier, ACVAE, Encoder, Decoder
from libs_transfer.training.modules import SpectraDataset, ClassDataset, train_test_spectra_samples, ACVAE_test_spectra, ModelConfig
from libs_transfer.training.CNN_conc_baseline import CNN

from libs_transfer.training.evaluation_metrics import (
    compute_mean_reference_spectra, build_reference_stack, 
    get_raw_spectra_subsets, build_raw_reference_stack, 
    PCAMeanSpectraEvaluator, PCASpectraEvaluator, 
    KMeansMeanSpectraEvaluator, KMeansRawSpectraEvaluator
)


def prepare_training_data(path, wavelength_range=(250, 850), exclude_id=None):
    with h5py.File(path, 'r') as hf:
        wavelen = np.array(hf['calibration'][()])
        spectra_0_idx = np.where(wavelen == wavelength_range[0])[0][0] - 25
        spectra_last_idx = np.where(wavelen == wavelength_range[1])[0][0] + 25

        labels = np.array(hf['labels_one_hot'])
        all_spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx+1])
        all_atm = np.array(hf['atm_one_hot'][()])
        all_ene = np.array(hf['energy_one_hot'][()])

        conditions = np.array([np.kron(row1, row2) for row1, row2 in zip(all_atm, all_ene)])

        if exclude_id is not None:
            c = np.where(np.argmax(conditions, axis=1) != exclude_id)[0]
            
            all_spectra = all_spectra[c]
            labels = labels[c]
            
            cols_to_keep = [i for i in range(conditions.shape[1]) if i != exclude_id]
            conditions = conditions[c][:, cols_to_keep]

    return all_spectra, conditions, labels, wavelen, spectra_0_idx, spectra_last_idx

def preprocess_training_data(all_spectra):
    emis_norm = all_spectra / np.sum(all_spectra, axis=1, keepdims=True)
    emis_std = StandardScaler().fit(emis_norm)
    del emis_norm

    all_spectra = np.maximum(all_spectra, 0) # relu
    total_emis = np.expand_dims(np.sum(all_spectra, 1), 1)

    std = StandardScaler().fit(all_spectra)
    all_spectra = std.transform(all_spectra)

    std_emis = MinMaxScaler().fit(total_emis)
    total_emis = std_emis.transform(total_emis)

    mean = torch.tensor(std.mean_)
    std_ = torch.tensor(std.scale_)

    return all_spectra, emis_std, total_emis, mean, std_, std

def load_metadata(data_dir):
    with open(os.path.join(data_dir, 'label_dict.json'), 'r') as fp:
        reg_conditions_dict = json.load(fp)
    with open(os.path.join(data_dir, 'atm_dict.json'), 'r') as fp:
        atm_dict = json.load(fp)
    with open(os.path.join(data_dir, 'perc_dict.json'), 'r') as fp:
        en_dict = json.load(fp)

    conc_df = pd.read_excel(os.path.join(data_dir, 'Concentrations.xlsx'))
    conc_df = conc_df.dropna().transpose()
    conc_df.columns = [a.upper() for a in conc_df.iloc[0].tolist()]
    conc_df = conc_df.iloc[1:]
    conc_df.index = [a.upper() for a in conc_df.index]
    
    return reg_conditions_dict, atm_dict, en_dict, conc_df

def load_auxiliary_models(spectra_shape, conditions, test_idx, atm_dict, en_dict, data_dir, device):
    num_energies = len(en_dict)
    num_conditions = conditions.shape[1]

    models_dict = {}
    scalers_dict = {}
    indices_dict = {}

    for i in range(num_conditions):
        atm_idx = i // num_energies
        en_idx = i % num_energies
        
        condition_text = atm_dict[str(atm_idx)]
        energy_text = en_dict[str(en_idx)]
        
        indices_dict[i] = np.where(np.argmax(conditions[test_idx], axis=1) == i)[0]
        
        scaler_path = os.path.join(data_dir, f'conc_std_{condition_text}_{energy_text}.joblib')
        if os.path.exists(scaler_path):
            scalers_dict[i] = joblib.load(scaler_path)
        else:
            continue

        output_shape = scalers_dict[i].scale_.shape[0] 
        model_path = os.path.join(data_dir, f'CNN_{condition_text}_{energy_text}_state_dict.pth')
        
        if os.path.exists(model_path):
            model = CNN(output_shape, inshape=spectra_shape - 2*25)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.to(device)
            model.eval()
            models_dict[i] = model
            
    return models_dict, scalers_dict, indices_dict

def generate_transfer_pairs(num_conditions, scalers_dict, reg_conditions_dict, conc_df, labels, all_spectra, total_emis, test_idx, indices_dict):
    transfer_dict = {}
    permutations = list(itertools.permutations(range(num_conditions), 2))

    for src_idx, tgt_idx in permutations:
        if src_idx not in scalers_dict or tgt_idx not in scalers_dict:
            continue

        in_to_out, cond_conc, cond_labels = ACVAE_test_spectra(
            reg_conditions_dict, conc_df, scalers_dict[src_idx], 
            labels[test_idx], all_spectra[test_idx], total_emis[test_idx], 
            indices_dict[src_idx], indices_dict[tgt_idx], src_idx, tgt_idx,
            num_conditions
        )
        
        transfer_dict[(src_idx, tgt_idx)] = {
            'data': in_to_out,
            'conc': cond_conc,
            'labels': cond_labels
        }

    return transfer_dict, permutations

def build_acvae_model(config, conditions_shape, mean, std_, device, n_conditions):
    classifier = Classifier(n_classes=conditions_shape[1]).to(device)
    mean = mean.to(device)
    std_ = std_.to(device)
    encoder = Encoder(n_classes_channels=1, in_channels=1, ks_lst=config.ks_list, out_channels_lst=config.channel_list, stride_lst=config.stride_list, skip_pad=config.skip_pad, add_fc=config.fc)
    decoder = Decoder(n_classes_channels=1, in_channels=config.channel_list[-1], ks_lst=config.ks_list[::-1], out_channels_lst=config.channel_list[:-1][1:]+[1], stride_lst=config.stride_list[::-1], skip_pad=config.skip_pad, add_fc=config.fc)

    for p in encoder.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)
    for p in decoder.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)

    acvae = ACVAE(encoder, decoder, classifier, mean, std_, True, n_conditions).to(device)
    acvae = torch.compile(acvae)
    
    optimizer = torch.optim.Adam(acvae.parameters(), lr=config.lr, weight_decay=config.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3)
    
    return acvae, classifier, optimizer, scheduler

def pretrain_classifier(classifier, class_dataset, device):
    clas_optim = torch.optim.Adam(classifier.parameters(), lr=5e-3)
    clas_sched = torch.optim.lr_scheduler.CosineAnnealingLR(clas_optim, T_max=5)
    
    for i in range(0): 
        epoch_loss, n_all = 0, 0
        for x, y in class_dataset:
            x = x[:, 25:-25].unsqueeze(1).to(device)
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

def train_epoch(acvae, ds, optimizer, device):
    acvae.train()
    metrics = {k: 0 for k in ['kl', 'gaus', 'clas', 'recon', 'convert', 'd']}
    n_all = 0
    
    for x_0, x_1, y_0, y_1, e_0, e_1 in ds:

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            x_0, x_1 = x_0.to(device), x_1.to(device)
            y_0, y_1 = y_0.to(device), y_1.to(device)
            e_0, e_1 = e_0.to(device), e_1.to(device)

            kl_loss, gaus_neg_log_like, ClsLoss_class, ClsLoss_recon, ClsLoss_convert, loss_d = acvae.calc_loss(x_0, x_1, y_0, y_1, e_0, e_1, crop_by=25)

            vae_loss = gaus_neg_log_like + kl_loss + ClsLoss_class + ClsLoss_recon + ClsLoss_convert + loss_d
            
            a = x_0.shape[0]
            metrics['kl'] += kl_loss * a
            metrics['gaus'] += gaus_neg_log_like * a
            metrics['clas'] += ClsLoss_class * a
            metrics['recon'] += ClsLoss_recon * a
            metrics['convert'] += ClsLoss_convert * a
            metrics['d'] += loss_d * a
            n_all += a

        optimizer.zero_grad()
        vae_loss.backward()
        optimizer.step()
        
    return {k: v/n_all for k, v in metrics.items()}

def evaluate_acvae(acvae, models_dict, scalers_dict, transfer_dict, permutations, std, emis_std, labels, test_idx, device, eval_km_mean, eval_km_raw, eval_cos_mean, eval_cos_raw):
    acvae.eval()
    all_final_ys, all_final_preds, all_cnn_preds = [], [], []
    test_spectra_mean_list, test_spectra_list = [], []
    target_labels = np.unique(np.argmax(labels[test_idx], axis=1)) 
    
    with torch.no_grad():
        for src, tgt in permutations:
            if tgt not in models_dict: continue
            
            model = models_dict[tgt]
            scaler = scalers_dict[tgt]
            data_batch = transfer_dict[(src, tgt)]['data']
            source_conc = transfer_dict[(src, tgt)]['conc']
            cond_labels = transfer_dict[(src, tgt)]['labels']
            
            target_transformed_list = []
            target_orig_preds = []
            
            for x_0, y_0, y_1, e_0, e_1 in data_batch:
                x_01 = acvae(x_0.to(device), y_0.to(device), y_1.to(device), e_0.to(device), e_1.to(device), crop_by=25)
                x_01 = x_01.cpu().detach().numpy().reshape(x_01.shape[0], -1)
                x_01 = np.maximum(std.inverse_transform(np.pad(x_01, ((0, 0), (25, 25)), mode='constant')), 0)
                x_01 = emis_std.transform(x_01 / np.sum(x_01, axis=1, keepdims=True))[:, 25:-25]
                target_transformed_list.append(x_01.reshape(-1, x_01.shape[-1]))

                x_0_orig = emis_std.transform(std.inverse_transform(x_0.cpu().detach()) / np.sum(std.inverse_transform(x_0.cpu().detach()), axis=1, keepdims=True))[:, 25:-25]
                pred = model(torch.tensor(x_0_orig, dtype=torch.float).to(device))
                target_orig_preds.append(scaler.inverse_transform(pred.cpu().detach().numpy()))

            if not target_transformed_list: continue
            
            target_transformed = np.concatenate(target_transformed_list, axis=0)
            target_orig_preds = np.concatenate(target_orig_preds, axis=0)
            
            test_spectra_list.append(target_transformed)
            
            for lab in target_labels:
                lab_indices = np.where(cond_labels == lab)[0]
                if len(lab_indices) > 0:
                    test_spectra_mean_list.append(np.mean(target_transformed[lab_indices], axis=0))
                else:
                    test_spectra_mean_list.append(np.zeros_like(target_transformed[0]))

            pair_preds, pair_ys = [], []
            test_dataloader = DataLoader(ClassDataset(target_transformed, source_conc), batch_size=25)
            for x, y in test_dataloader:
                pred = model(x.to(device))
                pair_preds.append(scaler.inverse_transform(pred.cpu().detach().numpy()))
                pair_ys.append(scaler.inverse_transform(y.numpy()))
                
            all_final_ys.append(np.concatenate(pair_ys, axis=0))
            all_final_preds.append(np.concatenate(pair_preds, axis=0))
            all_cnn_preds.append(target_orig_preds)

    ys = np.concatenate(all_final_ys, axis=0) if all_final_ys else np.array([])
    preds = np.concatenate(all_final_preds, axis=0) if all_final_preds else np.array([])
    cnn_preds = np.concatenate(all_cnn_preds, axis=0) if all_cnn_preds else np.array([])
    
    rmse, rmse_cnn = None, None
    if len(ys) > 0 and len(preds) > 0:
        rmse = np.round(np.sqrt(np.mean((ys - preds) ** 2, axis=0)), 3)
        rmse_cnn = np.round(np.sqrt(np.mean((cnn_preds - preds) ** 2, axis=0)), 3)

    clus_acc, clus_loss, clus_acc_, clus_loss_, cosine_sim, cosine_sim_ = 0,0,0,0,0,0
    if test_spectra_mean_list:
        test_spectra_mean = np.vstack(test_spectra_mean_list)
        test_spectra = np.concatenate(test_spectra_list, axis=0)
        
        clus_acc, clus_loss = eval_km_mean.calc_kmeans_loss(test_spectra_mean)
        cosine_sim = eval_cos_mean.cosine_similarity(test_spectra_mean)
        
        clus_acc_, clus_loss_ = eval_km_raw.calc_kmeans_loss(test_spectra)
        cosine_sim_ = eval_cos_raw.cosine_similarity(test_spectra)

    return rmse, rmse_cnn, clus_acc, clus_acc_, cosine_sim, cosine_sim_

def train_acvae_pipeline(data_path='./examples/processed_data/spectra.h5', data_dir='./examples/processed_data/', epochs=5, batch_size=64, device=None, test_split=0.5):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")

    all_spectra, conditions, labels, wavelen, s0_idx, slast_idx = prepare_training_data(data_path)
    all_spectra, emis_std, total_emis, mean, std_, std = preprocess_training_data(all_spectra)
    train_idx, test_idx = train_test_spectra_samples(conditions, labels, test_split=test_split)

    ds = SpectraDataset(all_spectra[train_idx], conditions[train_idx], labels[train_idx], total_emis[train_idx], batch_size=batch_size)
    ds.random_select_samples(shuffle_data_pairs=True)

    reg_conditions_dict, atm_dict, en_dict, conc_df = load_metadata(data_dir)
    models_dict, scalers_dict, indices_dict = load_auxiliary_models(all_spectra.shape[1], conditions, test_idx, atm_dict, en_dict, data_dir, device)
    
    num_conditions = conditions.shape[1]
    transfer_dict, permutations = generate_transfer_pairs(
        num_conditions, scalers_dict, reg_conditions_dict, conc_df, 
        labels, all_spectra, total_emis, test_idx, indices_dict
    )

    target_labels = np.unique(np.argmax(labels[test_idx], axis=1)) 

    all_spectra_cropped = all_spectra[:, 25:-25]
    
    mean_spectra_dict = compute_mean_reference_spectra(all_spectra_cropped, conditions, labels, target_labels)
    stacked_mean_ref = build_reference_stack(mean_spectra_dict, num_conditions, target_labels)
    
    raw_spectra_dict = get_raw_spectra_subsets(all_spectra_cropped, conditions, labels, target_labels)
    stacked_raw_ref = build_raw_reference_stack(raw_spectra_dict, num_conditions, target_labels)
    
    eval_cos_mean = PCAMeanSpectraEvaluator(stacked_mean_ref, n_components=15)
    eval_cos_raw = PCASpectraEvaluator(stacked_raw_ref, n_components=15)
    eval_km_mean = KMeansMeanSpectraEvaluator(stacked_mean_ref, n_components=15, k_clusters=6)
    eval_km_raw = KMeansRawSpectraEvaluator(stacked_raw_ref, n_components=15, k_clusters=6)

    config = ModelConfig(checkpoint_path='./examples/model/checkpoints/', lr=5e-2, wd=1e-6, ks_list=[16]*7, stride_list=[1]*7, channel_list=[8]*7, skip_pad=6, resume=False, fc=False, pretrain=True)
    acvae, classifier, optimizer, scheduler = build_acvae_model(config, conditions, mean, std_, device, num_conditions)
    
    if not hasattr(config, 'loaded_checkpoint') and config.pretrain:
        class_dataset = DataLoader(ClassDataset(all_spectra[train_idx], conditions[train_idx]), batch_size=250, shuffle=True)
        pretrain_classifier(classifier, class_dataset, device)

    iterator = tqdm(range(config.epoch if hasattr(config, 'epoch') else 0, epochs), desc='model training')
    
    for epoch in iterator:
        torch.cuda.empty_cache()
        
        metrics = train_epoch(acvae, ds, optimizer, device)
        scheduler.step()
        
        losses = {k: f'{v:.5f}' for k, v in metrics.items()}
        ds.random_select_samples(shuffle_data_pairs=True)
        config.checkpoint(acvae, optimizer, scheduler, epoch)
        iterator.set_postfix(losses)

        rmse, rmse_cnn, c_acc, c_acc_, c_sim, c_sim_ = evaluate_acvae(
            acvae, models_dict, scalers_dict, transfer_dict, permutations, 
            std, emis_std, labels, test_idx, device,
            eval_km_mean, eval_km_raw, eval_cos_mean, eval_cos_raw
        )
        
        if rmse is not None:
            iterator.write(f'\nRMSE mean: {round(np.mean(rmse), 3):.3f} | RMSE mean CNN: {round(np.mean(rmse_cnn), 3):.3f}')
        iterator.write(f'Clustering Acc (Mean): {c_acc:.3f} | Cosine Sim (Mean): {c_sim:.3f}')
        iterator.write(f'Clustering Acc (All): {c_acc_:.3f} | Cosine Sim (All): {c_sim_:.3f}\n')

    return acvae
