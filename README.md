**This repository is a bit messy, as there is quite a lot of interest in this work - I will try to make this as easy to use as possible in the nearest future. Here is what you need to know to run the model:**

# Training your own model

## Preparing data
prepare `.h5` file as:
```python
with h5py.File('./spectra.h5', 'w') as hdf:
    hdf.create_dataset('spectra', data=data_np) # ~(n_spectra, spectral_data)
    hdf.create_dataset('calibration', data=calibration) # ~(spectral_data) [nm]

    hdf.create_dataset('atmosphere', data=np.array(atm_).flatten().tolist()) # ~(n_spectra)
    hdf.create_dataset('atm_one_hot', data=atm_one_hot) # ~(n_spectra, n_atmospheres)

    hdf.create_dataset('labels', data=np.array(label_).flatten().tolist()) # ~(n_spectra)
    hdf.create_dataset('labels_one_hot', data=label_one_hot) # ~(n_spectra, n_labels)

    hdf.create_dataset('energy', data=np.array(perc_).flatten().tolist()) # ~(n_spectra)
    hdf.create_dataset('energy_one_hot', data=perc_one_hot) # ~(n_spectra, n_energies)
```

for that you can use `./new_data_to_h5.py`

The final data has to contain the spectra for all the samples, together with the data on sample labels, atmospheres and energies.

## Preparing the Concentration Predition Models for testing the model's performance
run `CNN_conc_baseline.py`

for that you also need to have sample concentrations saved as `Concentrations.xlsx`.

After running the script, save the standard scaler and CNN weights for each of the conditions you are testing.

## Training the model
In order to train the model use `discriminator_new_data.py`.

In there you can choose how you want to preprocess the data before training e.g.

```python
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
```

Whatever you do, the data needed should be assigned to those 3 variables (`all_spectra`, `condtions` and `labels`).

run the training script, uncomment the `torch.save` in order to save model, done.


## Using the model
you may look at `transfer_spectra.py` for reference on how to use the model for spectra transfer.