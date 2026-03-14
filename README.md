# Spectra Transfer ACVAE
This repository is actively being refactored to make it as easy to use as possible. Here is everything you need to know to prepare your data, train the baseline predictors, and train the ACVAE model from scratch.

## Quick Start: The Full Pipeline
If your data directories are already set up, you can run the entire data extraction, CNN baseline training, and ACVAE evaluation pipeline in just a few lines of code.

```python
from libs_transfer.training import train_concentration_predictors
from libs_transfer.prepare_data import data_to_h5
from libs_transfer.training import train_acvae_pipeline
import warnings
import torch

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.manual_seed(123456)

# 1. Parse raw text files into an HDF5 dataset
data_to_h5('./examples/example_raw_data', './examples/processed_data')

# 2. Train CNN Concentration Predictors for all conditions
train_concentration_predictors(data_folder='./examples/processed_data')

# 3. Train and Evaluate the ACVAE
acvae = train_acvae_pipeline(test_split=0.5)
```

### Preparing the Data
Your raw data should be in standard tab-separated text files (e.g., exported from a LIBS Discovery system). The first column represents the wavelength ([nm]), and the subsequent columns represent the intensity ([a.u.]) of separete measurements.

Example Al2O3-AM.txt format:

```bash
  [nm]    [a.u]        [a.u]
200.000 47.646484   -17.080078 ...
200.020 98.522461   -17.432617 ...
...
```
Organize your raw .txt files into a nested directory structure so the script can automatically label your data by Atmosphere, Energy, and Sample Name:

```bash
raw_data_folder/
├── VACUUM/                 <-- Atmosphere
│   ├── 100/                <-- Energy
│   │   ├── Al2O3.txt       <-- Sample Name
│   │   └── SiO2.txt
├── EARTH/
│   ├── 50/
│   │   ├── Al2O3.txt
...
```

Use the `data_to_h5` function to parse this folder and automatically generate the .h5 dataset and corresponding .json dictionaries:

```python
from libs_transfer.prepare_data import data_to_h5

# This generates spectra.h5, atm_dict.json, perc_dict.json, and label_dict.json
data_to_h5(in_path='./raw_data_folder', out_folder='./processed_data')
```

### Preparing the Concentration Predictors (CNN)
To evaluate how well the ACVAE transfers spectra between conditions, the pipeline requires baseline CNN models to predict chemical concentrations.

Ensure you have a Concentrations.xlsx file saved in your processed data folder. The first row should contain the element names (e.g., SIO2, AL2O3), and the first column should contain the sample names matching your .txt files.

Run the CNN training script. The script will automatically discover every Atmosphere + Energy combination in your .h5 file and train a specific CNN for each one.

```python
from libs_transfer.training.CNN_conc_baseline import train_concentration_predictors

# Trains and saves a CNN and standard scaler for EVERY condition in spectra.h5
train_concentration_predictors(
    epochs=40, 
    batch_size=128, 
    data_folder='./processed_data'
)
```

After running this, your data folder will be populated with files like `CNN_EARTH_50_state_dict.pth` and `conc_std_EARTH_50.joblib`.

### Training the ACVAE Model
Once your .h5 data and baseline CNNs are prepared, training the ACVAE is entirely automated.

The refactored train_acvae_pipeline will automatically:

Load your .h5 file and metadata dictionaries.

Detect all available conditions and load the corresponding CNN predictors.

Dynamically generate permutations for every possible transfer direction (e.g., VACUUM 100 $\rightarrow$ EARTH 50, EARTH 100 $\rightarrow$ EARTH 50, etc.).

Evaluate clustering accuracy, cosine similarity, and RMSE metrics on-the-fly.

```python
from libs_transfer.training.discriminator_new_data import train_acvae_pipeline

# Start the automated training and evaluation pipeline
acvae_model = train_acvae_pipeline(
    data_path='./processed_data/spectra.h5', 
    data_dir='./processed_data/', 
    epochs=5, 
    batch_size=64,
    test_split=0.5
)
```
> [!NOTE]
> If you need to drop a specific condition from training, you can modify the `prepare_training_data` call inside the pipeline script to pass `exclude_id=X`.

### Using the Model
To use your trained model to transfer spectra between conditions, please refer to `examples/transfer_spectra.py` for reference on formatting inputs for the generator.
