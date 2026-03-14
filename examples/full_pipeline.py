from libs_transfer.training import train_concentration_predictors
from libs_transfer.prepare_data import data_to_h5
from libs_transfer.training import train_acvae_pipeline
import warnings
import torch

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.manual_seed(123456)

data_to_h5('./examples/example_raw_data', './examples/processed_data')

train_concentration_predictors(data_folder='./examples/processed_data')

acvae = train_acvae_pipeline(test_split=0.5)
