from libs_transfer.training.discriminator_new_data import train_acvae_pipeline
import warnings
import torch

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.manual_seed(123456)

acvae = train_acvae_pipeline()
