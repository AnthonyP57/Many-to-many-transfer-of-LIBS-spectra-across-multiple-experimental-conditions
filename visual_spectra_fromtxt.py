import polars as pl
from data_visualization import xy_plot
import numpy as np

df = pl.read_csv('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/Data_Buday/Mars/Mars samples/Al2O3-BM.txt', separator="\t")

df = df[:,1:].transpose(include_header=False, column_names=[str(i) for i in df[:,0]])

# print(df)

xy_plot([np.linspace(200, 1000, df.shape[1])], [df[0].to_numpy().reshape(-1)])