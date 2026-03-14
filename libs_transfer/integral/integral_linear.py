import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

concentrations = pd.read_csv('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/integrals/concentrations.csv')
conc_bool = concentrations.iloc[:,0] != 0
concentrations = concentrations[conc_bool]
integrals = pd.read_csv('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/integrals/integrals.csv')
integrals = integrals[conc_bool]
stds = pd.read_csv('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/integrals/stds.csv')
stds = stds[conc_bool]

element_to_oxide_dict = {
"Al_wavelen" : "AL2O3",
"Si_wavelen" : "SIO2",
"Ca_wavelen" : "CAO",
"Fe_wavelen" : "FE2O3",
"K_wavelen" : "K2O",
"Mg_wavelen" : "MGO",
"Mn_wavelen" : "MNO",
"Ti_wavelen" : "TIO2",
}

conditions = range(3)
# conditions = [2]
samples = range(42)
# samples = [0]

cond_dict = {
    0:'Vacuum 100%',
    1:'Earth 50%',
    2:'Earth 100%',
}

for c in conditions:
    conc_ = concentrations[concentrations['condition'] == c]
    integ_ = integrals[integrals['condition'] == c]
    std_ = stds[stds['condition'] == c]

    # for s in samples:
    #     conc_ = conc[conc['sample'] == s]
    #     integ_ = integ[integ['sample'] == s]
    #     std_ = std[std['sample'] == s]

        # conc_.drop(columns=['condition', 'sample'], inplace=True)
        # integ_.drop(columns=['condition', 'sample'], inplace=True)
        # std_.drop(columns=['condition', 'sample'], inplace=True)

    for e in element_to_oxide_dict.keys():
        _conc_ = conc_[element_to_oxide_dict[e]].to_numpy().reshape(-1)
        _integ_ = integ_[e].to_numpy().reshape(-1)
        _std_ = std_[e].to_numpy().reshape(-1)

        slope, intercept = np.polyfit(_conc_, _integ_, 1)
        regression_line = slope * _conc_ + intercept

        SS_res = np.sum((_integ_ - regression_line) ** 2)
        SS_tot = np.sum((_integ_ - np.mean(_integ_)) ** 2)
        r_squared = 1 - (SS_res / SS_tot)
        
        plt.figure(figsize=(16, 9))
        plt.errorbar(_conc_, _integ_, yerr=_std_, fmt='o', capsize=5, 
                    color='royalblue', ecolor='lightgray', 
                    elinewidth=2, markeredgewidth=2, 
                    label='Data with std deviation')
        
        plt.plot(_conc_, regression_line, color='red', linewidth=1, label='Regression Line')
        plt.annotate(f'R² = {r_squared:.2f}', xy=(0.05, 0.9), xycoords='axes fraction', 
             fontsize=14, color='black')
        
        regression_eq = f'y = {slope:.2f}x + {intercept:.2f}'

        plt.annotate(regression_eq, xy=(0.05, 0.85), xycoords='axes fraction', 
                    fontsize=14, color='black')

        plt.title(f'{cond_dict[c]}: {element_to_oxide_dict[e]}')
        plt.xlabel('concentration [%]', fontsize=14)
        plt.ylabel('signal integral', fontsize=14)
        plt.grid()

        plt.savefig(f"/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/integrals/plots/{c}-{e}.png")
        plt.close()