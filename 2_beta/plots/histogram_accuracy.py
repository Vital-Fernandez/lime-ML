import numpy as np
import lime
import pandas as pd
from lime.model import gaussian_model
from lime.recognition import detection_function
from lime.plots import STANDARD_PLOT
from matplotlib import pyplot as plt, rc_context
from pathlib import Path

cfg_file = '../../3_gamma/training_sample_v3_old.toml'
cfg = lime.load_cfg(cfg_file)
output_folder = Path(cfg['data_location']['output_folder'])
figures_folder = Path('D:\Dropbox\Astrophysics\Tools\LineMesurer')

amp_array = np.array(cfg['ml_grid_design']['amp_array'])
sigma_gas_array = np.array(cfg['ml_grid_design']['sigma_gas_um_array']) * 1000
delta_lam_array = np.array(cfg['ml_grid_design']['delta_lambda_um_array']) * 1000
noise_sig_array = np.array(cfg['ml_grid_design']['noise_array'])

line = 'H1_4861A'
mu_line = 4861.0
data_points = 400

df_values = lime.load_log(output_folder/'accuracy_table_v2.txt')
x_ratios = df_values.sigma.to_numpy()/df_values.delta.to_numpy()
y_ratios = df_values.amp.to_numpy()/df_values.noise.to_numpy()

idcs_failure = df_values.gauss == 'None'
idcs_detection = (~idcs_failure) & (x_ratios > 0.3)

# Swap None strings to nan
df_values.loc[idcs_failure, 'gauss'] = np.nan
df_values.loc[idcs_failure, 'gauss_err'] = np.nan
df_values['gauss'] = df_values['gauss'].astype(float)
df_values['gauss_err'] = df_values['gauss_err'].astype(float)

# Plot
x_detection = np.linspace(0.2, 20, 100)
y_detection = detection_function(x_detection)
factor = 1

# Data color scale
intg_relative_error = (df_values.intg.to_numpy() - df_values.flux_true.to_numpy()) / df_values.flux_true.to_numpy()
gauss_relative_error = (df_values.gauss.to_numpy() - df_values.flux_true.to_numpy()) / df_values.flux_true.to_numpy()

intg_flux = df_values.intg.to_numpy()
gauss_flux = df_values.gauss.to_numpy()
true_flux = df_values.flux_true

label_name = {'gauss': r'Gaussian fluxes',
              'intg' : r'Integrated fluxes'}

param_dict = {'gauss': gauss_relative_error,
              'intg' : intg_relative_error}

param_flux = {'gauss': gauss_flux,
              'intg' : intg_flux}

param_dict = {'gauss': gauss_relative_error,
              'intg' : intg_relative_error}

# Plot data
STANDARD_PLOT.update({'axes.labelsize': 25, 'legend.fontsize': 16, 'figure.figsize': (8, 8)})

original_cmap = plt.cm.cubehelix
inverted_cmap = original_cmap.reversed()
bin_size = np.arange(-0.5, 0.5, step=0.05)

with rc_context(STANDARD_PLOT):

    fig, ax = plt.subplots()

    for param_type, param_array in param_dict.items():

        array_data = param_array[idcs_detection]
        mean_data, std_data = np.nanmedian(array_data), np.nanstd(array_data)
        low, high = np.nanpercentile(array_data, [0.01, 0.99])
        idcs_array_crop = (array_data > -0.50) & (array_data < 0.50)
        # data_crop = param_flux[param_type][idcs_detection][idcs_array_crop] - true_flux[idcs_detection][idcs_array_crop]
        data_crop = array_data[idcs_array_crop]

        print(param_type, data_crop.size)

        ax.hist(data_crop, density=True, alpha=0.5, bins=bin_size, label=label_name[param_type])


    ax.update({'xlabel': r'$\frac{F_{measured}}{F_{true}} - 1$',
               'ylabel': r'Probability density count'})

    ax.legend()

    plt.tight_layout()

    plt.show()


