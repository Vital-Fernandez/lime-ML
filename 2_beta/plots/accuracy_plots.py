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
gauss_relative_error = np.abs(df_values.gauss.to_numpy() - df_values.flux_true.to_numpy()) / df_values.flux_true.to_numpy()
intg_relative_error = np.abs(df_values.intg.to_numpy() - df_values.flux_true.to_numpy()) / df_values.flux_true.to_numpy()

label_name = {'gauss': r'$\frac{F_{Gaussian}}{F_{true}} - 1$',
              'intg' : r'$\frac{F_{integration}}{F_{true}} - 1$'}

param_dict = {'gauss': gauss_relative_error,
              'intg' : intg_relative_error}

# Plot data
STANDARD_PLOT.update({'axes.labelsize': 30, 'legend.fontsize': 15, 'figure.figsize': (10, 8)})

original_cmap = plt.cm.cubehelix
inverted_cmap = original_cmap.reversed()


for param_type, param_array in param_dict.items():

    with rc_context(STANDARD_PLOT):

        fig, ax = plt.subplots()

        # Cosmic ray limits
        ax.axvline(0.3, label='Cosmic ray boundary', linestyle='--', color='black')

        # Gaussian measurements
        ax.plot(x_detection, y_detection, color='black', label='Detection boundary')

        # Measurement points
        ratio_scatter = ax.scatter(x_ratios[idcs_detection], y_ratios[idcs_detection], c=param_array[idcs_detection]*factor,
                                   cmap=inverted_cmap, edgecolor=None, vmin=0*factor, vmax=0.3*factor)
        cbar = plt.colorbar(ratio_scatter, ax=ax)
        cbar.set_label(label_name[param_type], rotation=270, labelpad=70)

        # Failure points
        ax.scatter(x_ratios[idcs_failure], y_ratios[idcs_failure], marker='x', facecolor='red',
                   label='Gaussian fit failure')

        ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}}$',
                   'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$'})

        ax.set_yscale('log')

        ax.legend(loc=4, framealpha=1)

        plt.tight_layout()

        plt.show()
        # plt.savefig(figures_folder/f'{param_type}_relative_error.png', dpi=400)


