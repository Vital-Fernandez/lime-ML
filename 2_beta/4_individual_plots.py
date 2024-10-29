from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rc_context

import lime
from lime.plots import STANDARD_PLOT
from tools import normalization_1d

STANDARD_PLOT.update({'axes.labelsize': 15, 'legend.fontsize': 10, 'figure.figsize': (8, 4)})
label_dict = {'sigma_lambda_ratio': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}}$',
              'amp_noise_ratio': r'$\frac{A_{gas}}{\sigma_{noise}}$'}

# Load cfg
cfg_file = '../3_gamma/training_sample_v3_old.toml'
cfg = lime.load_cfg(cfg_file)
output_folder = Path(cfg['data_location']['output_folder'])
version = cfg['data_grid']['version']
sample_database = output_folder/f'sample_database_{version}.txt'

# Load database
df_values = lime.load_log(sample_database)

# Load the spectra
wave_db = pd.read_csv(f'{output_folder}/sample_wave_{version}.csv')
flux_db = pd.read_csv(f'{output_folder}/sample_flux_{version}.csv')
detect_db = pd.read_csv(f'{output_folder}/sample_detection_{version}.csv', names=['detect'])

# Define the targets
idcs = (df_values.success_fitting)
idcs = idcs & (df_values.amp_noise_ratio >= 5) & (df_values.amp_noise_ratio <= 10)
idcs = idcs & (df_values.sigma_lambda_ratio <= 1.625)

# Loop through the solutions
df_crop = df_values.loc[idcs]
for index in df_crop.index:
    with rc_context(STANDARD_PLOT):

        # Get the data
        x_array, y_array = wave_db.loc[index], flux_db.loc[index]
        int_ratio, width_ratio = df_crop.loc[index, 'amp_noise_ratio'], df_crop.loc[index, 'sigma_lambda_ratio']
        n_pixels = 8 * width_ratio

        # Log normalization
        y_array = normalization_1d(y_array, 10000)

        # Create the figure
        fig, ax = plt.subplots()

        # Line plot and legend
        ax.step(x_array, y_array, where='mid')

        # Line peak
        mu_index = df_crop.loc[index, 'mu_index']
        ax.scatter(x_array[mu_index], y_array[mu_index], color='red', label='Peak')

        # Line width
        line_half_width = int(n_pixels/2)
        label = r'{:0.0f} $n_{{pixels}}$ '.format(n_pixels)
        idx_min = mu_index-line_half_width if mu_index-line_half_width > 0 else 0
        idx_max = mu_index+line_half_width if mu_index+line_half_width < x_array.size else x_array.size - 1
        x_lims = [x_array[idx_min], x_array[idx_max]]
        y_lims = [y_array[idx_min], y_array[idx_max]]
        ax.scatter(x_lims, y_lims, label=f'Limits ({label})', color='black')

        # Background according to detection
        detect = detect_db.loc[index, 'detect']
        color = 'palegreen' if detect else 'xkcd:salmon'
        fig.set_facecolor(color)
        ax.set_facecolor(color)

        # Figure wording
        box_size_label = r'{:0.0f} $n_{{pixels}}$ '.format(x_array.size)
        title = (f'{label_dict["amp_noise_ratio"]} = {int_ratio}; {label_dict["sigma_lambda_ratio"]} = {width_ratio:0.2f};'
                 f' box size = {box_size_label}')
        ax.update({'xlabel': r'Wavelength', 'ylabel': r'Flux', 'title': title})
        ax.legend()
        plt.show()


