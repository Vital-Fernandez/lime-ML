from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt, rc_context

import lime
from tools import STANDARD_PLOT
from lime.recognition import detection_function

# Configuration file
cfg_file = '../3_gamma/training_sample_v3.toml'
cfg = lime.load_cfg(cfg_file)
sample_params = cfg['sample_data_v3']

# Data location
version = sample_params['version']
output_folder = Path(sample_params['output_folder'])
sample_database = output_folder/f'sample_database_{version}.txt'

# Load database
df_values = lime.load_frame(sample_database)

# Compute the plot values
x_detection = np.linspace(0.2, 20, 100)
y_detection = detection_function(x_detection)

x_ratios = df_values.sigma.to_numpy()/df_values.delta.to_numpy()
y_ratios = df_values.amp.to_numpy()/df_values.noise.to_numpy()

# Crop the sample to make the plot viable:
num_samples = 500
random_indices = np.random.choice(x_ratios.size, size=num_samples, replace=False)
x_ratios, y_ratios = x_ratios[random_indices], y_ratios[random_indices]

idcs_detect = y_ratios >= detection_function(x_ratios)
idcs_fail = None if np.all(np.isnan(df_values.success_fitting.to_numpy())) else ~df_values.success_fitting.to_numpy()

# Plot
STANDARD_PLOT.update({'axes.labelsize': 30, 'legend.fontsize': 20, 'figure.figsize': (8, 8)})
with rc_context(STANDARD_PLOT):

    fig, ax = plt.subplots()

    ax.axvline(0.3, label='Cosmic ray boundary', linestyle='--', color='purple')

    ax.plot(x_detection, y_detection, color='black', label='Detection boundary')

    ax.scatter(x_ratios[idcs_detect], y_ratios[idcs_detect], color='palegreen', label='Positive detection')
    ax.scatter(x_ratios[~idcs_detect], y_ratios[~idcs_detect], color='xkcd:salmon', label='Negative detection')

    if idcs_fail is not None:
        ax.scatter(x_ratios[idcs_fail], y_ratios[idcs_fail], marker='x', facecolor='red', label='Fit failure')

    # Desi range
    ax.axvspan(0.10, 3.60, alpha=0.2, color='tab:blue', label='DESI range')

    ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}}$',
               'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$'})

    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()


# Pixels width plot
STANDARD_PLOT.update({'axes.labelsize': 30, 'legend.fontsize': 20, 'figure.figsize': (8, 8)})
n_sigma = 8
x_ratios = n_sigma * x_ratios

with rc_context(STANDARD_PLOT):

    fig, ax = plt.subplots()

    ax.axvline(0.3 * n_sigma, label='Cosmic ray boundary', linestyle='--', color='purple')

    ax.plot(x_detection * n_sigma, y_detection, color='black', label='Detection boundary')

    ax.scatter(x_ratios[idcs_detect], y_ratios[idcs_detect], color='palegreen', label='Positive detection')
    ax.scatter(x_ratios[~idcs_detect], y_ratios[~idcs_detect], color='xkcd:salmon', label='Negative detection')

    if idcs_fail is not None:
        ax.scatter(x_ratios[idcs_fail], y_ratios[idcs_fail], marker='x', facecolor='red', label='Fit failure')

    # Desi range
    ax.axvspan(0.10 * n_sigma, 3.60 * n_sigma, alpha=0.2, color='tab:blue', label='DESI range')

    ax.update({'xlabel': r'Line width (pixels)',
               'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$'})

    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()
