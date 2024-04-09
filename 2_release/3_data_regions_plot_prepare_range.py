from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt, rc_context

import lime
from lime.plots import STANDARD_PLOT
from lime.recognition import detection_function

# Load cfg
cfg_file = 'config_file.toml'
cfg = lime.load_cfg(cfg_file)
output_folder = Path(cfg['data_location']['output_folder'])
version = cfg['data_grid']['version']
sample_database = output_folder/f'sample_database_{version}.txt'
int_sample_size = cfg['data_grid']['int_sample_size']
int_sample_limts = cfg['data_grid']['int_sample_limits']

res_sample_size = cfg['data_grid']['res_sample_size']
res_sample_limts = cfg['data_grid']['res_sample_limits']

int_ratio_range = np.logspace(int_sample_limts[0], int_sample_limts[1], int_sample_size, base=10000)
res_ratio_range = np.linspace(res_sample_limts[0], res_sample_limts[1], res_sample_size)
xv, yv = np.meshgrid(res_ratio_range, int_ratio_range)


idcs_detect = yv >= detection_function(xv)

# noise_fix = 1
# resolution_fix = 1
#
# for int_ratio in int_ratio_range:
#     for res_ratio in res_ratio_range:
#
#         amp = int_ratio * noise_fix
#         sigma = res_ratio * resolution_fix


# # Load database
# df_values = lime.load_log(sample_database)

x_detection = np.linspace(res_sample_limts[0], res_sample_limts[1], 100)
y_detection = detection_function(x_detection)
#
# x_ratios = df_values.sigma.to_numpy()/df_values.delta.to_numpy()
# y_ratios = df_values.amp.to_numpy()/df_values.noise.to_numpy()
# idcs_detect = y_ratios >= detection_function(x_ratios)
#
# idcs_fail = ~df_values.success_fitting.to_numpy()
#
# STANDARD_PLOT.update({'axes.labelsize': 30, 'legend.fontsize': 20, 'figure.figsize': (8, 8)})
#
with rc_context(STANDARD_PLOT):

    fig, ax = plt.subplots()

    ax.axvline(0.3, label='Cosmic ray boundary', linestyle='--', color='purple')

    ax.plot(x_detection, y_detection, color='black', label='Detection boundary')

    # ax.scatter(xv, yv)
    ax.scatter(xv[idcs_detect], yv[idcs_detect], color='palegreen', label='Positive detection')
    ax.scatter(xv[~idcs_detect], yv[~idcs_detect], color='xkcd:salmon', label='Negative detection')

    # Desi range
    ax.axvspan(0.10, 3.60, alpha=0.2, color='tab:blue', label='DESI range')

    ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}}$',
               'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$'})

    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()
