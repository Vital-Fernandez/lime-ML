from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt, rc_context

import lime
from lime.recognition import detection_function

lime.theme.set_style('dark')
fig_cfg = lime.theme.fig_defaults()
fig_cfg['figure.figsize'] = (10, 10)
fig_cfg['axes.labelsize'] = 40
fig_cfg['xtick.labelsize'] = 25
fig_cfg['ytick.labelsize'] = 25
fig_cfg['legend.fontsize'] = 20

# Load cfg
cfg_file = '../3_gamma/training_sample_v3.toml'
cfg = lime.load_cfg(cfg_file)
output_folder = Path(cfg['data_location']['output_folder'])
version = cfg['data_grid']['version']
sample_database = output_folder/f'sample_database_{version}.txt'
int_sample_size = cfg['data_grid']['int_sample_size']
int_sample_limts = cfg['data_grid']['int_sample_limits']

res_sample_size = cfg['data_grid']['res_sample_size']
res_sample_limts = cfg['data_grid']['res_sample_limits']
res_sample_limts = [0.10, 15]

int_ratio_range = np.logspace(int_sample_limts[0], int_sample_limts[1], int_sample_size, base=10000)
res_ratio_range = np.linspace(res_sample_limts[0], res_sample_limts[1], res_sample_size)
xv, yv = np.meshgrid(res_ratio_range, int_ratio_range)


idcs_detect = yv >= detection_function(xv)

x_detection = np.linspace(res_sample_limts[0], res_sample_limts[1], 100)
y_detection = detection_function(x_detection)

output_plot ='/home/vital/Dropbox/Astrophysics/Seminars/Univap_2024/detection_map_dark.png'

with rc_context(fig_cfg):

    fig, ax = plt.subplots()

    ax.axvline(0.3, label='Cosmic ray boundary', linestyle='--', color='purple')

    ax.plot(x_detection, y_detection, color='black', label='Detection boundary')

    # ax.scatter(xv, yv)
    ax.scatter(xv[idcs_detect], yv[idcs_detect], color='palegreen', label='Positive detection')
    ax.scatter(xv[~idcs_detect], yv[~idcs_detect], color='xkcd:salmon', label='Negative detection')

    # Desi range
    ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}}$',
               'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$'})

    ax.set_yscale('log')
    ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))

    plt.tight_layout()
    plt.savefig(output_plot)
