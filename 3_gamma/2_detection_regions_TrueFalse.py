from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt, rc_context

import lime
from lime.recognition import detection_function, cosmic_ray_function, broad_component_function
from lime.plots import theme

# Output plot
plot_address = '/home/vital/Dropbox/detection_regions.png'

# Configuration file
cfg_file = r'training_sample_v3_old.toml'
cfg = lime.load_cfg(cfg_file)

# Parameters
max_detection_limit = 10000
continuum_limit = 3
box_sigma = 2
sigma_box = 2

# Compute the plot values
x_detection = np.linspace(0, 10, 100)
y_detection = detection_function(x_detection)

x_pixel_lines = np.linspace(0.2, 0.6, 100)
y_pixel_lines = cosmic_ray_function(x_pixel_lines)
idcs_crop = y_pixel_lines > 5

fig_cfg = theme.fig_defaults({'axes.labelsize': 8,
                              'legend.fontsize': 6,
                              'figure.figsize': (3, 3),
                              'hatch.linewidth': 0.3,
                              "legend.fontsize" : 8})

cr_limit = 1000
detection_limit_pixel = 0.5439164154163089
x_cr_low = np.linspace(0.01, detection_limit_pixel, 100)
y_cr_low = cosmic_ray_function(x_cr_low)
idcs_high = y_cr_low > cr_limit
y_cr_low[idcs_high] = cr_limit

x_cr_high = x_cr_low[idcs_high]
y_cr_high = cosmic_ray_function(x_cr_high)

# Broad range
max_broad_int = max_detection_limit/2
min_broad_int = 2 * detection_function(box_sigma)

x_broad = np.linspace(cosmic_ray_function(max_broad_int, 'intensity_ratio'), box_sigma, 100)
y_broad = cosmic_ray_function(x_broad)
idx_single = y_broad > 2 * detection_function(0.3)
y_broad[~idx_single] = 2 * detection_function(x_broad[~idx_single])

# Narrow range
x_narrow = np.linspace(0.1, sigma_box / broad_component_function(2), 100)
y_narrow = np.full(x_narrow.size, 2 * y_broad.min())

min_int_ratio, max_int_ratio = 2, 10000
int_ratio_narrow = np.linspace(y_narrow.mean(), 10000, 100)
sigma_narrow_max = sigma_box / broad_component_function(int_ratio_narrow)

# Plot
with rc_context(fig_cfg):

    fig, ax = plt.subplots()

    # Detection line
    ax.plot(x_detection, y_detection, color='black', label='Detection boundary')


    # Positive and negative detection
    ax.fill_between(x_detection, y_detection, 10000, alpha=0.5, color='forestgreen', label='line')
    ax.fill_between(x_detection, 0, y_detection, alpha=0.5, color='red', label='white-noise')
    # ax.fill_between(x_detection, continuum_limit, y_detection, alpha=0.5, color='salmon', label='continuum')

    # # Single line
    # ax.plot(x_pixel_lines[idcs_crop], y_pixel_lines[idcs_crop], linestyle='--', color='black', label='Single pixel boundary')

    # # Cosmic and single-pixel lines
    # ax.fill_between(x_cr_low, detection_function(x_cr_low), y_cr_low, color='orange', label='Pixel-line')
    # ax.fill_between(x_cr_high, cr_limit, y_cr_high, color='yellow', label='Cosmic-ray')
    #
    # # Broad line component
    # ax.fill_between(x_broad, y_broad, max_detection_limit/2,  label='broad', color='none', edgecolor='#A330C9', hatch='..',
    #                 linewidth=0.3, zorder=2)
    #
    # # Narrow
    # ax.fill_betweenx(int_ratio_narrow, 0, sigma_narrow_max, hatch="////", color='none', edgecolor="#A330C9", label='narrow',
    #                  linewidth=0.3, zorder=2)
    #
    # # Doublet
    # x_min, x_max = 1.2, 1.6  # x-axis boundaries
    # y_min, y_max = 20, 10000  # y-axis boundaries
    # x_values = np.array([x_min, x_max])
    # y1, y2 = np.full_like(x_values, y_min), np.full_like(x_values, y_max)
    # ax.fill_between(x_values, y1, y2, color='#3FC7EB', alpha=0.5, label='Doublet', edgecolor='none',  zorder=2)

    # Wording
    ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}} = \sigma_{line, pixels}$ (Gaussian sigma in pixels)',
               'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$ (Signal-to-noise)'})
    ax.legend(loc='lower center', ncol=3, framealpha=0.95)

    # Axis format
    ax.set_yscale('log')
    ax.set_xlim(0, 10)
    ax.set_ylim(0.01, 10000)

    # Upper axis
    ax2 = ax.twiny()
    ticks_values = ax.get_xticks()
    ticks_labels = [f'{tick:.0f}' for tick in ticks_values*6]
    ax2.set_xticks(ticks_values)  # Set the tick positions
    ax2.set_xticklabels(ticks_labels)
    ax2.set_xlabel(r'$b_{pixels}$ (detection box width in pixels)', fontsize=6)

    # Grid
    ax.grid(axis='x', color='0.95', zorder=1)
    ax.grid(axis='y', color='0.95', zorder=1)

    plt.tight_layout()
    plt.show()
    # plt.savefig(plot_address)