from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt, rc_context

import lime
from lime.recognition import detection_function, cosmic_ray_function
from lime.plots import theme
from matplotlib.ticker import MultipleLocator, AutoMinorLocator





# Configuration file
cfg_file = r'/home/vital/PycharmProjects/lime-ML/1_beta/config_file.toml'
cfg = lime.load_cfg(cfg_file)

# Compute the plot values
x_detection = np.linspace(0, 10, 100)
y_detection = detection_function(x_detection)

x_pixel_lines = np.linspace(0.2, 0.6, 100)
y_pixel_lines = cosmic_ray_function(x_pixel_lines)
idcs_crop = y_pixel_lines > 5

fig_cfg = theme.fig_defaults({'axes.labelsize': 8,
                              'legend.fontsize': 6,
                              'figure.figsize': (3, 3)})


cr_limit = 1000
x_cr_low = np.linspace(0.01, 0.5555, 100)
y_cr_low = cosmic_ray_function(x_cr_low)
idcs_high = y_cr_low > cr_limit
y_cr_low[idcs_high] = cr_limit

x_cr_high = x_cr_low[idcs_high]
y_cr_high = cosmic_ray_function(x_cr_high)

# x_cr_high =

# Plot
with rc_context(fig_cfg):

    fig, ax = plt.subplots()

    # Detection line
    ax.plot(x_detection, y_detection, color='black', label='Detection boundary')

    # Single line
    ax.plot(x_pixel_lines[idcs_crop], y_pixel_lines[idcs_crop], linestyle='--', color='purple', label='Single pixel '
                                                                                                      'line boundary')

    # Positive negative
    ax.fill_between(x_detection, 1, y_detection, alpha=0.5, color='salmon', label='continuum')
    ax.fill_between(x_detection, 0, 1, alpha=0.5, color='red', label='white-noise')
    ax.fill_between(x_detection, y_detection, 10000, alpha=0.5, color='forestgreen', label='line')

    # ax.fill_betweenx(y_cr_range, 0, 0.5, color='orange')
    ax.fill_between(x_cr_low, detection_function(x_cr_low), y_cr_low, color='orange', label='Pixel-line')
    ax.fill_between(x_cr_high, cr_limit, y_cr_high, color='yellow', label='Cosmic-ray')

    ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}} = \sigma_{line, pixels}$ (Gaussian sigma in pixels)',
               'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$ (Signal-to-noise)'})

    # Legend
    ax.legend(loc=1)

    # Axis format
    ax.set_yscale('log')
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0.1, 10000)
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))

    # Upper axis
    ax2 = ax.twiny()
    ticks_values = ax.get_xticks()
    ticks_labels = [f'{tick:.0f}' for tick in ticks_values*6]
    ax2.set_xticklabels(ticks_labels)
    ax2.minorticks_on()
    ax2.set_xlabel(r'$b_{pixels}$ (detection box width in pixels)', fontsize=6)

    # Grid
    ax.grid(axis='x', color='0.95')
    ax.grid(axis='y', color='0.95')

    plt.tight_layout()
    plt.show()
