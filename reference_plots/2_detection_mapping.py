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

# Plot
with rc_context(fig_cfg):

    fig, ax = plt.subplots()

    ax.plot(x_detection, y_detection, color='black', label='Detection boundary')

    ax.fill_between(x_detection, 0, y_detection, alpha=0.5, color='salmon', label='Positive line detection')
    ax.fill_between(x_detection, y_detection, 10000, alpha=0.5, color='forestgreen', label='Negative line detection')

    ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}} = \sigma_{line, pixels}$ (Gaussian sigma in pixels)',
               'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$ (Signal-to-noise)'})

    ax.legend()

    # Axis format
    ax.set_yscale('log')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10000)
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
