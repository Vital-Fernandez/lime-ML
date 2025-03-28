from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt, rc_context

import lime
from lime.recognition import detection_function, cosmic_ray_function, broad_component_function
from lime.plots import theme
from pathlib import Path

# Output plot
output_folder = Path('/home/vital/Dropbox/Astrophysics/Seminars/2024_BootCamp')

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

x_pixel_lines = np.linspace(0, 0.6, 100)
y_pixel_lines = cosmic_ray_function(x_pixel_lines)
idcs_crop = y_pixel_lines > 5

# theme.set_style('dark')
fig_cfg = theme.fig_defaults({'figure.dpi': 150,
                              'axes.labelsize': 9,
                              'axes.titlesize':9,
                              'figure.figsize': (4, 4),
                              'hatch.linewidth': 0.3,
                              "legend.fontsize" : 8})

cr_limit = 1000
detection_limit_pixel = 0.5439164154163089
x_cr_low = np.linspace(0, detection_limit_pixel, 100)
y_cr_low = cosmic_ray_function(x_cr_low)
idcs_high = y_cr_low > cr_limit
y_cr_low[idcs_high] = cr_limit

x_cr_high = x_cr_low[idcs_high]
y_cr_high = cosmic_ray_function(x_cr_high)

# Broad range
max_cr_boundary_res = cosmic_ray_function(10000, res_ratio_check=False)
x_broad = np.linspace(max_cr_boundary_res, box_sigma, 100)
y_broad = cosmic_ray_function(x_broad)
idx_detect = y_broad < 2 * detection_function(x_broad)
y_broad[idx_detect] = 2 * detection_function(x_broad)[idx_detect]

# Narrow range
x_narrow = np.linspace(0.1, sigma_box / broad_component_function(2), 100)
y_narrow = np.full(x_narrow.size, 2 * y_broad.min())

min_int_ratio, max_int_ratio = 2, 10000
int_ratio_narrow = np.linspace(y_narrow.mean(), 10000, 100)
sigma_narrow_max = sigma_box / broad_component_function(int_ratio_narrow)

# Doublet
x_double_min, x_doublet_max = 1.2, 1.6  # x-axis boundaries
y_doublet_min, y_doublet_max = 20, 10000  # y-axis boundaries
x_doublet_values = np.array([x_double_min, x_doublet_max])

# Peak/trough
start_x, end_x = 3, 1.5
start_y, end_y = 5000, 5000

# Plot
with rc_context(fig_cfg):

    fig, ax = plt.subplots()

    # Detection boundary
    ax.plot(x_detection, y_detection, color='black', label='Detection boundary')

    # Single pixel line boundary
    ax.plot(x_pixel_lines[idcs_crop], y_pixel_lines[idcs_crop], linestyle='--', color='black', label='Single pixel boundary')

    # Positive and negative detection
    ax.fill_between(x_detection, y_detection, 10000, alpha=0.5, color='forestgreen', label='line', edgecolor='none')
    ax.fill_between(x_detection, 0, continuum_limit, alpha=0.5, color='red', label='white-noise', edgecolor='none')
    ax.fill_between(x_detection, continuum_limit, y_detection, alpha=0.5, color='salmon', label='continuum', edgecolor='none')

    # Cosmic and single-pixel lines
    ax.fill_between(x_cr_low, detection_function(x_cr_low), y_cr_low, color='orange', label='Pixel-line', edgecolor='none')
    ax.fill_between(x_cr_high, cr_limit, y_cr_high, color='yellow', label='Cosmic-ray', edgecolor='none')

    # Broad line component
    ax.fill_between(x_broad, y_broad, max_detection_limit/2,  label='broad', color='none', edgecolor='#A330C9', hatch='..',
                    linewidth=0.3, zorder=2)

    # Narrow
    ax.fill_betweenx(int_ratio_narrow, 0, sigma_narrow_max, hatch="////", color='none', edgecolor="#A330C9", label='narrow',
                     linewidth=0.3, zorder=2)

    # Doublet
    y1, y2 = np.full_like(x_doublet_values, y_doublet_min), np.full_like(x_doublet_values, y_doublet_max)
    ax.fill_between(x_doublet_values, y1, y2, color='#3FC7EB', alpha=0.5, label='Doublet', edgecolor='none', zorder=2)

    # Peak/trough
    peak_res_min, peak_res_max = 3, 4
    peak_int_min, peak_int_max = 1000, 10000
    x_peak = np.linspace(peak_res_min, peak_res_max, 100)
    ax.fill_between(x_peak, peak_int_min, peak_int_max, color='#C69B6D', alpha=0.8, label='peak-trough', edgecolor='black', zorder=2)
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(facecolor='black', edgecolor='black', width=0.1, headwidth=6, headlength=5))
    # Wording
    ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}} = \sigma_{pixels}$ (Gaussian sigma in pixels)',
               'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$ (Signal-to-noise)'})
    ax.legend(loc='lower center', ncol=2, framealpha=0.95)

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
    ax2.set_xlabel(r'$b_{pixels}$ (detection box width in pixels)')

    # Grid
    ax.grid(axis='x', color='0.95', zorder=1)
    ax.grid(axis='y', color='0.95', zorder=1)

    plt.tight_layout()
    plt.show()
    # plt.savefig(plot_address)
    # plt.savefig(output_folder/'diagnostic_plot.png')
