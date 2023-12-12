import numpy as np
import lime
import pandas as pd
from lime.model import gaussian_model
from lime.recognition import detection_function
from lime.plots import STANDARD_PLOT
from matplotlib import pyplot as plt, rc_context
from pathlib import Path
import scipy

cfg_file = 'config_file.toml'
cfg = lime.load_cfg(cfg_file)
output_folder = Path(cfg['data_location']['output_folder'])

amp_array = np.array(cfg['ml_grid_design']['amp_array'])
sigma_gas_array = np.array(cfg['ml_grid_design']['sigma_gas_um_array']) * 1000
delta_lam_array = np.array(cfg['ml_grid_design']['delta_lambda_um_array']) * 1000
noise_sig_array = np.array(cfg['ml_grid_design']['noise_array'])

line = 'H1_4861A'
mu_line = 4861.0
data_points = 400


df_values = lime.load_log(output_folder/'accuracy_table_v0.txt')
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

chi_intg = (df_values.intg.to_numpy() - df_values.flux_true.to_numpy()) / df_values.intg_err.to_numpy()
chi_gauss = (df_values.gauss.to_numpy() - df_values.flux_true.to_numpy()) / df_values.gauss_err.to_numpy()

intg_percentage = np.abs(df_values.intg.to_numpy()/df_values.flux_true.to_numpy() - 1)
gauss_percentage = np.abs(df_values.gauss.to_numpy()/df_values.flux_true.to_numpy() - 1)
factor = 1

x_plot, y_plot, z_plot = x_ratios[idcs_detection], y_ratios[idcs_detection], intg_percentage[idcs_detection]

# xi, yi = np.linspace(x_plot.min(), x_plot.max(), 50000), np.linspace(y_plot.min(), y_plot.max(), 5000)
xi, yi = np.linspace(x_plot.min(), x_plot.max(), 500), np.linspace(y_plot.min(), y_plot.max(), 5000)
xi, yi = np.meshgrid(xi, yi)

# Interpolate; there's also method='cubic' for 2-D data such as here
zi = scipy.interpolate.griddata((x_plot, y_plot), z_plot * factor, (xi, yi), method='cubic')



fig, ax = plt.subplots()
im = ax.imshow(zi, vmin = 0 * factor, vmax = 0.3 * factor, origin='lower', cmap='Greys',
               extent=[x_plot.min(), x_plot.max(), y_plot.min(), y_plot.max()], aspect='auto')
# Gaussian measurements
ax.plot(x_detection, y_detection, color='black', label='Detection boundary')
ax.set_yscale('log')
plt.colorbar(im, ax=ax)
plt.show()

# STANDARD_PLOT.update({'axes.labelsize': 30, 'legend.fontsize': 20, 'figure.figsize': (8, 8)})
#
# with rc_context(STANDARD_PLOT):
#
#     fig, ax = plt.subplots()
#
#     # Cosmic ray limits
#     ax.axvline(0.3, label='Cosmic ray boundary', linestyle='--', color='black')
#
#     # Gaussian measurements
#     ax.plot(x_detection, y_detection, color='black', label='Detection boundary')
#
#     # Detections
#     # ax.scatter(x_ratios[idcs_detection], y_ratios[idcs_detection], label='Measurements')
#     # ax.scatter(x_ratios[~idcs_detect_dood], y_ratios[~idcs_detect_dood], label='False detection')
#
#
#     # ratio_scatter = ax.scatter(x_ratios[idcs_detection], y_ratios[idcs_detection], c=np.abs(chi_gauss[idcs_detection]),
#     #                            cmap='viridis', edgecolor=None, vmin=0, vmax=6)
#     # cbar = plt.colorbar(ratio_scatter, ax=ax)
#
#     ratio_scatter = ax.scatter(x_ratios[idcs_detection], y_ratios[idcs_detection], c=intg_percentage[idcs_detection],
#                                cmap='viridis', edgecolor=None, vmin=0, vmax=0.2)
#     cbar = plt.colorbar(ratio_scatter, ax=ax)
#
#     # Failure points
#     ax.scatter(x_ratios[idcs_failure], y_ratios[idcs_failure], marker='x', facecolor='black',
#                label='Gaussian fitting measurement')
#
#     ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}}$',
#                'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$'})
#
#     ax.set_yscale('log')
#
#     ax.legend()
#
#     plt.tight_layout()
#     plt.show()


