import numpy as np
import pandas as pd
import lime
from pathlib import Path
from matplotlib import pyplot as plt, rc_context
from tools import detection_function
from plots import PLOT_CONF, LineVisualizationMapper, line_type_color
from lime.plots import STANDARD_PLOT

cfg = lime.load_cfg('../2_release/config_file.toml')
output_folder = Path(cfg['data_location']['output_folder'])
output_database = output_folder/f'manual_selection_db.txt'

# Grid with parameter ranges
delta_lambda_limits = cfg['ml_grid_design']['Deltalambda_boundaries_um_array']
sigma_gas_um_array = cfg['ml_grid_design']['sigma_gas_um_array']

# # First Amp - noise by resolution spectra
# counter = 0
# delta_lambda_range_um = cfg['ml_grid_design']['delta_lambda_um_array']
# for i, lambda_step in enumerate(delta_lambda_range_um):
#
#     grid_params = {'amp': cfg['ml_grid_design']['amp_array'],
#                    'mu': 0,
#                    'sigma': 0.001,
#                    'noise': cfg['ml_grid_design']['noise_array'],
#                    'lambda_step': lambda_step}
#
#     print(f'- Grid number {counter}')
#     LineVisualizationMapper(output_folder / f'manual_selection_db.txt', 'noise', 'amp', **grid_params)
#     counter += 1

# # Second run by gas velocity dispersion
# counter = 0
# for i, sigma in enumerate(sigma_gas_um_array):
#
#     grid_params = {'amp': cfg['ml_grid_design']['amp_array'],
#                    'mu': 0,
#                    'sigma': sigma,
#                    'noise': cfg['ml_grid_design']['noise_array'],
#                    'lambda_step': 0.0014}
#
#     print(f'- Grid number {counter}: sigma_gas = {sigma}')
#     LineVisualizationMapper(output_database, 'noise', 'amp', **grid_params)
#     counter += 1

# Load the visual inspection results and compute the parameter space ratios
detection_db = pd.read_csv(output_database, delim_whitespace=True, header=0, index_col=0)
detection_db['A_sigma_ratio'] = detection_db.amp.values/detection_db.noise.values
detection_db['sigma_delta_ratio'] = detection_db.sigma.values/detection_db.lambda_step.values

# Plot the results

label_dict = {'True': 'Positive detection', 'False': 'Negative detection',
              'undecided': 'Inconclusive', 'None': 'Not checked'}

STANDARD_PLOT.update({'axes.labelsize': 30, 'legend.fontsize': 20, 'figure.figsize': (12, 12)})

with rc_context(STANDARD_PLOT):

    fig, ax = plt.subplots()
    for i, diag in enumerate(['True', 'False', 'undecided']):

        idcs = detection_db.line == diag

        x_ratio = detection_db.loc[idcs, 'sigma_delta_ratio'].values
        y_ratio = detection_db.loc[idcs, 'A_sigma_ratio'].values
        color = line_type_color[diag]

        ax.scatter(x_ratio, y_ratio, color=color, label=label_dict[diag])

    ax.axvline(0.3, label='Cosmic ray boundary', linestyle='--', color='purple')
    # _ax.scatter(0.4, 10, label='Cosmic ray boundary', marker='x', color='purple')
    x_range = np.linspace(0.2, 20, 100)
    function = detection_function(x_range)
    ax.plot(x_range, function, color='black', label='Detection boundary')

    ax.legend()
    ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}}$',
               'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$'})

    ax.set_yscale('log')
    plt.show()
