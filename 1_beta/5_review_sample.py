import numpy as np
import pandas as pd
import lime
from matplotlib import pyplot as plt, rc_context

from tools import detection_function
from plots import PLOT_CONF, plot_ratio_distribution, plot_line_window


# Load the configuration parameters
cfg = lime.load_cfg('../2_release/config_file.toml')
BOX_SIZE = int(cfg['ml_grid_design']['box_size_pixels'])
SAMPLE_SIZE = int(cfg['ml_grid_design']['sample_size'])
TRUE_LINE_FRACTION = cfg['ml_grid_design']['sample_line_percentage']
output_folder = cfg['data_location']['output_folder']
training_params_file = cfg['ml_grid_design']['training_params_file']
version = 'v2_cost1_logNorm'

# Load the database
params_db = pd.read_csv(f'{output_folder}/{training_params_file}', header=0)
wave_db = pd.read_csv(f'{output_folder}/sample_wave_training_{version}.csv', header=0)
flux_db = pd.read_csv(f'{output_folder}/sample_flux_training_{version}.csv', header=0)
line_check = np.loadtxt(f'{output_folder}/sample_detection_training_{version}.csv', dtype=bool)

# Unpack the parameters
amp = params_db['amp'].values
mu_index = params_db['mu_index'].values
sigma = params_db['sigma'].values
noise = params_db['noise'].values
lambda_step = params_db['lambda_step'].values
n_cont1 = params_db['n_cont10'].values
n_cont10 = params_db['n_cont10'].values
amp_noise_ratio = params_db['amp_noise_ratio'].values
sigma_lambda_ratio = params_db['sigma_lambda_ratio'].values

# # Check the distributions
# for param in params_db.columns:
#     if param != 'line_check':
#         plot_ratio_distribution(params_db[param].values, param, units=cfg[f'{param}_distribution']['units'])

# Check the line visibility occurrence
with rc_context(PLOT_CONF):

    # Detection curve
    x_range = np.linspace(0.2, 50, 1000)
    function = detection_function(x_range)

    # Figure
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(sigma_lambda_ratio[line_check], amp_noise_ratio[line_check], color='palegreen', alpha=0.1, label='True')
    ax.scatter(sigma_lambda_ratio[~line_check], amp_noise_ratio[~line_check], color='xkcd:salmon', alpha=0.1, label='False')
    ax.scatter(0.4, 10, label='Cosmic ray boundary', marker='x', color='purple')
    ax.plot(x_range, function, color='black')
    ax.legend()
    ax.update({'title': f'Visual line detection sample',
               'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda}$',
               'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$'})
    ax.set_yscale('log')
    plt.show()

# Check the individual lines
for i, idx in enumerate(line_check):

    # Generate continuum level with noise
    x_array = wave_db.loc[i].values
    y_array = flux_db.loc[i].values

    plot_line_window(x_array, y_array, amp_noise_ratio[i], sigma_lambda_ratio[i], idx)

    # if idx == True and amp_noise_ratio[i] < 10:
    #     plot_line_window(x_array, y_array, amp_noise_ratio[i], sigma_lambda_ratio[i], idx)