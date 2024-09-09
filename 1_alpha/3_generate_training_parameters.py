import numpy as np
import pandas as pd
import lime

from matplotlib import pyplot as plt, rc_context
from tools import param_distribution, detection_function
from plots import PLOT_CONF, plot_ratio_distribution

# Load the configuration parameters
cfg = lime.load_cfg('config_file.ini')
SAMPLE_SIZE = int(cfg['ml_grid_design']['sample_size'])
BOX_SIZE = int(cfg['ml_grid_design']['box_size_pixels'])
output_folder = cfg['data_location']['output_folder']
training_params_file = cfg['ml_grid_design']['training_params_file']

# Detection curve
x_range = np.linspace(0.2, 50, 100)
function = detection_function(x_range)

# Generate the parameter arrays from the user defined distributions
dist_arrays = {}
param_list = ['noise', 'lambda_step', 'amp_noise_ratio', 'sigma_lambda_ratio', 'mu_index', 'n_cont10', 'n_cont1']
for param in param_list:
    # In case the parameters are constrained by the spectral window
    for limit in ['low_limit', 'high_limit']:
        if limit in cfg[f'{param}_distribution']:
            if cfg[f'{param}_distribution'][limit] == 'box_size':
                cfg[f'{param}_distribution'][limit] = BOX_SIZE

    dist_arrays[param] = param_distribution(SAMPLE_SIZE, display=True, **cfg[f'{param}_distribution'])

# Generate the amplitude and gas sigma arrys from the noise and lambda step arrays
dist_arrays['amp'] = dist_arrays['amp_noise_ratio'] * dist_arrays['noise']
dist_arrays['sigma'] = dist_arrays['sigma_lambda_ratio'] * dist_arrays['lambda_step']

# Plot the output distributions
show_plots = True
if show_plots:
    for param, array in dist_arrays.items():
        print(f'Distribution for {param}')
        plot_ratio_distribution(array, param, cfg[f'{param}_distribution']['units'])

# Plot the resulting Amp/noise vs Sigma/deltalambda sample
with rc_context(PLOT_CONF):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(dist_arrays['sigma_lambda_ratio'], dist_arrays['amp_noise_ratio'], alpha=0.1)
    ax.plot(x_range, function, color='black')
    ax.legend()
    ax.update({'title': f'Visual line detection sample',
               'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda}$',
               'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$'})
    ax.set_yscale('log')
    plt.show()

# Set detetion label (all true initially)
dist_arrays['line_check'] = True

# Save as a text file
df = pd.DataFrame(dist_arrays)
output_db = f'{output_folder}/{training_params_file}'
df.to_csv(output_db, index=False)
print(f'Database saved to {output_db}')
