import numpy as np
import pandas as pd
import lime
from lime.model import gaussian_model

from tools import detection_function

# Load the configuration parameters
cfg = lime.load_cfg('config_file.ini')
BOX_SIZE = int(cfg['ml_grid_design']['box_size_pixels'])
SAMPLE_SIZE = int(cfg['ml_grid_design']['sample_size'])
TRUE_LINE_FRACTION = cfg['ml_grid_design']['sample_line_percentage']
output_folder = cfg['data_location']['output_folder']
training_params_file = cfg['ml_grid_design']['training_params_file']
rnd = np.random.RandomState()
version = 'v2_cost1_logNorm'

# Load the database
db = pd.read_csv(f'{output_folder}/{training_params_file}', header=0)

# Unpack the parameters
amp = db['amp'].values
mu_index = db['mu_index'].values
sigma = db['sigma'].values
noise = db['noise'].values
lambda_step = db['lambda_step'].values
n_cont1 = db['n_cont10'].values
n_cont10 = db['n_cont10'].values
amp_noise_ratio = db['amp_noise_ratio'].values
sigma_lambda_ratio = db['sigma_lambda_ratio'].values

# Estate line occurrence
x_lim, y_lim = 0.2, detection_function(sigma_lambda_ratio)
random_bad = rnd.randint(0, 100, size=SAMPLE_SIZE) < TRUE_LINE_FRACTION
idcs_detected = (sigma_lambda_ratio > x_lim) & \
                ((amp_noise_ratio > y_lim) | ((sigma_lambda_ratio < 0.4) & (amp_noise_ratio > 10))) & \
                random_bad

# Check the fraction of good/bad lines
print(f'- Initial good/total {np.sum(idcs_detected)}/{idcs_detected.size}; '
      f'bad/total {np.sum(~idcs_detected)}/{idcs_detected.size}')

# Generate the line sample
rnd = np.random.RandomState()
x_line_container = np.empty([SAMPLE_SIZE, BOX_SIZE])
y_line_container = np.empty([SAMPLE_SIZE, BOX_SIZE])
for i, idx in enumerate(idcs_detected):

    # Generate continuum level with noise
    x_array = np.linspace(0, lambda_step[i] * (BOX_SIZE-1), BOX_SIZE)
    y_cont = 0 + rnd.normal(0, noise[i], BOX_SIZE)

    # Peak location
    mu = x_array[5]

    # Add line if required
    if idx:
        y_array = gaussian_model(x_array, amp[i], mu, sigma[i]) + y_cont
    else:
        y_array = y_cont

    # Normalization
    y_norm = np.log10(y_array + 10)

    # Store the line
    x_line_container[i, :] = x_array
    y_line_container[i, :] = y_norm

# Rearrange the lines training sample into a SAMPLE_SIZE x BOX_SIZE array
column_names = np.full(BOX_SIZE, 'Pixel')
column_names = np.char.add(column_names, np.arange(BOX_SIZE).astype(str))
wave_db = pd.DataFrame(data=x_line_container, columns=column_names)
flux_db = pd.DataFrame(data=y_line_container, columns=column_names)

# Save the wavelength, flux and detection (bool) arrays in three different files
wave_db.to_csv(f'{output_folder}/sample_wave_training_{version}.csv', index=False)
flux_db.to_csv(f'{output_folder}/sample_flux_training_{version}.csv', index=False)
np.savetxt(f'{output_folder}/sample_detection_training_{version}.csv', np.c_[idcs_detected], fmt="%i")
