import numpy as np
import lime
import pandas as pd
from lime.model import gaussian_model
from lime.recognition import detection_function
from pathlib import Path
from tqdm import tqdm
from itertools import product

# Configuration file
cfg_file = '../3_gamma/training_sample_v3_old.toml'
cfg = lime.load_cfg(cfg_file)
sample_params = cfg['sample_data_v3']

# Data location
version = sample_params['version']
output_folder = Path(sample_params['output_folder'])
sample_database = output_folder/f'sample_database_{version}.txt'

# Grid parameters
amp_array = np.array(sample_params['amp_array'])
sigma_gas_array = np.array(sample_params['sigma_gas_um_array']) * 1000
delta_lam_array = np.array(sample_params['delta_lambda_um_array']) * 1000
noise_sig_array = np.array(sample_params['noise_array'])

# Box parameters
limit_O, limit_f = sample_params['box_limits']
small_box_reslimit = sample_params['small_box_reslimit']
small_box_width = sample_params['small_box_size']
large_box_width = sample_params['large_box_size']
box_size = limit_O + limit_f
number_sigmas = 8

# Sample parameters
int_sample_size = sample_params['int_sample_size']
int_sample_limts = sample_params['int_sample_limits']

res_sample_size = sample_params['res_sample_size']
res_sample_limts = sample_params['res_sample_limits']
resolution_fix = sample_params['resolution_fix']

int_ratio_range = np.logspace(int_sample_limts[0], int_sample_limts[1], int_sample_size, base=10000)
res_ratio_range = np.linspace(res_sample_limts[0], res_sample_limts[1], res_sample_size)

sample_size = int_sample_size * res_sample_size
noise_array = np.random.uniform(0.50, 10, size=sample_size)

# Line parameters
line = sample_params['line']
data_points = sample_params['data_points']
width_factor = sample_params['width_factor']

# Data containers
df_values = pd.DataFrame(index=np.arange(sample_size), columns=sample_params['headers'])

x_line_container = np.full([sample_size, box_size], np.nan)
y_line_container = np.full([sample_size, box_size], np.nan)
detect_container = np.full(sample_size, np.nan)

# Progress variables
combinations = np.array(list(product(int_ratio_range, res_ratio_range)))
bar = tqdm(combinations, desc="Item", mininterval=0.2, unit=" combinations")
# pbar = tqdm(array_product, unit=" line")

# Perform fits check
lime_fit_check = False

# Generate x range
mu_line = 0
w0 = - data_points / 2
wf = data_points / 2
wave = np.arange(w0, wf, resolution_fix)

# Loop through the conditions
for idx, (int_ratio, res_ratio) in enumerate(bar):

    noise_i = noise_array[idx]
    random_noise = np.random.normal(0, noise_i, data_points)

    amp = int_ratio * noise_i
    sigma = res_ratio * resolution_fix

    # Continuum level
    cont = 0

    # Compute spectrum
    flux = gaussian_model(wave, amp, 0, sigma) + random_noise + cont

    # Theoretical flux
    theo_flux = amp * 2.5066282746 * sigma
    true_error = noise_i * np.sqrt(2 * width_factor * resolution_fix * sigma)

    if lime_fit_check:

        # Create the LiMe spectrum
        spec = lime.Spectrum(wave, flux, redshift=0, norm_flux=1)

        # Fit the line for a wide band
        w3, w4 = mu_line - width_factor * sigma, mu_line + width_factor * sigma
        idcs_bands = np.searchsorted(wave, ([wave[10], wave[20], w3, w4, wave[-20], wave[-10]]))
        bands = wave[idcs_bands]
        spec.fit.bands(line, bands)

        # Check whether the line was fitted
        success_fit = True if spec.fit.line.observations == 'no' else False

        if success_fit:
            gauss_flux, gauss_err = spec.frame.loc[line, ['profile_flux', 'profile_flux_err']].to_numpy()
        else:
            gauss_flux, gauss_err = np.nan, np.nan

        # Store the measurements
        intg_flux, intg_err, cont, cont_err, m_cont, n_cont = spec.frame.loc[line, ['intg_flux', 'intg_flux_err',
                                                                                    'cont', 'cont_err',
                                                                                    'm_cont', 'n_cont']].to_numpy()

    else:

        intg_flux, intg_err = np.nan, np.nan
        gauss_flux, gauss_err = np.nan, np.nan
        cont_err = np.nan
        m_cont, n_cont = np.nan, np.nan
        line_pixels, box_pixels = np.nan, np.nan
        detection, success_fit = np.nan, np.nan

    line_pixels = sigma * number_sigmas
    box_pixels = small_box_width if res_ratio < small_box_reslimit else large_box_width

    # Detection label
    detection = True if int_ratio >= detection_function(res_ratio) else False

    # Store the spectrum
    mu_index = np.searchsorted(wave, mu_line)
    idx_O, idx_f = mu_index - limit_O, mu_index + limit_f
    wave_box, flux_box = wave[idx_O:idx_f], flux[idx_O:idx_f]

    x_line_container[idx, :] = wave_box
    y_line_container[idx, :] = flux_box
    detect_container[idx] = detection

    # Line location for the box selection
    mu_index = np.searchsorted(wave_box, mu_line)

    df_values.loc[idx, :] = (amp, sigma,
                           noise_i, resolution_fix,
                           amp/noise_i, sigma/resolution_fix,
                           intg_flux, intg_err,
                           gauss_flux, gauss_err,
                           theo_flux, true_error,
                           cont, cont_err,
                           m_cont, n_cont,
                           line_pixels, box_pixels,
                           detection, success_fit,
                           mu_index)


# Rearrange the lines training sample into a SAMPLE_SIZE x BOX_SIZE array
column_names = np.full(box_size, 'Pixel')
column_names = np.char.add(column_names, np.arange(box_size).astype(str))
wave_db = pd.DataFrame(data=x_line_container, columns=column_names)
flux_db = pd.DataFrame(data=y_line_container, columns=column_names)

# Save the wavelength, flux and detection (bool) arrays in three different files
wave_db.to_csv(f'{output_folder}/sample_wave_{version}.csv', index=False)
flux_db.to_csv(f'{output_folder}/sample_flux_{version}.csv', index=False)
np.savetxt(f'{output_folder}/sample_detection_{version}.csv', detect_container)
lime.save_frame(sample_database, df_values)




