import numpy as np
import lime
import pandas as pd
from lime.model import gaussian_model
from lime.recognition import detection_function
from pathlib import Path
from matplotlib import pyplot as plt
from tools import normalization_1d
from pip._internal.cli.progress_bars import get_download_progress_renderer
from tqdm import tqdm
from itertools import product

cfg_file = 'config_file.toml'
cfg = lime.load_cfg(cfg_file)
output_folder = Path(cfg['data_location']['output_folder'])

# Recover the grid parameters
amp_array = np.array(cfg['data_grid']['amp_array'])
sigma_gas_array = np.array(cfg['data_grid']['sigma_gas_um_array']) * 1000
delta_lam_array = np.array(cfg['data_grid']['delta_lambda_um_array']) * 1000
noise_sig_array = np.array(cfg['data_grid']['noise_array'])

version = cfg['data_grid']['version']
limit_O, limit_f = cfg['data_grid']['box_limits']
box_size = limit_O + limit_f
sample_database = output_folder/f'sample_database_{version}.txt'
small_box_reslimit = cfg['data_grid']['small_box_reslimit']
small_box_width = cfg['data_grid']['small_box_size']
large_box_width = cfg['data_grid']['large_box_size']

# Perform fits check
lime_fit_check = False

# Sample size
int_sample_size = cfg['data_grid']['int_sample_size']
int_sample_limts = cfg['data_grid']['int_sample_limits']

res_sample_size = cfg['data_grid']['res_sample_size']
res_sample_limts = cfg['data_grid']['res_sample_limits']

int_ratio_range = np.logspace(int_sample_limts[0], int_sample_limts[1], int_sample_size, base=10000)
res_ratio_range = np.linspace(res_sample_limts[0], res_sample_limts[1], res_sample_size)

sample_size = int_sample_size * res_sample_size
noise_array = np.random.uniform(0.50,10, size=sample_size)
resolution_fix = 1

# Line parameters
line = cfg['flux_testing_single']['line']
mu_line = cfg['flux_testing_single']['mu']
data_points = cfg['flux_testing_single']['data_points']
width_factor = cfg['flux_testing_single']['width_factor']

# Create empty container for the data
df_values = pd.DataFrame(index=np.arange(sample_size), columns=cfg['data_grid']['headers'])

x_line_container = np.full([sample_size, box_size], np.nan)
y_line_container = np.full([sample_size, box_size], np.nan)
detect_container = np.full(sample_size, np.nan)

# Progress variables
combinations = np.array(list(product(int_ratio_range, res_ratio_range)))
bar = tqdm(combinations, desc="Item", mininterval=0.2, unit=" combinations")
# pbar = tqdm(array_product, unit=" line")

for idx, (int_ratio, res_ratio) in enumerate(bar):

    noise_i = noise_array[idx]
    random_noise = np.random.normal(0, noise_i, data_points)

    amp = int_ratio * noise_i
    sigma = res_ratio * resolution_fix
    inst_delta = resolution_fix

    # Continuum level
    cont = 0

    # Wavelength limits spectrum
    w0 = mu_line - inst_delta * data_points / 2
    wf = mu_line + inst_delta * data_points / 2

    # Compute spectrum
    wave = np.arange(w0, wf, inst_delta)
    flux = gaussian_model(wave, amp, mu_line, sigma) + random_noise + cont

    # Theoretical flux
    theo_flux = amp * 2.5066282746 * sigma
    true_error = noise_i * np.sqrt(2 * width_factor * inst_delta * sigma)

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
            gauss_flux, gauss_err = spec.log.loc[line, ['profile_flux', 'profile_flux_err']].to_numpy()
        else:
            gauss_flux, gauss_err = np.nan, np.nan

        # Store the measurements
        intg_flux, intg_err, cont, cont_err, m_cont, n_cont = spec.log.loc[line, ['intg_flux', 'intg_flux_err',
                                                                                  'cont', 'cont_err',
                                                                                  'm_cont', 'n_cont']].to_numpy()

    else:

        intg_flux, intg_err = np.nan, np.nan
        gauss_flux, gauss_err = np.nan, np.nan
        cont_err = np.nan
        m_cont, n_cont = np.nan, np.nan
        line_pixels, box_pixels = np.nan, np.nan
        detection, success_fit = np.nan, np.nan

    line_pixels = res_ratio * 8
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
                           noise_i, inst_delta,
                           amp/noise_i, sigma/inst_delta,
                           intg_flux, intg_err,
                           gauss_flux, gauss_err,
                           theo_flux, true_error,
                           cont, cont_err,
                           m_cont, n_cont,
                           line_pixels, box_pixels,
                           detection, success_fit,
                           mu_index)


# Save the sample database


# Rearrange the lines training sample into a SAMPLE_SIZE x BOX_SIZE array
column_names = np.full(box_size, 'Pixel')
column_names = np.char.add(column_names, np.arange(box_size).astype(str))
wave_db = pd.DataFrame(data=x_line_container, columns=column_names)
flux_db = pd.DataFrame(data=y_line_container, columns=column_names)

# Save the wavelength, flux and detection (bool) arrays in three different files
wave_db.to_csv(f'{output_folder}/sample_wave_{version}.csv', index=False)
flux_db.to_csv(f'{output_folder}/sample_flux_{version}.csv', index=False)
np.savetxt(f'{output_folder}/sample_detection_{version}.csv', detect_container)
lime.save_log(df_values, sample_database)
print(df_values)




