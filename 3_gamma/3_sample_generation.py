import numpy as np
import pandas as pd

import lime
from lime.model import gaussian_model
from lime.recognition import detection_function, cosmic_ray_function, broad_component_function

from pathlib import Path
from tqdm import tqdm
from itertools import product


# Read sample configuration
cfg_file = 'training_sample_v3.toml'
sample_params = lime.load_cfg(cfg_file)

version = sample_params['data_labels']['version']
cfg = sample_params[f'training_data_{version}']

# State the output files
output_folder = Path(sample_params['data_labels']['output_folder'])/version
output_folder.mkdir(parents=True, exist_ok=True)
sample_database = f'{output_folder}/training_multi_sample_{version}.csv'

# Grid parameters
n_sigma = cfg['n_sigma']
err_n_sigma =  cfg['err_n_sigma']

box_pixels = cfg['box_pixels']
sigma_pixels = box_pixels/n_sigma

res_ratio_min = cfg['res-ratio_min']
int_ratio_min, int_ratio_max, int_ratio_base = cfg['int-ratio_min'], cfg['int-ratio_max'], cfg['int-ratio_log_base']
cr_boundary = cfg['cosmic-ray']['cosmic-ray_boundary']
# inverted_classes = {value: key for key, value in cfg['classes'].items()}

instr_res = cfg['instr_res']
cont_level = cfg['cont_level']
sample_size = cfg['int-ratio_points'] * cfg['res-ratio_points']

int_ratio_min_log = np.log(int_ratio_min) / np.log(int_ratio_base)
int_ratio_max_log = np.log(int_ratio_max) / np.log(int_ratio_base)

int_ratio_range = np.logspace(int_ratio_min_log, int_ratio_max_log, cfg['int-ratio_points'], base=int_ratio_max)
res_ratio_range = np.linspace(res_ratio_min, sigma_pixels, cfg['res-ratio_points'])
combinations = np.array(list(product(int_ratio_range, res_ratio_range)))
print(f'Int_ratio size: {cfg["int-ratio_points"]}')
print(f'Res_ratio size: {cfg["res-ratio_points"]}')
print(f'combinations : {combinations.size}')


doublet_min_res, doublet_max_res =  cfg['doublet']['min_res_ratio'], cfg['doublet']['max_res_ratio']
doublet_min_detection_factor = cfg['doublet']['min_detection_factor']

broad_int_max = cfg['broad']['broad_int_max_factor'] * cfg['int-ratio_max']
broad_int_min_factor = cfg['broad']['min_detection_factor']

narrow_low_limit_arr_log = np.log(combinations[:, 0] * cfg['broad']['narrow_broad_min_factor']) / np.log(int_ratio_base)
narrow_upper_limit_arr_log = np.log(int_ratio_max * cfg['broad']['narrow_broad_max_factor']) / np.log(int_ratio_base)
narrow_int_arr = np.power(int_ratio_base, np.random.uniform(narrow_low_limit_arr_log, narrow_upper_limit_arr_log))

narrow_broad_ratio = narrow_int_arr/combinations[:, 0]
res_narrow_upper_limit_arr = combinations[:, 1] / broad_component_function(narrow_broad_ratio)
narrow_res_arr = np.random.uniform(res_ratio_min, res_narrow_upper_limit_arr)

# Generate the random data
uniform_noise_arr = np.random.uniform(cfg['noise_min'], cfg['noise_max'], size=(sample_size,1))
normal_noise_matrix = np.random.normal(loc=0, scale=uniform_noise_arr, size=(sample_size, cfg['uncrop_array_size']))

doublet_res_factor_arr = np.random.uniform(cfg['doublet']['res-ratio_difference_factor'][0],
                                           cfg['doublet']['res-ratio_difference_factor'][1], size=sample_size)

doublet_int_factor_arr = np.random.uniform(cfg['doublet']['int-ratio_difference_factor'][0],
                                           cfg['doublet']['int-ratio_difference_factor'][1], size=sample_size)


# Generate "wavelength" range
mu_line = cfg['mu_line']
wave_arr = np.arange(- cfg['uncrop_array_size'] / 2, cfg['uncrop_array_size'] / 2, instr_res)
idx_zero = np.searchsorted(wave_arr, mu_line)
idx_0, idx_f = int(idx_zero - box_pixels/2), int(idx_zero + box_pixels/2)

# Containers for the data
flux_containers, coords_containers = {}, {}
for feature_label, feature_number in cfg['classes'].items():
    flux_containers[feature_label] = np.full([sample_size, box_pixels], np.nan)
    coords_containers[feature_label] = np.full([sample_size, 2], np.nan)

# Extra container for the narrow component of the broad feature
coords_containers['narrow'] = np.full([sample_size, 2], np.nan)
flux_containers['narrow'] = np.full([sample_size, box_pixels], np.nan)

# Loop through the conditions
bar = tqdm(combinations, desc="Item", mininterval=0.2, unit=" combinations")
for idx, (int_ratio, res_ratio) in enumerate(bar):

    # Continuum components
    cont_arr = cont_level
    white_noise_arr = normal_noise_matrix[idx, :]
    noise_i = uniform_noise_arr[idx]

    # Line components
    amp = int_ratio * noise_i
    sigma = res_ratio * instr_res
    line_pixels = sigma * n_sigma
    theo_flux = amp * 2.5066282746 * sigma
    true_error = noise_i * np.sqrt(2 * err_n_sigma * instr_res * sigma)

    # Reference values
    detection_value = detection_function(res_ratio)
    cosmic_ray_res = cosmic_ray_function(int_ratio, res_ratio_check=False)


    # Detection cases
    if int_ratio >= detection_value:

        # Flux array
        flux_arr = gaussian_model(wave_arr, amp, mu_line, sigma) + white_noise_arr + cont_arr

        # Line
        if res_ratio > cosmic_ray_res:
            shape_class = 'emission'

        # Single pixel
        else:
            if int_ratio > cr_boundary: # Cosmic ray
                shape_class = 'cosmic-ray'
            else: # Pixel line
                shape_class = 'pixel-line'

        # Store the data
        flux_containers[shape_class][idx, :] = flux_arr[idx_0:idx_f]
        coords_containers[shape_class][idx, :] = int_ratio, res_ratio

    # Continuum cases
    else:
        flux_arr = gaussian_model(wave_arr, amp, mu_line, sigma) + white_noise_arr + cont_arr
        shape_class = 'white-noise'

        # Store the data
        flux_containers[shape_class][idx, :] = flux_arr[idx_0:idx_f]
        coords_containers[shape_class][idx, :] = int_ratio, res_ratio


    # Doublet
    if ((res_ratio >= doublet_min_res) and (res_ratio <= doublet_max_res) and
            (int_ratio >= (doublet_min_detection_factor * detection_value))):

        # Add randomness to Gaussian profiles
        sigma1, sigma2 = sigma, sigma * doublet_res_factor_arr[idx]
        amp1, amp2 = amp, amp * doublet_int_factor_arr[idx]
        mu1, mu2 = mu_line - sigma1, mu_line + sigma2

        # Generate the profiles
        gauss1 = gaussian_model(wave_arr, amp1, mu1, sigma1)
        gauss2 = gaussian_model(wave_arr, amp2, mu2, sigma2)
        flux_arr = gauss1 + gauss2 + white_noise_arr + cont_arr

        # Store the data
        flux_containers['doublet'][idx, :] = flux_arr[idx_0:idx_f]
        coords_containers['doublet'][idx, :] = int_ratio, res_ratio

    # Broad line
    if ((res_ratio > cosmic_ray_res) and (int_ratio >= (broad_int_min_factor * detection_value)) and
            (int_ratio < broad_int_max)):

        ampB, ampN = amp, narrow_int_arr[idx]
        sigmaB, sigmaN = sigma, narrow_res_arr[idx]
        muB, muN = 0, 0

        gaussB = gaussian_model(wave_arr, ampB, muB, sigmaB)
        gaussN = gaussian_model(wave_arr, ampN, muN, sigmaN)
        flux_arr = gaussB + gaussN + white_noise_arr + cont_arr

        # Store the data
        flux_containers['broad'][idx, :] = flux_arr[idx_0:idx_f]
        coords_containers['broad'][idx, :] = int_ratio, res_ratio

        flux_containers['narrow'][idx, :] = flux_arr[idx_0:idx_f]
        coords_containers['narrow'][idx, :] = narrow_int_arr[idx], narrow_res_arr[idx]


print(f'Joining the files')
list_ids, list_fluxes, list_coords = [], [], []
for feature_label, id_number in cfg['classes'].items():
    if feature_label in flux_containers:

        flux_arr = flux_containers[feature_label]
        idcs_valid = ~np.isnan(flux_arr.sum(axis=1))

        list_fluxes.append(flux_arr[idcs_valid, :])
        list_coords.append(coords_containers[feature_label][idcs_valid, :])
        list_ids.append(np.full(flux_arr[idcs_valid, :].shape[0], feature_label))

        # Store the coordinates from the narrow element of the broad class
        if feature_label == 'broad':
            extra_number_id = id_number + 0.5
            list_fluxes.append(flux_containers['narrow'][idcs_valid, :])
            list_coords.append(coords_containers['narrow'][idcs_valid, :])
            list_ids.append(np.full(flux_arr[idcs_valid, :].shape[0], 'narrow'))

print(f'Stacking the tables')
total_sample = np.hstack([np.hstack(list_ids).reshape(-1, 1), np.vstack(list_coords)])
total_sample = np.hstack([total_sample, np.vstack(list_fluxes)])

# Convert to dataframe and save it
column_names = np.full(box_pixels, 'Pixel')
column_names = ['shape_class', 'int_ratio', 'res_ratio'] + list(np.char.add(column_names, np.arange(box_pixels).astype(str)))

print(f'Saving to: {sample_database}')
# lime.save_frame(f'{output_folder}/training_multi_sample_{version}.fits', sample_database)
sample_db = pd.DataFrame(data=total_sample, columns=column_names)
sample_db.to_csv(sample_database, index=False, compression='gzip')