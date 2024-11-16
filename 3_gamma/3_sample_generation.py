import numpy as np
import pandas as pd
from pathlib import Path
import lime
from lime.model import gaussian_model
from lime.recognition import detection_function, cosmic_ray_function, broad_component_function

from tqdm import tqdm
from itertools import product
from model_tools import get_memory_usage_of_variables

# Read sample configuration
cfg_file = 'training_sample_v4.toml'
sample_params = lime.load_cfg(cfg_file)

version = sample_params['data_labels']['version']
cfg = sample_params[f'training_data_{version}']

# State the output files
output_folder = Path(sample_params['data_labels']['output_folder'])/version
output_folder.mkdir(parents=True, exist_ok=True)
sample_database_file = f'{output_folder}/training_multi_sample_{version}.csv'

# Grid parameters
n_sigma = cfg['n_sigma']
err_n_sigma =  cfg['err_n_sigma']

box_pixels = cfg['box_pixels']
sigma_pixels = box_pixels/n_sigma

res_ratio_min = cfg['res-ratio_min']
int_ratio_min, int_ratio_max, int_ratio_base = cfg['int-ratio_min'], cfg['int-ratio_max'], cfg['int-ratio_log_base']
cr_boundary = cfg['cosmic-ray']['cosmic-ray_boundary']

instr_res = cfg['instr_res']
sample_size = cfg['int-ratio_points'] * cfg['res-ratio_points']
half_sample = int(sample_size/2)
int_ratio_min_log = np.log(int_ratio_min) / np.log(int_ratio_base)
int_ratio_max_log = np.log(int_ratio_max) / np.log(int_ratio_base)


int_ratio_range = np.logspace(int_ratio_min_log, int_ratio_max_log, cfg['int-ratio_points'], base=int_ratio_max)
res_ratio_range = np.linspace(res_ratio_min, sigma_pixels, cfg['res-ratio_points'])
combinations = np.array(list(product(int_ratio_range, res_ratio_range)))
print(f'Int_ratio size: {cfg["int-ratio_points"]}')
print(f'Res_ratio size: {cfg["res-ratio_points"]}')
print(f'combinations : {combinations.size}')

doublet_min_res = cfg['doublet']['min_res_ratio']
doublet_max_res = cfg['doublet']['max_res_ratio']
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
# uniform_noise_arr = np.random.uniform(cfg['noise_min'], cfg['noise_max'], size=(sample_size,1))
# normal_noise_matrix = np.random.normal(loc=0, scale=uniform_noise_arr, size=(sample_size, cfg['uncrop_array_size']))
uniform_noise_arr = np.full((sample_size,1), 1)
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

# Continuum definition
cont_level = cfg['cont_level']
angle_min = cfg['angle_min']
angle_max = cfg['angle_max']
gradient_arr = np.tan(np.deg2rad(np.random.uniform(angle_min, angle_max, sample_size))) #np.full(sample_size, np.tan(np.deg2rad(0)))
cateto_length = box_pixels/2

m_cont = (sigma_pixels - res_ratio_min)/(angle_max-angle_min)
n_cont = sigma_pixels - angle_max * m_cont

white_noise_min_int_ratio = cfg['white_noise']['min_int_ratio']
white_noise_max_int_ratio = cfg['white_noise']['max_int_ratio']

# Containers for the data
flux_containers, coords_containers = {}, {}
for feature_label, feature_number in cfg['classes'].items():
    if feature_label != 'undefined':
        flux_containers[feature_label] = np.full([sample_size, box_pixels], np.nan)
        coords_containers[feature_label] = np.full([sample_size, 2], np.nan)

# Convert to dataframe and save it
column_names = np.full(box_pixels, 'Pixel')
column_names = ['shape_class', 'int_ratio', 'res_ratio'] + list(np.char.add(column_names, np.arange(box_pixels).astype(str)))

# Extra container for the narrow component of the broad feature
coords_containers['narrow'] = np.full([sample_size, 2], np.nan)
flux_containers['narrow'] = np.full([sample_size, box_pixels], np.nan)

# Loop through the conditions
bar = tqdm(combinations, desc="Item", mininterval=0.2, unit=" combinations")
for idx, (int_ratio, res_ratio) in enumerate(bar):

    # Continuum components
    cont_arr = gradient_arr[idx] * wave_arr + cont_level
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

                # Extra points for the cosmic rays
                flux_containers[shape_class][sample_size - 1 - idx, :] = flux_arr[idx_0:idx_f] * np.random.uniform(0.98, 1.02, box_pixels)
                coords_containers[shape_class][sample_size - 1 - idx, :] = int_ratio, res_ratio

                # Extra points for the cosmic rays
                flux_containers[shape_class][half_sample - idx, :] = flux_arr[idx_0:idx_f] * np.random.uniform(0.98, 1.02, box_pixels)
                coords_containers[shape_class][half_sample - idx, :] = int_ratio, res_ratio

            else: # Pixel line
                shape_class = 'pixel-line'


        # Store the data
        flux_containers[shape_class][idx, :] = flux_arr[idx_0:idx_f]
        coords_containers[shape_class][idx, :] = int_ratio, res_ratio

        # Absorption:
        flux_arr = gaussian_model(wave_arr, -amp, mu_line, sigma) + white_noise_arr + cont_arr

        if res_ratio > cosmic_ray_res:
            shape_class = 'absorption'
        else:
            shape_class = 'dead-pixel'

        flux_containers[shape_class][idx, :] = flux_arr[idx_0:idx_f]
        coords_containers[shape_class][idx, :] = int_ratio, res_ratio

    # Continuum cases
    else:

        # White noise
        if (int_ratio >= white_noise_min_int_ratio) and (int_ratio <= white_noise_max_int_ratio):
            shape_class = 'white-noise'
            white_noise_arr = np.random.normal(loc=0, scale=1/int_ratio, size=cfg['uncrop_array_size'])
            flux_arr = white_noise_arr + cont_arr

        # Continuum
        else:
            shape_class = 'continuum'
            flux_arr = gaussian_model(wave_arr, amp, mu_line, sigma) + white_noise_arr + cont_arr

        # Conversion of the gradient from the form:
        # x_coord = m_cont * res_ratio + n_cont

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



# Loop through the categories and get number of valid entries
print(f'Preparing dataset container')
valid_entries_dict, counter_lines = {}, 0
for feature_label, id_number in cfg['classes'].items():
    if feature_label in flux_containers:

        flux_arr = flux_containers[feature_label]
        idcs_valid = ~np.isnan(flux_arr.sum(axis=1))
        valid_entries_dict[feature_label] = idcs_valid
        counter_lines += idcs_valid.sum()

        # Store the coordinates from the narrow element of the broad class
        if feature_label == 'broad':
            valid_entries_dict['narrow'] = idcs_valid
            counter_lines += idcs_valid.sum()

# Create numpy array and fill it with data:
print('Creating single array')
i_row = 0
label_arr = np.empty(counter_lines).astype(str)
total_sample_arr = np.full((counter_lines, len(column_names)), np.nan)
for feature_label, id_number in cfg['classes'].items():
    if feature_label in flux_containers:

        # Get the number of entries
        idcs_valid = valid_entries_dict[feature_label]
        n_entries = idcs_valid.sum()

        # Assign the values
        label_arr[i_row:i_row+n_entries] = feature_label
        total_sample_arr[i_row:i_row+n_entries, 1:3] = coords_containers[feature_label][idcs_valid, :]
        total_sample_arr[i_row:i_row+n_entries, 3:] = flux_containers[feature_label][idcs_valid, :]

        # Set new starting point
        i_row += n_entries

        # Store the coordinates from the narrow element of the broad class
        if feature_label == 'broad':
            idcs_valid = valid_entries_dict['narrow']
            n_entries = idcs_valid.sum()
            label_arr[i_row:i_row + n_entries] = 'narrow'
            total_sample_arr[i_row:i_row + n_entries, 1:3] = coords_containers[feature_label][idcs_valid, :]
            total_sample_arr[i_row:i_row + n_entries, 3:] = flux_containers[feature_label][idcs_valid, :]
            i_row += n_entries


print('\nClearing the memory 0')
flux_containers.clear()
coords_containers.clear()
del flux_containers
del coords_containers
del flux_arr
del doublet_res_factor_arr
del doublet_int_factor_arr
del normal_noise_matrix
del uniform_noise_arr
del narrow_res_arr
gc.collect()

# Create empty dataframe and add the data
sample_db = pd.DataFrame(data=total_sample_arr, columns=column_names)
sample_db.loc[:, 'shape_class'] = label_arr
del total_sample_arr
del label_arr
gc.collect()
get_memory_usage_of_variables()

# Save the data:
# print(f'\nSaving to: {sample_database_file}')
# sample_db.to_csv(sample_database_file, index=False)

# Save equal number of entries by the minimum amount
min_count = sample_db['shape_class'].value_counts().min()
sample_db = sample_db.groupby('shape_class').sample(n=min_count, random_state=42).reset_index(drop=True)

print(f'\nSaving to: {sample_database_file}, ({min_count} points per category)')
sample_db.to_csv(sample_database_file, index=False)
