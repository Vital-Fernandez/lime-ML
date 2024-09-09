import numpy as np
import pandas as pd
import lime
import joblib
from pathlib import Path
from tools import analysis_check, normalization_1d
from matplotlib import pyplot as plt, rc_context
from tools import feature_scaling, STANDARD_PLOT
import numpy as np

image_conv_array = np.linspace(0,1,33)

def line_detection(spec, box_width, machine_path, approximation, scale_type, log_base=None):

    # Load the model
    model = joblib.load(machine_path)
    model_dim = model.n_features_in_

    # Recover the flux (without mask?)
    input_flux = spec.flux if not np.ma.is_masked(spec.flux) else spec.flux.data

    # Reshape to the detection interval
    range_box = np.arange(box_width)
    n_intervals = input_flux.size - box_size + 1
    input_flux = input_flux[np.arange(n_intervals)[:, None] + range_box]

    # Remove nan entries
    idcs_nan_rows = np.isnan(input_flux).any(axis=1)
    input_flux = input_flux[~idcs_nan_rows, :]

    # Normalize the flux
    input_flux = feature_scaling(input_flux, transformation=scale_type, log_base=log_base)
    # min_pixel_flux = input_flux.min(axis=1)
    # input_flux = np.emath.logn(10000, input_flux - min_pixel_flux[:, np.newaxis] + 1)

    # Perform the 1D detection
    if box_width == model_dim:
        detection_array = model.predict(input_flux)
    else:
        array_2D = np.tile(input_flux[:, None, :], (1, approximation.size, 1))
        array_2D = array_2D > approximation[::-1, None]
        array_2D = array_2D.astype(int)
        array_2D = array_2D.reshape((input_flux.shape[0], 1, -1))
        array_2D = array_2D.squeeze()
        detection_array = model.predict(array_2D)

    # Reshape array original shape and add with of positive entries
    mask = np.zeros(spec.flux.shape, dtype=bool)
    idcs_detect = np.argwhere(detection_array) + range_box
    idcs_detect = idcs_detect.flatten()
    idcs_detect = idcs_detect[idcs_detect < detection_array.size]
    mask[idcs_detect] = True

    return mask

# Read configuration
cfg_file = '../3_gamma/training_sample_v3.toml'
cfg = lime.load_cfg(cfg_file)
output_folder = Path(cfg['data_location']['output_folder'])

# Recover configuration entries
version = cfg['data_grid']['version']
# label = 'small_box_logMinMax'
# scale_type = 'log-min-max'
label = 'small_box_min_max'
scale_type = 'min-max'

subfolder = output_folder/f'_results_{label}_{version}'
model_name = 'GradientDescent'
box_size = cfg['data_grid']['small_box_size']
# conversion_array = np.array(cfg['data_grid']['conversion_array'])
conversion_array = np.array(cfg['data_grid']['conversion_array_min_max'])

# Load the spectra
wave_desi, flux_desi = np.loadtxt('/home/vital/Astrodata/LiMe_ml/desi_spectrum.txt', unpack=True)
desi_spec = lime.Spectrum(wave_desi, flux_desi, redshift=0.054257, norm_flux=1)
# desi_spec.plot.spectrum(rest_frame=True)

wave_array, flux_array, err_array = np.loadtxt(output_folder/'manga_spectrum.txt', unpack=True)
manga_spec = lime.Spectrum(wave_array, flux_array, err_array, redshift=0.0475, norm_flux=1e-17, pixel_mask=np.isnan(err_array))
# manga_spec.plot.spectrum(rest_frame=True)

spec_dict = {'desi': desi_spec, 'manga': manga_spec}

for instr, spec in spec_dict.items():

    # Compute the masks:
    mask_path =  subfolder/f'_1D_{label}_{version}_{model_name}.joblib'
    mask1D = line_detection(spec, box_size, mask_path, conversion_array, scale_type=scale_type, log_base=10000)

    mask_path =  subfolder/f'_2D_{label}_{version}_{model_name}.joblib'
    mask2D = line_detection(spec, box_size, mask_path, conversion_array, scale_type=scale_type, log_base=10000)

    # Get the data:
    wave = spec.wave.data if np.ma.is_masked(spec.wave) else spec.wave
    flux = spec.flux.data if np.ma.is_masked(spec.flux) else spec.flux

    with rc_context(STANDARD_PLOT):

        fig, ax = plt.subplots()
        ax.step(wave, flux, label=f'{instr} spectrum')
        ax.scatter(wave[mask1D], flux[mask1D], label=f'1D model', color='tab:orange')
        ax.scatter(wave[mask2D], flux[mask2D], label=f'2D model', color='tab:red', marker='1')
        ax.legend()
        ax.update({'title': f'Gradient descent, {scale_type} feature scaling', 'xlabel': r'Wavelength $(\AA)$',
                   'ylabel': 'Flux'})
        plt.show()

