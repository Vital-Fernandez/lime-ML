import numpy as np
import pandas as pd
import lime
import joblib
from pathlib import Path
from matplotlib import pyplot as plt, rc_context

# Read configuration
cfg_file = 'config_file.toml'
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
desi_spec = lime.Spectrum(wave_desi, flux_desi, units_flux='1e-17*FLAM', redshift=0.054257)
desi_spec.fit.continuum(degree_list=[3, 6, 6], emis_threshold=[3, 2, 1.5], plot_steps=False)
# desi_spec.plot.spectrum(rest_frame=True, include_cont=True)

wave_array, flux_array, err_array = np.loadtxt(output_folder/'manga_spectrum.txt', unpack=True)
manga_spec = lime.Spectrum(wave_array, flux_array, err_array, redshift=0.0475, norm_flux=1e-17, pixel_mask=np.isnan(err_array))
# manga_spec.plot.spectrum(rest_frame=True)

spec_dict = {'desi': desi_spec, 'manga': manga_spec}

for instr, spec in spec_dict.items():

    # Compute the masks:
    model_1d_path = None #subfolder/f'_1D_{label}_{version}_{model_name}.joblib'
    model_2d_path = None #subfolder/f'_2D_{label}_{version}_{model_name}.joblib'
    spec.infer.bands(box_size, conversion_array, scale_type, model_1d_path=model_1d_path, model_2d_path=model_2d_path)

    # mask1D, mask2D = spec.infer.line_1d_pred(confidence=75), spec.infer.line_2d_pred(confidence=75)

    # Get the data:
    wave = spec.wave_rest.data if np.ma.isMaskedArray(spec.wave_rest) else spec.wave_rest
    flux = spec.flux.data if np.ma.isMaskedArray(spec.flux) else spec.flux

    spec.plot.spectrum(detection_band='line_2d_pred', rest_frame=True)

    # with rc_context(fig_conf):
    #
    #     fig, ax = plt.subplots()
    #     ax.step(wave, flux, label=f'{instr} spectrum')
    #     ax.scatter(wave[mask1D], flux[mask1D], label=f'1D model', color='tab:orange')
    #     ax.scatter(wave[mask2D], flux[mask2D], label=f'2D model', color='tab:red', marker='1')
    #     ax.legend()
    #     ax.update({'title': f'Gradient descent, {scale_type} feature scaling', 'xlabel': r'Wavelength $(\AA)$',
    #                'ylabel': 'Flux'})
    #     plt.show()
    #
