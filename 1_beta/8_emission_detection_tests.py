import numpy as np
import lime
import matplotlib.pyplot as plt
from pathlib import Path
from tools import load_nirspec_fits, ml_line_detection


# Data location
cfg = lime.load_cfg('../2_release/config_file.toml')
fits_folder = Path(cfg['data_location']['fits_folder'])
output_folder = Path(cfg['data_location']['output_folder'])
fits_file = fits_folder/'F170LP-G235M_NRS1/fluxcal/F170LP-G235M_NRS1_s40017921_fluxcal_x1d.fits'
BOX_SIZE = int(cfg['ml_grid_design']['box_size_pixels'])
version = 'v2_cost1_logNorm'
machine_path = output_folder/f'LogitistRegression_{version}.joblib'


# Get the observation data
wave, flux, err, hdr = load_nirspec_fits(fits_file)
z_obj = 3.179
norm_flux = 1e-5

# Create a spectrum object
spec = lime.Spectrum(wave, flux, norm_flux=norm_flux, redshift=z_obj, units_wave='um', units_flux='mJy')
obj_cont = spec.continuum_fitting()
# spec.plot_spectrum(spec_label=fits_file.stem, comp_array=obj_cont)

norm_spec = np.log10((spec.flux/obj_cont - 1) + 10)

# _fig, _ax = plt.subplots(figsize=(12, 12))
# _ax.step(spec.wave, norm_spec)
# plt.show()

spec.line_detection(ml_detection=True)

# Spectrum detection via machine learning
ml_detection_mask = ml_line_detection(norm_spec, BOX_SIZE, machine_path, normalize=False)
fig, ax = plt.subplots(figsize=(12, 12))
ax.step(spec.wave, norm_spec)
ax.scatter(spec.wave[ml_detection_mask], norm_spec[ml_detection_mask], marker='o', color='palegreen')
ax.set_title('Line detection')
plt.show()


# # Test on a real line location
# idx_3 = np.searchsorted(spec.wave, (2.08732))
# idx_4 = idx_3 + 11
# detection_array = ml_line_detection(norm_spec[idx_3:idx_4], int(BOX_SIZE), machine_path, normalize=False)
#
# _fig, _ax = plt.subplots(figsize=(12, 12))
# _ax.step(spec.wave[idx_3:idx_4], norm_spec[idx_3:idx_4])
# _ax.step(spec.wave[idx_3:idx_4][detection_array], norm_spec[idx_3:idx_4][detection_array])
# plt.show()
