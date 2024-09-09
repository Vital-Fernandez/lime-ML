import numpy as np
import lime
from tools import load_nirspec_fits
from pathlib import Path

# Data location
cfg = lime.load_cfg('config_file.ini')
fits_folder = Path(cfg['data_location']['fits_folder'])
output_folder = Path(cfg['data_location']['output_folder'])
fits_file = fits_folder/'F170LP-G235M_NRS1/fluxcal/F170LP-G235M_NRS1_s40017921_fluxcal_x1d.fits'

# Get the observation data
wave, flux, err, hdr = load_nirspec_fits(fits_file)
z_obj = 3.179
norm_flux = 1e-5

# Create a spectrum object
spec = lime.Spectrum(wave, flux, redshift=z_obj, norm_flux=norm_flux, units_wave='um', units_flux='mJy')
spec.plot_spectrum(spec_label=fits_file.stem)

# Generate a spectral mask from the one available in LiMe
obj_mask = lime.spectral_mask_generator(lines_list=['H1_0.4861um', 'O3_0.4959um', 'O3_0.5007um', 'He1_0.5876um',
                                                    'H1_0.6563um'], units_wave='um')

# Line detection via peak intensity
peaks_table, matched_masks_DF = spec.match_line_mask(obj_mask, noise_region=np.array([0.55, 0.56]))
spec.plot_spectrum(peaks_table=peaks_table, match_log=matched_masks_DF, spec_label=fits_file.stem)

# Measure the emission lines
for i, lineLabel in enumerate(matched_masks_DF.index.values):
    wave_regions = matched_masks_DF.loc[lineLabel, 'w1':'w6'].values
    spec.fit_from_wavelengths(lineLabel, wave_regions)
    spec.display_results(fit_report=True)
