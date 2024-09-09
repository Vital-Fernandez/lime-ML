import numpy as np
import lime
from pathlib import Path
from tools import load_nirspec_fits

obj_subSample = [0,1,8,14,18,20,21,28,29,35,42,45,49,54,75,79,86,86,98,111,114,115,118,122,139,162,169,170,174,179,181,
188,197,205,212,216,221,223,246,251,275,283,296,327,328,346,383,390,392,419,432]

# Data location
cfg = lime.load_cfg('config_file.ini')
fits_folder = Path(cfg['data_location']['fits_folder'])
output_folder = Path(cfg['data_location']['output_folder'])
dispersion_folders = list(fits_folder.glob("*NRS*"))

i_counter = 0
for folder in dispersion_folders:

    fits_list = folder.glob("*x1d.fits")

    for fits in fits_list:

        if i_counter in obj_subSample:

            # Load the data
            wave, flux, err, hdr = load_nirspec_fits(fits)

            # Mask nan, zero and negative entries from the spectrum
            mask = np.isnan(err) | (flux < 0) | (flux == 0)

            # Reject spectra with more invalid entries than valid
            if np.sum(mask)/mask.size < 0.5:

                # Define the spectrum object
                spec = lime.Spectrum(wave, flux, err, units_wave='um', units_flux='mJy', pixel_mask=mask)

                # Fit the continuum and find the peaks
                spec.line_detection(poly_degree=[3, 4, 4, 4], plot_cont_calc=False, plot_peak_calc=True)

        i_counter += 1
