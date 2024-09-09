import lime
import numpy as np
from pathlib import Path


# Declare the data location
cfg = lime.load_cfg('config_file.ini')
fits_folder = Path(cfg['data_location']['fits_jwst'])
spectra_list = list(fits_folder.glob("*.csv"))

# Spectra properties
objList = cfg['jwst_spectra']['obj_list']
z_array = cfg['jwst_spectra']['z_array']
norm_flux = cfg['jwst_spectra']['norm_flux']
noise_region = cfg['jwst_spectra']['noise_region_array']

# Loop throught the objects
for i, file_path in enumerate(spectra_list):

    # Load the data
    wave_array, flux_array = np.loadtxt(file_path, unpack=True, skiprows=1, delimiter=',')
    flux_array = flux_array/1e15

    # Create lime spectrum object
    obj_spec = lime.Spectrum(wave_array, flux_array, redshift=z_array[i], units_wave='um', units_flux='Flam')
    obj_spec.convert_units(units_wave='A', norm_flux=norm_flux)

    # Check for the emission lines
    obj_mask = lime.spectral_mask_generator()
    obj_mask = obj_spec.line_detection(lines_log=obj_mask, plot_peak_calc=True, ml_detection=True)
    obj_spec.plot_spectrum(spec_label=f'Line detection, {objList[i]}, z = {obj_spec.redshift}',
                           match_log=obj_mask, frame='rest')

    # Measure the emission lines
    for lineLabel in obj_mask.index.values:
        wave_regions = obj_mask.loc[lineLabel, 'w1':'w6'].values
        obj_spec.fit_from_wavelengths(lineLabel, wave_regions)

    # Display the spectrum with the gaussian profiles
    obj_spec.plot_spectrum(spec_label=f'{objList[i]}, z = {obj_spec.redshift}', include_fits=True)

    # Save the results
    lime.save_line_log(obj_spec.log, fits_folder/f'{objList[i]}.txt')
    lime.save_line_log(obj_spec.log, fits_folder/f'{objList[i]}.pdf', parameters=['eqw',
                                                                                  'intg_flux', 'intg_err',
                                                                                  'v_r', 'v_r_err',
                                                                                  'sigma_vel', 'sigma_vel_err'])

# Load the configuration
cfg = lime.load_cfg('config_file.ini')

# Get the line masks
line_masks = lime.spectral_mask_generator(units_wave='A')

# Create the Spectrum object
spec = lime.Spectrum(wave_array, flux_array, redshift=cfg['sample_data']['redshift'])

# Perform the fittings
for line in line_masks:
    line_mask = line_masks.loc[line].values
    spec.fit_from_wavelengths(line, line_mask, user_cfg=cfg['SHOC579_line_fitting'])

