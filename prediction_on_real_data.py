import numpy as np
from astropy.io import fits
import joblib
import lime


def import_osiris_fits(file_address, ext=0):

    # Open fits file
    with fits.open(file_address) as hdul:
        data, header = hdul[ext].data, hdul[ext].header

    w_min, dw, n_pix = header['CRVAL1'],  header['CD1_1'], header['NAXIS1']
    w_max = w_min + dw * n_pix
    wavelength = np.linspace(w_min, w_max, n_pix, endpoint=False)

    return wavelength, data, header

def normalize_lines(y, size_array = 750):

    array_container = np.zeros(size_array)

    norm = np.sqrt((y[0]**2+y[-1]**2)/2)
    y_norm = (y - norm)/norm

    array_container[:y.size] = y_norm

    return array_container

# Load the model
filename = 'sgd_v1.joblib'
sgd_clf = joblib.load(filename)

# State the data files
obsFitsFile = '/home/vital/PycharmProjects/lime/examples/sample_data/gp121903_BR.fits'
lineMaskFile = '/home/vital/PycharmProjects/lime/examples/sample_data/osiris_bands.txt'
cfgFile = '/home/vital/PycharmProjects/lime/examples/sample_data/config_file.cfg'

# Load configuration
obs_cfg = lime.load_cfg(cfgFile)
fit_cfg = obs_cfg['gp121903_line_fitting']

# Load mask
maskDF = lime.load_lines_log(lineMaskFile)

# Load spectrum
wave, flux, header = import_osiris_fits(obsFitsFile)

# Declare line measuring object
z_obj = obs_cfg['sample_data']['z_array'][2]
norm_flux = obs_cfg['sample_data']['norm_flux']
gp_spec = lime.Spectrum(wave, flux, redshift=z_obj, norm_flux=norm_flux)
gp_spec.plot_spectrum()


# Measure the emission lines
lineList = ['H1_3750A', 'H1_4861A', 'O3_4959A', 'H1_6563A_b', 'S2_4069A', 'Ar4_4740A']

for i, lineLabel in enumerate(lineList):
    wave_regions = maskDF.loc[lineLabel, 'w1':'w6'].values

    gp_spec.fit_from_wavelengths(lineLabel, wave_regions, user_cfg=fit_cfg)

    # Establish spectrum line and continua regions
    idcsEmis, idcsBlue, idcsRed = gp_spec.define_masks(gp_spec.wave_rest, gp_spec.flux, gp_spec.mask, merge_continua=False)
    wave_i, flux_i = gp_spec.wave[idcsEmis], gp_spec.flux[idcsEmis]
    flux_i_norm = normalize_lines(flux_i)
    line_prediction = sgd_clf.predict([flux_i_norm])
    print(f'{lineLabel} (line): {line_prediction}:')

for i, lineLabel in enumerate(lineList):
    wave_regions = maskDF.loc[lineLabel, 'w1':'w6'].values

    gp_spec.fit_from_wavelengths(lineLabel, wave_regions, user_cfg=fit_cfg)

    # Establish spectrum line and continua regions
    idcsEmis, idcsBlue, idcsRed = gp_spec.define_masks(gp_spec.wave_rest, gp_spec.flux, gp_spec.mask,
                                                       merge_continua=False)
    wave_i, flux_i = gp_spec.wave[idcsBlue], gp_spec.flux[idcsBlue]
    flux_i_norm = normalize_lines(flux_i)
    line_prediction = sgd_clf.predict([flux_i_norm])
    print(f'{lineLabel} (continuum): {line_prediction}:')

    # gp_spec.display_results(fit_report=True, plot=True, log_scale=True, frame='obs')
