import numpy as np
import joblib
from scipy import stats
from astropy.io import fits
from lmfit.models import PolynomialModel
import matplotlib.pyplot as plt

from lime.model import gaussian_model

c_kmpers = 299792.458  # Km/s

params_labels_dict = dict(amp=r'$A_{gas}$',
                          mu=r'$\mu$',
                          mu_index=r'$\mu_{index}$',
                          sigma=r'$\sigma_{gas}$',
                          lambda_step=r'$\Delta \lambda$',
                          noise=r'$\sigma_{noise}$',
                          n_cont10=r'$n_{cont_{10}}$',
                          n_cont1=r'$n_{cont_{1}}$',
                          amp_noise_ratio=r'$\frac{A_{gas}}{\sigma_{noise}}$',
                          sigma_lambda_ratio=r'$\frac{\sigma_{gas}}{\delta \lambda}$')

params_units_dict = dict(A=r'$\AA$',
                         um=r'$\mu m$',
                         nm=r'$nm$',
                         flux=r'flux units',
                         pix='pixels')

params_title_dict = dict(amp=r'Emission line amplitude',
                         mu=r'Emission line center',
                         mu_index=r'Emission line center (pixel in array)',
                         sigma=r'Gas velocity dispersion',
                         lambda_step=r'Spectral resolution',
                         noise=r'Spectrum noise',
                         n_cont10=r'Continuum level (10)',
                         n_cont1=r'Continuum level (1)',
                         amp_noise_ratio='Line amplitude by continuum noise ratio',
                         sigma_lambda_ratio=r'Line dispersion velocity by spectrometer resolution')

params_names_dict = dict(trunc_normal='Truncated normal',
                         uniform='Uniform',
                         discrete_uniform='Discrete uniform')


def ml_line_detection(flux_array, box_width, machine_path, normalize=True):

    # Load the machine file
    ml = joblib.load(machine_path)

    # flux_array = np.array(flux_array, ndmin=2)
    detection_mask = np.zeros(flux_array.shape).astype(bool)

    flux_array = np.array(flux_array, ndmin=2)

    # Case of 1D
    spectrum_pixels = flux_array.shape[1]
    for i in np.arange(spectrum_pixels):
        if i + box_width <= spectrum_pixels:
            y = flux_array[:, i:i + box_width]
            y = y/np.max(y, axis=1) if normalize else y
            if not np.any(np.isnan(y)):
                detection_mask[i:i + box_width] = detection_mask[i:i + box_width] | ml.predict(y)[0]
                # print(f'y {i} ({np.sum(y)}): {ml.predict(y)[0]}')

    return detection_mask


def detection_function(x_ratio):

    function = 5 + 0.5/np.square(x_ratio) + 0.5 * np.square(x_ratio)

    # function = 2.5 + 1/np.square(x_ratio - 0.1) + 0.5 * np.square(x_ratio)

    return function


def load_nirspec_fits(file_address, fits_shape='1d'):

    if fits_shape == '1d':
        ext = 1
        with fits.open(file_address) as hdu_list:
            data_table, header = hdu_list[ext].data, hdu_list[ext].header
            wave_array, flux_array, err_array = data_table['WAVELENGTH'], data_table['FLUX'], data_table['FLUX_ERROR']

    return wave_array, flux_array, err_array, header


def sn_formula(amp, mu, sigma, lambda_step, noise):

    SN = (amp/noise) * np.sqrt(np.pi * sigma / 3)

    return SN


def normal_curve(amp, mu, sigma, lambda_step, noise):

    rnd = np.random.RandomState()

    # Box size depends on the gas spatial velocity
    # w_b, w_r = 2, 5
    # w_lim = (-3 * sigma - w_b, 3 * sigma + w_r)
    # x = np.arange(w_lim[0], w_lim[1], lambda_step)

    # Box size remains constant
    w_lim = np.array([-11 * lambda_step/2, 11 * lambda_step/2])
    # x = np.arange(w_lim[0], w_lim[1], lambda_step)
    x = np.linspace(w_lim[0], w_lim[1], 11)

    cont = rnd.normal(0, noise, x.size)
    y = gaussian_model(x, amp, mu, sigma) + cont

    y_norm = y / np.max(y)

    return x, y_norm


from plots import plot_distribution


def param_distribution(sample_size, display=False, **kwargs):

    param_name = kwargs['param_label']
    units_param = kwargs['units']
    dist_type = kwargs['dist']

    if dist_type == 'trunc_normal':

        mu, sigma = kwargs['mu'], kwargs['sigma']
        low_lim, high_lim = kwargs['low_limit'], kwargs['high_limit']
        an, bn = (low_lim - mu) / sigma, (high_lim - mu) / sigma
        x = np.linspace(low_lim, high_lim, 100)

        pdf = stats.truncnorm.pdf(x, an, bn, loc=mu, scale=sigma)
        param_array = stats.truncnorm.rvs(an, bn, loc=mu, scale=sigma, size=sample_size)

    if dist_type == 'uniform':

        low_lim, high_lim = kwargs['low_limit'], kwargs['high_limit']
        x = np.linspace(low_lim, high_lim, 100)

        # Generate the density function and the maximum array
        pdf = stats.uniform.pdf(x, low_lim, high_lim)
        param_array = stats.uniform.rvs(low_lim, high_lim, size=sample_size)

    if dist_type == 'discrete_uniform':

        low_lim, high_lim = kwargs['low_limit'], kwargs['high_limit']
        x = np.linspace(low_lim, high_lim, 100)

        pdf = stats.randint.pmf(x, low_lim, high_lim)
        param_array = stats.randint.rvs(low_lim, high_lim, size=sample_size)

    print(f'{param_name}:\nInput limits: {low_lim}-{high_lim}\nActual limits: {np.min(param_array)}-{np.max(param_array)}\n')

    if display:

        title = f'{params_title_dict[param_name]}, {params_labels_dict[param_name]}'

        x_label = f'{params_labels_dict[param_name]}'
        x_label += '' if units_param == 'none' else f' ({params_units_dict[units_param]})'

        if dist_type in ['trunc_normal']:
            label = f'{params_names_dict[dist_type]} distribution\n' \
                    f'$\mu$={mu}, $\sigma$={sigma}, limits = ({low_lim}, {high_lim})'

        if dist_type in ['uniform', 'discrete_uniform']:
            label = f'{params_names_dict[dist_type]} distribution\n' \
                    f'limits = ({low_lim}, {high_lim})'

        plot_distribution(x, param_array, pdf, dist_label=label, title=title, x_label=x_label,
                          verbose=True)

    return param_array




