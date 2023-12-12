import numpy as np
import pandas as pd
from scipy import stats
from functions import SAMPLE_SIZE, SEED_VALUE, SAMPLE_LINE_PERCENTAGE, PARMS_FILE
from functions import plot_ratio_distribution, plot_distribution
from matplotlib import rcParams

show_plots = False

STANDARD_PLOT = {'figure.figsize': (10, 8),
                 'axes.titlesize': 16,
                 'axes.labelsize': 16,
                 'legend.fontsize': 14,
                 'xtick.labelsize': 14,
                 'ytick.labelsize': 14}

rcParams.update(STANDARD_PLOT)


# line standard deviation velocity distribution (angstroms)
mu, sigma = 6.0, 6
low_lim, high_lim = 0.01, 13
an, bn = (low_lim - mu) / sigma, (high_lim - mu) / sigma

x = np.linspace(0, 13, 100)
pdf = stats.truncnorm.pdf(x, an, bn, loc=mu, scale=sigma)
sigma_gaussian_array = stats.truncnorm.rvs(an, bn, loc=mu, scale=sigma, size=SAMPLE_SIZE, random_state=SEED_VALUE)

# -- Plot the distribution
label = f'Truncated Gaussian distribution\n$\mu$={mu}, $\sigma$={sigma}, limits = ({low_lim}, {high_lim})'
title = r'Velocity dispersion, $\sigma_{g}$ $(\AA)$'
plot_distribution(x, sigma_gaussian_array, pdf, dist_label=label, title=title, x_label=r' $\sigma_{g}$',
                  verbose=show_plots)

# Spectrum pixel width (angstroms)
mu_i, sigma_i = (1.5 * sigma_gaussian_array)/2, (1.5 * sigma_gaussian_array)/2
low_lim, high_lim = 0.01, 1.5 * sigma_gaussian_array
an, bn = (low_lim - mu_i) / sigma_i, (high_lim - mu_i) / sigma_i

lambda_step_array = stats.truncnorm.rvs(an, bn, loc=mu_i, scale=sigma_i, size=(1, sigma_gaussian_array.size),
                                        random_state=SEED_VALUE)[0]

# -- Plot the distribution
label = f'Truncated Gaussian distribution\n$\mu=1.5\cdot\sigma_{{g}}$, $\sigma=1.5\cdot\sigma_{{g}}$,' \
        f' limits = (0.01, $1.5\cdot\sigma_{{g}}$)'
title = r'Pixel width size, $\Delta\lambda$ $(\AA)$'
plot_ratio_distribution(lambda_step_array, x_label='$\Delta\lambda$ $(\AA)$', label=label,  title=title, verbose=show_plots)


# Amplitude emission line
mu, sigma = 2.0, 3
low_lim, high_lim = 0, 4.5
an, bn = (low_lim - mu) / sigma, (high_lim - mu) / sigma

x = np.linspace(0, 5, 100)
pdf = stats.truncnorm.pdf(x, an, bn, loc=mu, scale=sigma)
x_array = stats.truncnorm.rvs(an, bn, loc=mu, scale=sigma, size=SAMPLE_SIZE, random_state=SEED_VALUE)
amp_array = np.power(10, x_array)

label = r'Truncated Gaussian'

# -- Plot the distribution
label = f'Truncated Gaussian distribution\n$\mu$={mu}, $\sigma$={sigma}, limits = ({low_lim}, {high_lim})'
title = r'Velocity dispersion, $A_{g} = 10^{x}$ (flux units)'
plot_distribution(x, x_array, pdf, dist_label=label, title=title, x_label=r'x',
                  verbose=show_plots)

# Noise distribution
mu, sigma = 1.5, 0.5
low_lim, high_lim = 0, 2
an, bn = (low_lim - mu) / sigma, (high_lim - mu) / sigma

x = np.linspace(-1, 5, 100)
pdf = stats.truncnorm.pdf(x, an, bn, loc=mu, scale=sigma)
noise_array = stats.truncnorm.rvs(an, bn, loc=mu, scale=sigma, size=SAMPLE_SIZE, random_state=SEED_VALUE)

label = f'Truncated Gaussian distribution\n$\mu$={mu}, $\sigma$={sigma}, limits = ({low_lim}, {high_lim})'
title = r'Continuum noise dispersion, $\sigma_{cont}$ (flux units)'
plot_distribution(x, noise_array, pdf, dist_label=label, title=title, x_label=r'$\sigma_{cont}$ (flux units)',
                  verbose=show_plots)

# Mask limits distribution
mu, sigma = 0, 2
low_lim, high_lim = 0, 10
an, bn = (low_lim - mu) / sigma, (high_lim - mu) / sigma

x = np.linspace(-1, 5, 100)
pdf = stats.truncnorm.pdf(x, an, bn, loc=mu, scale=sigma)
w_blue_array = stats.truncnorm.rvs(an, bn, loc=mu, scale=sigma, size=SAMPLE_SIZE, random_state=SEED_VALUE) * -1
w_red_array = stats.truncnorm.rvs(an, bn, loc=mu, scale=sigma, size=SAMPLE_SIZE, random_state=SEED_VALUE+1)

label = f'Truncated Gaussian distribution\n$\mu$={mu}, $\sigma$={sigma}, limits = ({low_lim}, {high_lim})'
title = r'Mask size edges $(-w_{blue}$, $w_{red})$ $(\AA)$'
plot_distribution(x, w_red_array, pdf, dist_label=label, title=title, x_label=r'$-w_{blue}$, and $w_{red}$ $(\AA)$',
                  verbose=show_plots)

# Generate the spectra data
rnd = np.random.RandomState(SEED_VALUE)
line_array = rnd.randint(0, 100, size=SAMPLE_SIZE) < SAMPLE_LINE_PERCENTAGE

# Percentage of true cases
SN_array = (amp_array/noise_array) * np.sqrt((np.pi/3) * sigma_gaussian_array)
idcs_low_SN = SN_array[line_array] < 4
print(f'N° lines in sample: {SN_array[line_array].size} of total size {line_array.size}')
print(f'N° lines with S/N < 4: {np.sum(idcs_low_SN)} of {line_array.size} lines')
print(f'S/N min value: {np.min(SN_array[line_array]):.2f}, max value {np.max(SN_array[line_array]):.2f}')
plot_ratio_distribution(SN_array, verbose=show_plots)

# Save as a text file
data_dict = {'amp': amp_array, 'sigma_g': sigma_gaussian_array,
             'noise': noise_array, 'lambda_step': lambda_step_array,
             'w_blue': w_blue_array, 'w_red': w_red_array,
             'line_check': line_array}

# Save as a text file
df = pd.DataFrame(data_dict)
df.to_csv(PARMS_FILE, index=False)
