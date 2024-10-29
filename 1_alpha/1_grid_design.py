import numpy as np
import lime
from pathlib import Path
from astropy.io import fits
from matplotlib import pyplot as plt, rcParams, cm

from tools import c_kmpers, detection_function
from plots import PLOT_CONF, sn_evolution_plot

PLOT_CONF['axes.labelsize'] = 20

rcParams.update(PLOT_CONF)

cfg = lime.load_cfg('../3_gamma/training_sample_v3_old.toml')
fits_folder = Path(cfg['data_location']['fits_folder'])
output_folder = Path(cfg['data_location']['fits_folder'])

all_dispensers = ['PRISM', 'G140M', 'G235M', 'G395M', 'G140H', 'G235H', 'G395H']
example_dispensers = ['PRISM', 'G140M', 'G235M', 'G395M']

# Parameter settings
ml_design_params = cfg['ml_grid_design']

# S/N evolution with the spectrum and emission parameters
amp_array = np.array([1, 2, 3, 4, 5, 10])
noise_array = np.ones(amp_array.size)
sigma_gas_array = np.linspace(0.03, 12.50, 100)
sn_evolution_plot(amp_array, noise_array, sigma_gas_array, SN_low_limit=2.5, SN_upper_limit=5)

# Getting the resolution power of the dispenser-filter combinations
dispersion_curves_folder = fits_folder/'nirspec_dispersion_curves'
list_files = list(dispersion_curves_folder.glob("*.fits"))

dispersion_dict = {}
limits_dict = {}
for file in list_files:
    with fits.open(file) as hdul:
        data, hdr = hdul[1].data, hdul[0].header
        dispenser = hdr['COMPNAME']
        wave, R = data['WAVELENGTH'], data['R']
        dispersion_dict[dispenser] = (wave, R)
        limits_dict[f'{dispenser}_wave'] = (np.round(wave.min(), 2), np.round(wave.max(), 2))
        limits_dict[f'{dispenser}_R'] = (np.floor(R.min()), np.ceil(np.ceil(R.max())))


# Plot the dispensers resolving power curves
fig, ax = plt.subplots(figsize=(12, 12))
for dispenser, curve in dispersion_dict.items():
    if dispenser in example_dispensers:
        wave, resol_power = curve
        ax.plot(wave, resol_power, label=dispenser)
        print(f'{dispenser}_wave_lims_array={limits_dict[f"{dispenser}_wave"][0]},{limits_dict[f"{dispenser}_wave"][1]}')
        print(f'{dispenser}_R_lims_array={limits_dict[f"{dispenser}_R"][0]},{limits_dict[f"{dispenser}_R"][1]}')
ax.update({'xlabel': r'Wavelength range ($\mu m$)', 'ylabel': r'Resolving power', 'title': 'NIRSPec Dispersion curves'})
ax.legend()
plt.show()


# Plot the box size limits
sigma_range = np.linspace(50, 500, 100)
fig, ax = plt.subplots(figsize=(12, 12))
for dispenser, curve in dispersion_dict.items():
    if dispenser in example_dispensers:
        R_max = limits_dict[f"{dispenser}_R"][1]
        b_output = 8 / c_kmpers * sigma_range * R_max
        ax.plot(sigma_range, b_output, label=dispenser)


# Plot box size
label_box_size = f'Current box limit size {ml_design_params["box_size_pixels"]}'
ax.axhline(ml_design_params['box_size_pixels'], linestyle='--', color='black', label=label_box_size)
ax.update({'xlabel': r'$\sigma_{vel}$ ($km/s$)',
           'ylabel': r'$b_{pix}=\frac{8\cdot\sigma_{\lambda}}{\Delta\lambda}=\frac{8}{c}\cdot\sigma_{vel}\cdot R_{max}$',
           'title': 'Box size in pixels as a function of gas velocity dispersion'})
ax.legend()
plt.show()


# Plot sigma_gas/delta_lambda limits per instrument
x_range = np.linspace(0.2, 20, 50)
sigma_range = np.linspace(50, 500, 100)
function = detection_function(x_range)

fig, ax = plt.subplots(figsize=(12, 12))
cmap, color_i = cm.get_cmap(), 0

for dispenser, curve in dispersion_dict.items():
    if dispenser in example_dispensers:
        R_lim = np.array(limits_dict[f"{dispenser}_R"])
        sigma_lim = np.array((sigma_range[0], sigma_range[-1]))
        x_min, x_max = (1/c_kmpers) * sigma_lim * R_lim
        color_curve = cmap(color_i / len(example_dispensers))
        label = f'{dispenser}: {x_min:.2f}-{x_max:.2f}'
        ax.axvspan(x_min, x_max, alpha=0.5, color=color_curve, label=label)
        color_i += 1

# Plot the detection function
ax.plot(x_range, function)

# Wording
ax.update({'title':  f'Parameter space ratios for the NIRSpec dispenser-filter set-ups',
           'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}}$',
           'ylabel':  r'$\frac{A_{gas}}{\sigma_{noise}}$'})
ax.legend()
plt.show()


