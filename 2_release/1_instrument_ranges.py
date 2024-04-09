import numpy as np
import lime
from matplotlib import pyplot as plt, rcParams, cm
from lime.recognition import detection_function
from lime.plots import STANDARD_PLOT

c_kmpers = 299792.458  # Km/s

STANDARD_PLOT['axes.labelsize'] = 20

rcParams.update(STANDARD_PLOT)

cfg = lime.load_cfg('config_file.toml')

# S/N evolution with the spectrum and emission parameters
amp_array = np.array([1, 2, 3, 4, 5, 10])
noise_array = np.ones(amp_array.size)

# Plot sigma_gas/delta_lambda limits per instrument
x_range = np.linspace(0.2, 20, 50)
sigma_vel_range = np.linspace(15, 120, 100)
function = detection_function(x_range)

fig, ax = plt.subplots(figsize=(12, 12))
cmap, color_i = cm.get_cmap(), 0

# Loop though the instrumental ranges
example_dispensers = ['DESI_B', 'DESI_R', 'DESI_Z']
for dispenser in example_dispensers:
    R_lim = np.array(cfg['disp_wavelength_ranges'][f"{dispenser}_R"])
    sigma_lim = np.array((sigma_vel_range[0], sigma_vel_range[-1]))
    x_min, x_max = (1/c_kmpers) * sigma_lim * R_lim
    color_curve = cmap(color_i / len(example_dispensers))
    label = f'{dispenser}: {x_min:.2f}-{x_max:.2f}'
    ax.axvspan(x_min, x_max, alpha=0.2, color=color_curve, label=label)
    color_i += 1

# Plot the detection function
ax.plot(x_range, function)

# Plot the cosmic ray function
ax.axvline(0.3, label='Cosmic ray boundary', linestyle='--', color='purple')

# Wording
ax.update({'title':  f'Parameter space ratios for the NIRSpec dispenser-filter set-ups',
           'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}}$',
           'ylabel':  r'$\frac{A_{gas}}{\sigma_{noise}}$'})
ax.legend()
plt.show()


