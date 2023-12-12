import numpy as np
import lime
from pathlib import Path
from astropy.io import fits
from matplotlib import pyplot as plt, rcParams, cm
from lime.recognition import detection_function
from lime.plots import STANDARD_PLOT

c_kmpers = 299792.458  # Km/s

STANDARD_PLOT['axes.labelsize'] = 20

rcParams.update(STANDARD_PLOT)

cfg = lime.load_cfg('../2_release/config_file.toml')
fits_folder = Path(cfg['data_location']['fits_folder'])
output_folder = Path(cfg['data_location']['fits_folder'])



# Parameter settings
ml_design_params = cfg['ml_grid_design']

# S/N evolution with the spectrum and emission parameters
amp_array = np.array([1, 2, 3, 4, 5, 10])
noise_array = np.ones(amp_array.size)
sigma_gas_array = np.linspace(0.03, 12.50, 100)

example_dispensers = ['MIKE_BLUE', 'MIKE_RED',
                      'XSHOOTER_UVB', 'XSHOOTER_VIS', 'XSHOOTER_NIR',
                      'NIRSPEC_PRISM', 'NIRSPEC_G235M']

limits_dict = {}
limits_dict[f'MIKE_BLUE_wave'] = np.array([3200.0, 5000.0])
limits_dict[f'MIKE_RED_wave'] = np.array([4900.0, 10000.0])
limits_dict[f'MIKE_BLUE_R'] = np.array([22000.0, 28000.0])
limits_dict[f'MIKE_RED_R'] = np.array([65000.0, 83000.0])

limits_dict[f'XSHOOTER_UVB_wave'] = np.array([293.6, 593.0]) * 10
limits_dict[f'XSHOOTER_VIS_wave'] = np.array([525.3, 1048.9]) * 10
limits_dict[f'XSHOOTER_NIR_wave'] = np.array([982.7, 2480.7]) * 10

limits_dict[f'XSHOOTER_UVB_R'] = np.array([3300, 9100])
limits_dict[f'XSHOOTER_VIS_R'] = np.array([5400, 17400])
limits_dict[f'XSHOOTER_NIR_R'] = np.array([3500, 11300])

limits_dict[f'NIRSPEC_PRISM_wave'] = np.array([0.5, 6.0]) * 10000
limits_dict[f'NIRSPEC_G235M_wave'] = np.array([1.5, 3.5]) * 10000
limits_dict[f'NIRSPEC_PRISM_R'] = np.array([30.0, 422.0])
limits_dict[f'NIRSPEC_G235M_R'] = np.array([636.0, 1500.0])


# Plot sigma_gas/delta_lambda limits per instrument
x_range = np.linspace(0.2, 20, 50)
sigma_range = np.linspace(50, 120, 100)
function = detection_function(x_range)

fig, ax = plt.subplots(figsize=(12, 12))
cmap, color_i = cm.get_cmap(), 0

for dispenser in example_dispensers:
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


