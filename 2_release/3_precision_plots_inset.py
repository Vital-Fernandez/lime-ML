import numpy as np
import lime
import pandas as pd
from lime.model import gaussian_model
from lime.recognition import detection_function
from lime.plots import STANDARD_PLOT
from matplotlib import pyplot as plt, rc_context
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cfg_file = 'config_file.toml'
cfg = lime.load_cfg(cfg_file)
output_folder = Path(cfg['data_location']['output_folder'])
figures_folder = Path('D:\Dropbox\Astrophysics\Tools\LineMesurer')

amp_array = np.array(cfg['ml_grid_design']['amp_array'])
sigma_gas_array = np.array(cfg['ml_grid_design']['sigma_gas_um_array']) * 1000
delta_lam_array = np.array(cfg['ml_grid_design']['delta_lambda_um_array']) * 1000
noise_sig_array = np.array(cfg['ml_grid_design']['noise_array'])

line = 'H1_4861A'
mu_line = 4861.0
data_points = 400

df_values = lime.load_log(output_folder/'accuracy_table_v2.txt')
x_ratios = df_values.sigma.to_numpy()/df_values.delta.to_numpy()
y_ratios = df_values.amp.to_numpy()/df_values.noise.to_numpy()

idcs_failure = df_values.gauss == 'None'
idcs_detection = (~idcs_failure) & (x_ratios > 0.3)

# Swap None strings to nan
df_values.loc[idcs_failure, 'gauss'] = np.nan
df_values.loc[idcs_failure, 'gauss_err'] = np.nan
df_values['gauss'] = df_values['gauss'].astype(float)
df_values['gauss_err'] = df_values['gauss_err'].astype(float)


# Plot
x_detection = np.linspace(0.2, 20, 100)
y_detection = detection_function(x_detection)
factor = 1
bin_size = np.arange(-0.4, 0.4, step=0.04)

param_dict = {'intg' : df_values.intg_err.to_numpy()/df_values.intg.to_numpy(),
              'gauss': df_values.gauss_err.to_numpy()/df_values.gauss.to_numpy(),
              'true' : df_values.true_err.to_numpy()/df_values.flux_true.to_numpy()}

label_name = {'gauss': r'$\frac{\sigma_{Gaussian}}{F_{Gaussian}}$',
              'intg' : r'$\frac{\sigma_{intg}}{F_{intg}}$',
              'true' : r'$\frac{\sigma_{true}}{F_{true}}$'}

label_hist = {'gauss': 'Gaussian',
              'intg' : 'Integrated'}

STANDARD_PLOT.update({'axes.labelsize': 30, 'legend.fontsize': 15, 'figure.figsize': (10, 8),
                      'font.family': 'Times New Roman', 'mathtext.fontset':'cm'})

original_cmap = plt.cm.cubehelix
inverted_cmap = original_cmap.reversed()

for param_type, param_array in param_dict.items():

    with rc_context(STANDARD_PLOT):

        fig, ax = plt.subplots()
        axins = inset_axes(ax, width="45%", height="35%")

        # Gaussian measurements
        ax.plot(x_detection, y_detection, color='black', label='Line detection')

        ratio_scatter = ax.scatter(x_ratios[idcs_detection], y_ratios[idcs_detection], c=param_array[idcs_detection]*factor,
                                   cmap=inverted_cmap, edgecolor=None, vmin=0*factor, vmax=0.3)
        cbar = plt.colorbar(ratio_scatter, ax=ax)
        cbar.set_label(label_name[param_type], rotation=270, labelpad=70)

        # Cosmic ray limits
        ax.axvline(0.3, label='Cosmic ray boundary', linestyle='--', color='black')

        # Histogram data
        for param_type2, param_fluxes in param_dict.items():
            if param_type2 != 'true':
                array_true = param_dict['true']
                idcs_array_crop = (param_fluxes[idcs_detection] > 0.01) & (param_fluxes[idcs_detection] < 0.40)
                data_crop = param_fluxes[idcs_detection][idcs_array_crop] - array_true[idcs_detection][idcs_array_crop]
                print(param_type2, np.median(data_crop))
                axins.hist(data_crop, density=True, alpha=0.35, bins=bin_size, label=label_hist[param_type2])

        # Plot format
        ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}}$',
                   'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$'})

        ax.set_yscale('log')

        ax.legend(loc=4, framealpha=1)
        axins.legend(prop=dict(size=14), loc=6)

        axins.tick_params(labelleft=False)
        axins.set_yticks([])
        axins.set_xticks([-0.3, -0.15, 0, 0.15, 0.3])
        axins.set_xlabel(r'$\frac{\sigma_{fit}}{F_{fit}} - \frac{\sigma_{true}}{F_{true}}$', fontsize=20)

        plt.tight_layout()

        # plt.show()
        plt.savefig(figures_folder/f'{param_type}_coefficient_variation.png', dpi=400)


