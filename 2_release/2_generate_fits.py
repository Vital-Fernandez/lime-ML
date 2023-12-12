import numpy as np
import lime
import pandas as pd
from lime.model import gaussian_model
from lime.recognition import detection_function
from lime.plots import STANDARD_PLOT
from matplotlib import pyplot as plt, rc_context
from pathlib import Path

cfg_file = 'config_file.toml'
cfg = lime.load_cfg(cfg_file)
output_folder = Path(cfg['data_location']['output_folder'])

amp_array = np.array(cfg['ml_grid_design']['amp_array'])
sigma_gas_array = np.array(cfg['ml_grid_design']['sigma_gas_um_array']) * 1000
delta_lam_array = np.array(cfg['ml_grid_design']['delta_lambda_um_array']) * 1000
noise_sig_array = np.array(cfg['ml_grid_design']['noise_array'])

line = 'H1_4861A'
mu_line = 4861.0
data_points = 400
width_factor = 4

df_values = pd.DataFrame(columns=['amp', 'sigma', 'noise', 'delta', 'intg', 'intg_err', 'gauss', 'gauss_err', 'flux_true',
                                  'detection', 'std_cont', 'true_err'])
i = 0
for amp in amp_array:
    for sigma in sigma_gas_array:
        for noise in noise_sig_array:
            for inst_delta in delta_lam_array:

                w0 = mu_line - inst_delta * data_points / 2
                wf = mu_line + inst_delta * data_points / 2

                wave = np.arange(w0, wf, inst_delta)
                flux = gaussian_model(wave, amp, mu_line, sigma) + np.random.normal(0, noise, data_points) + 10

                int_ratio, res_ratio = amp/noise, sigma/inst_delta
                detection = True if int_ratio >= detection_function(res_ratio) else False

                if (res_ratio < 20) and (amp/noise > 1) and (amp/noise < 1e5):
                    # df_values.loc[i, :] = amp, sigma, noise, inst_delta, None, None, None, None, None, detection

                    spec = lime.Spectrum(wave, flux, redshift=0)

                    w3, w4 = mu_line - width_factor * sigma, mu_line + width_factor * sigma
                    idcs_bands = np.searchsorted(wave, ([wave[10], wave[20], w3, w4, wave[-20], wave[-10]]))
                    bands = wave[idcs_bands]

                    spec.fit.bands(line, bands)

                    intg_flux, intg_err, std_cont = spec.log.loc[line, ['intg_flux', 'intg_flux_err', 'std_cont']].to_numpy()
                    theo_flux = amp * 2.5066282746 * sigma
                    true_error = noise * np.sqrt(2 * width_factor * inst_delta * sigma)

                    if spec.fit.line.observations == 'no':
                        gauss_flux, gauss_err = spec.log.loc[line, ['gauss_flux', 'gauss_flux_err']].to_numpy()

                    else:
                        gauss_flux, gauss_err = None, None

                    df_values.loc[i, :] = (amp, sigma, noise, inst_delta, intg_flux, intg_err, gauss_flux, gauss_err,
                                           theo_flux, detection, std_cont, true_error)

                    i += 1

lime.save_log(df_values, output_folder/'accuracy_table_v2.txt')

                # if detection:
                #     spec = lime.Spectrum(wave, flux, redshift=0)
                #     # spec.plot.spectrum()
                #
                #     w3, w4 = mu_line - 5 * sigma, mu_line + 5 * sigma,
                #     idcs_bands = np.searchsorted(wave, ([wave[10], wave[20], w3, w4, wave[-20], wave[-10]]))
                #     # print(i, inst_delta, f'n_points = {wave.size}', idcs_bands)
                #
                #     bands = wave[idcs_bands]
                #     spec.fit.bands(line, bands)
                #
                #     intg_flux, intg_err = spec.log.loc[line, ['intg_flux', 'intg_flux_err']].to_numpy()
                #     gauss_flux, gauss_err = spec.log.loc[line, ['gauss_flux', 'gauss_flux_err']].to_numpy()
                #     theo_flux = amp * 2.5066282746 * sigma
                #
                #     # Save the data
                #     df_values.loc[i, :] = amp, sigma, noise, inst_delta, intg_flux, intg_err, gauss_flux, gauss_err, theo_flux, detection
                #
                #     i += 1

print(df_values)

# w0_array = mu_line - delta_lam_array * data_points/2
# wf_array = mu_line + delta_lam_array * data_points/2

# for i, inst_delta in enumerate(delta_lam_array):
#
#     amp, sigma, noise = amp_array[i], sigma_gas_array[i], noise_sig_array[i]
#     wave = np.arange(w0_array[i], wf_array[i], inst_delta)
#     flux = gaussian_model(wave, amp_array[i], mu_line, sigma_gas_array[i]) + np.random.normal(0, noise, data_points) + 10
#
#     spec = lime.Spectrum(wave, flux, redshift=0)
#     # spec.plot.spectrum()
#
#     w3, w4 = mu_line - 5 * sigma, mu_line + 5 * sigma,
#     idcs_bands = np.searchsorted(wave, ([wave[10], wave[20], w3, w4, wave[-20], wave[-10]]))
#     # print(i, inst_delta, f'n_points = {wave.size}', idcs_bands)
#
#     bands = wave[idcs_bands]
#     spec.fit.bands(line, bands)
#
#     intg_flux, intg_err = spec.log.loc[line, ['intg_flux', 'intg_flux_err']].to_numpy()
#     gauss_flux, gauss_err = spec.log.loc[line, ['gauss_flux', 'gauss_flux_err']].to_numpy()
#     theo_flux = amp * 2.5066282746 * sigma
#     detection = True if amp/noise >= detection_function(sigma/inst_delta) else False
#
#     # Save the data
#     df_values.loc[i, :] = amp, sigma, noise, inst_delta, intg_flux, intg_err, gauss_flux, gauss_err, theo_flux, detection
#
#     # intg_diff, err_intg_diff = 100 * (intg_flux/theo_flux - 1), 100 * (intg_err/theo_flux - 1)
#     # gauss_diff, err_gauss_diff = 100 * (gauss_flux/theo_flux - 1), 100 * (gauss_err/theo_flux - 1)
#     #
#     # chi_intg = (intg_flux - theo_flux) / intg_err
#     # chi_gauss = (gauss_flux - theo_flux) / gauss_err
#     #
#     # print(f'Difference percentage ({theo_flux:0.2f})) Intg {intg_diff:0.2f}% ({intg_flux:0.4f}+/-{intg_err:0.4f}); Gauss {gauss_diff:0.2f}%')
#     # print(f'Difference Chi) Intg {chi_intg:0.3f}; Gauss {chi_gauss:0.2f}; Detection {detection}')
#     # print()
#     # spec.plot.bands()
#
# print(df_values)


# Plot
x_detection = np.linspace(0.2, 20, 100)
y_detection = detection_function(x_detection)

x_ratios = df_values.sigma.to_numpy()/df_values.delta.to_numpy()
y_ratios = df_values.amp.to_numpy()/df_values.noise.to_numpy()
# chi_intg = (df_values.intg.to_numpy() - df_values.flux_true.to_numpy()) / df_values.intg_err.to_numpy()
# chi_gauss = (df_values.gauss.to_numpy() - df_values.flux_true.to_numpy()) / df_values.gauss_err.to_numpy()

STANDARD_PLOT.update({'axes.labelsize': 30, 'legend.fontsize': 20, 'figure.figsize': (8, 8)})


with rc_context(STANDARD_PLOT):

    fig, ax = plt.subplots()

    ax.axvline(0.3, label='Cosmic ray boundary', linestyle='--', color='purple')

    ax.plot(x_detection, y_detection, color='black', label='Detection boundary')

    ratio_scatter = ax.scatter(x_ratios, y_ratios, cmap='RdBu', edgecolor=None)

    # ratio_scatter = ax.scatter(x_ratios, y_ratios, c=chi_gauss, cmap='RdBu', edgecolor=None, vmin=-4, vmax=4)
    # cbar = plt.colorbar(ratio_scatter, ax=ax)

    ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}}$',
               'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$'})

    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()


