import numpy as np
import lime
import pandas as pd
from lime.model import gaussian_model
from lime.recognition import detection_function
from lime.plots import STANDARD_PLOT
from matplotlib import pyplot as plt, rc_context
from pathlib import Path

cfg_file = '../../3_gamma/training_sample_v3.toml'
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
                                  'detection', 'std_cont'])
i = 0
amp_array = [0.01, 0.25, 0.75, 1, 2, 3, 4, 5, 10, 100, 1000]

import specutils
from specutils import Spectrum1D
from specutils.fitting import fit_lines
import astropy.units as u
from astropy.io import fits

for amp in amp_array:
    for noise in noise_sig_array:
        for inst_delta in delta_lam_array:
            for sigma in sigma_gas_array:

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

                    if spec.fit.line.observations == 'no':
                        gauss_flux, gauss_err = spec.log.loc[line, ['gauss_flux', 'gauss_flux_err']].to_numpy()

                    else:
                        gauss_flux, gauss_err = np.nan, np.nan

                    df_values.loc[i, :] = amp, sigma, noise, inst_delta, intg_flux, intg_err, gauss_flux, gauss_err, theo_flux, detection, std_cont

                    true_error = noise * np.sqrt(2 * width_factor * inst_delta * sigma)

                    print(amp/noise)
                    print(f'Intg:   {intg_flux:0.5f} +/- {intg_err:0.5f}    => {(intg_flux/theo_flux -1) * 100:0.2f} +/- {(intg_err/theo_flux) * 100:0.2f}  %')
                    print(f'Gauss:  {gauss_flux:0.5f} +/- {gauss_err:0.5f}  => {(gauss_flux/theo_flux -1) * 100:0.2f} +/- {(gauss_err/theo_flux)* 100:0.2f}  %')
                    print(f'True:   {theo_flux:0.5f} +/- {true_error:0.5f}  => {(theo_flux/theo_flux -1) * 100:0.2f} +/- {(true_error/theo_flux) * 100:0.2f}  %')
                    print()

                    spec.plot.bands()

                    i += 1

