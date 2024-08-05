import numpy as np
from lime.model import gaussian_model, gaussian_area, FWHM_FUNCTIONS
from lime.plots import theme
from matplotlib import rc_context, pyplot as plt

amp = 10
sigma = 1
mu = 0
cont = 0
noise = 1

amp_tiers = [1.5, 1.7, 7, 7.5]
sigma_tiers = [1, 2, 3, 4]

res_ratio = 1
x_gauss = np.arange(-10, 10, res_ratio)
print(x_gauss)
noise_gauss = np.random.normal(0, noise, size=x_gauss.shape)

with rc_context(theme.fig_defaults()):

    fig, ax = plt.subplots()

    for amp in amp_tiers:

        y_gauss = gaussian_model(x_gauss, amp, mu, sigma) + cont + noise_gauss

        gauss_plot = ax.step(x_gauss, y_gauss, label=f'Amp/noise = {amp}', where='post')

        text_curve = r'$\frac{A}{\sigma_{noise}}=$' + r'${}$'.format(amp)
        ax.text(0, (amp + cont) * 1.30, text_curve, fontsize=9, horizontalalignment='center')

        noise_mark = sigma * np.sqrt(2 * np.log(amp/noise))
        y_intesity_crit = np.exp(0.5 * np.power(sigma/noise, -2))
        print(noise_mark, y_intesity_crit)
        ax.axvline(noise_mark, color = gauss_plot[0].get_color(), linestyle=':')
        # for sigma_value in sigma_tiers:
        #     x_tier = np.array([-1*sigma_value, 1*sigma_value])
        #     sigma_tier = gaussian_model(x_tier, amp, mu, sigma) + cont
        #     ax.scatter(x_tier, sigma_tier, color='tab:orange')

    # y_gauss2 = gaussian_model(x_gauss, amp, mu, 0.2 *sigma) + cont + noise_gauss
    # ax.step(x_gauss, y_gauss2, label=f'extra', where='post')

    ax.axhline(cont, color='black', linestyle='--')
    ax.set_xticks(np.arange(-4, 5, 1))
    ax.grid(axis='x', color='0.95')
    ax.set_yscale('linear')
    ax.set_ylim(bottom=0, top=100)
    ax.set_xlabel(r'$\sigma_{line}$')
    ax.set_ylabel(r'Line amplitude to continuum noise ratio $\left(\frac{A}{\sigma_{noise}}\right)$')
    # ax.legend()
    plt.show()
