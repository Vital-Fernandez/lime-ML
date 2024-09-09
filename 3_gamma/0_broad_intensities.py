import numpy as np
from lime.model import gaussian_model, gaussian_area, FWHM_FUNCTIONS
from lime.plots import theme
from matplotlib import rc_context, pyplot as plt


def narrow_factor_pred(narrow_broad_ratio):
    return np.sqrt(1 + np.log(narrow_broad_ratio)/np.log(2))


amp = 10
sigma_broad = 2
mu = 0
cont = 5
noise = 1

amp_tiers = [10, 100, 1000]
amp_broad = 5

res_ratio = 1
x_gauss = np.arange(-10, 10, res_ratio)
print(x_gauss)
noise_gauss = np.random.normal(0, noise, size=x_gauss.shape)

with rc_context(theme.fig_defaults()):

    fig, ax = plt.subplots()

    y_broad = gaussian_model(x_gauss, amp_broad, mu, sigma_broad) + cont + noise_gauss
    ax.step(x_gauss, y_broad, color='black')

    for amp in amp_tiers:

        # y_gauss = gaussian_model(x_gauss, amp, mu, sigma) + cont + noise_gauss
        sigma_narrow = sigma_broad / narrow_factor_pred(amp)
        y_narrow = gaussian_model(x_gauss, amp, mu, sigma_narrow) + cont + noise_gauss

        text_curve = r'$\frac{A}{\sigma_{noise}}=$' + r'${}$'.format(amp)
        text_curve = text_curve + r', $\sigma_{narrow}=$' + r'${:.2f}$'.format(sigma_narrow)
        ax.step(x_gauss, y_narrow, label=text_curve)

        # ax.text(0, (amp + cont) * 1.30, text_curve, fontsize=9, horizontalalignment='center')

        # noise_mark = sigma * np.sqrt(2 * np.log(amp/noise))
        # y_intesity_crit = np.exp(0.5 * np.power(sigma/noise, -2))
        # print(noise_mark, y_intesity_crit)
        # ax.axvline(noise_mark, color = gauss_plot[0].get_color(), linestyle=':')
        # for sigma_value in sigma_tiers:
        #     x_tier = np.array([-1*sigma_value, 1*sigma_value])
        #     sigma_tier = gaussian_model(x_tier, amp, mu, sigma) + cont
        #     ax.scatter(x_tier, sigma_tier, color='tab:orange')

    # y_gauss2 = gaussian_model(x_gauss, amp, mu, 0.2 *sigma) + cont + noise_gauss
    # ax.step(x_gauss, y_gauss2, label=f'extra', where='post')

    ax.axhline(cont, color='black', linestyle='--')
    # ax.set_xticks(np.arange(-4, 5, 1))
    # ax.grid(axis='x', color='0.95')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\sigma_{line}$')
    ax.set_ylabel(r'Line amplitude to continuum noise ratio $\left(\frac{A}{\sigma_{noise}}\right)$')
    ax.legend()
    plt.show()
