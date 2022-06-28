import numpy as np
from matplotlib import pyplot as plt, cm, rcParams


def sn_formula(A_g, noise, sigma_g):

    SN = (A_g/noise) * np.sqrt(np.pi * sigma_g / 3)

    return SN


def sn_evolution_plot(A_g_array, noise_array, sigma_g_array, SN_limit=3, wavelength_ref = None, c_speed = 299792.458):

    STANDARD_PLOT = {'figure.figsize': (10, 8),
                     'axes.titlesize': 16,
                     'axes.labelsize': 16,
                     'legend.fontsize': 14,
                     'xtick.labelsize': 14,
                     'ytick.labelsize': 14}

    rcParams.update(STANDARD_PLOT)

    cmap = cm.get_cmap()

    fig, ax = plt.subplots()

    for i, Ag_noise in enumerate(A_g_array):
        SN_curve = sn_formula(Ag_noise, noise=noise_array[i], sigma_g=sigma_g_array)

        label = r'$\frac{A_{g}}{\sigma_{noise}}$'
        color_curve = cmap(i / len(A_g_array))

        if wavelength_ref is None:
            x_array = sigma_g_array
            x_label = r'$\sigma_{gas} (\AA)$'
        else:
            x_array = c_speed * sigma_g_array/wavelength_ref
            x_label = r'$\sigma_{gas} (km/s)$' + f' at ({wavelength_ref}$\AA$)'

        ax.plot(x_array, SN_curve, label=f'{label} = {Ag_noise}', color=color_curve)

    ax.axhline(y=SN_limit, color='black', linestyle='--', label=f'S/N = {SN_limit}')

    ax.set_yscale('log')
    ax.update({'xlabel': x_label, 'ylabel': r'$\frac{S}{N}$'})
    ax.legend(ncol=4, loc=4)
    plt.tight_layout()
    plt.show()
    # ax.set_xlim(-10, 210)
    # plt.savefig(f'/home/vital/Dropbox/Astrophysics/Tools/LineMesurer/SN_plot_line_angstroms.png')

    return


A_g_noise_coeff = np.array([1, 2, 3, 4, 5, 10])
noise_array = np.ones(A_g_noise_coeff.size)
sigma_gas_array = np.linspace(0.03, 12.50, 100)

wavelength = None
sn_evolution_plot(A_g_noise_coeff, noise_array, sigma_gas_array, wavelength_ref=wavelength)


# Ag_noise_ratio = np.linspace(0.5, 100, 1000)
# sigma_g = np.linspace(0.03, 12.50, 1000)
#
# fig, ax = plt.subplots()
# ax.plot(sigma_g, Ag_noise_ratio)
# ax.legend()
# ax.update({'xlabel': r'$\sigma_{gas}$', 'ylabel': r'$\frac{A_{g}}{\sigma_{noise}}$', 'title': r'$\frac{A_{g}}{\sigma_{noise}}$ for S/N = 1'})
# plt.show()