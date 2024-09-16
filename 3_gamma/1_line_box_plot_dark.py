import lime
import numpy as np
from lime.plots import theme
from matplotlib import rc_context, pyplot as plt, rc, cm, colors
from lime.model import gaussian_model, gaussian_area, FWHM_FUNCTIONS
from pathlib import Path

def FWHM_conv(sigma_km):

    const = 2 * np.sqrt(2 * np.log(2))

    return sigma_km * const

def delta_inst_conv(R, c_light):

    return c_light / R

def b_pixel_formula(k_const, R_line, sigma_line, c_light):

    return (k_const * sigma_line * R_line) / (c_light)

def R_formula(k_const, b_line, sigma_line, c_light):

    return (b_line * c_light) / (k_const * sigma_line)

def deltaLamb_formula(k_const, b_line, sigma_line):

    return (k_const * sigma_line)


k_line = 2 * 3
c_KMpS = 299792.458
b_range = np.arange(0, 49, 1)
pixel_range = np.arange(0, 49, 6)
# sigma_range = np.array([30, 70, 300, 850])
sigma_lines = np.array([30, 70, 300, 850])
FWHM_range = FWHM_conv(sigma_lines)

# Gaussian plot
noise = 1
sigma_shape = 1
mu = 0
cont = 5
amp_tiers = [10, 100, 1000, 5000]
x_gauss = np.arange(-8, 8, 1)
noise_gauss = np.random.normal(0, noise, size=x_gauss.shape)

# Format
font_sigmas = 6
b_coord = 24
colors_ranges = ['forestgreen', '#ffdb68', 'salmon']
hatch_ranges = ['\\\\', 'O', '//']
text_sigma_ranges = ['HII galaxies\n' + r'$\left(L(H\beta) \propto \sigma_{H\beta}\right)$',
                     'Green peas\n' + 'multi-component line',
                     'Seyferd\n' + 'broad line',]

# ax1.axvspan(30, 1500, alpha=0.5, color=) # NIRSPEC DESI VIS
# ax1.axvspan(2000, 5000, alpha=0.5, color='salmon') # DESI VIS
# ax1.axvspan(5400, 17400, alpha=0.5, color=)

# Create ranges velocity regions
# ranges_dict = {}
# for i, sigma in enumerate(sigma_range):
#     R_range = R_formula(k_line, b_range, sigma, c_KMpS)
#     ranges_dict[sigma] = [np.array(b_range), np.array(R_range)]

cmap = cm.get_cmap('Oranges')
norm = colors.Normalize(vmin=0, vmax=3)

output_folder=Path('/home/vital/Dropbox/Astrophysics/Seminars/BootCamp2025')
theme.set_style('dark')
theme.colors['fg']
lines_dict = {}
for i, sigma in enumerate(sigma_lines):
    R_range = R_formula(k_line, b_range, sigma, c_KMpS)
    lines_dict[sigma] = [np.array(b_range), np.array(R_range)]

fig_default = theme.fig_defaults()
fig_default['figure.figsize'] = (8, 4)
fig_default['figure.dpi'] = 2000

with rc_context(fig_default):

    fig, ax1 = plt.subplots()

    # # Shaded velocity intervals
    # for i, sigma in enumerate(sigma_range):
    #
    #     b_range, R_range = ranges_dict[sigma]
    #     ax1.plot(R_range, b_range, color='black', linewidth=0.5, linestyle='--')

        # # Shaded area
        # if i > 0:
        #     ax1.fill_betweenx(b_range, R_range, ranges_dict[sigma_range[i-1]][1], alpha=0.5, color='none')

    # Velocity regions intervals
    for i, items in enumerate(lines_dict.items()):

        sigma, values = items
        b_range, R_range = values

        ax1.plot(R_range, b_range, color=theme.colors['fg'], linewidth=0.5, linestyle='--')

        # Coordiantes text:
        R_text = R_formula(k_line, b_coord, sigma, c_KMpS)

        # Angle text
        angle_data = np.rad2deg(np.arctan2(b_range[-1] - b_range[0], R_range[-1] - R_range[0]))
        angle_screen = ax1.transData.transform_angles(np.array((angle_data,)), np.array([R_range[0], b_range[0]]).reshape((1, 2)))[0]

        FWHM = FWHM_conv(sigma)
        text = f'$\sigma = {sigma}\,km/s, FWHM = {FWHM:.0f}\,km/s$'
        ax1.text(R_text-500, b_coord, text, fontsize=font_sigmas, rotation=angle_data, rotation_mode='anchor',
                 transform_rotates_text=True, color=theme.colors['fg'], weight='bold')

    # y_lims
    # y_lims = ax1.get_ylim()
    y_lims = (0, 48)

    vertical_range = np.arange(-10, 60, 1)
    ax1.fill_betweenx(vertical_range, 30, 1500, alpha=0.5, color=colors_ranges[2], edgecolor='none')
    ax1.fill_betweenx(vertical_range, 2000, 5000, alpha=0.5, color=colors_ranges[1], edgecolor='none')
    ax1.fill_betweenx(vertical_range, 5400, 17400, alpha=0.5, color=colors_ranges[0], edgecolor='none')
    # ax1.axvspan(30, 1500, alpha=0.5, color=colors_ranges[2]) # NIRSPEC DESI VIS
    # ax1.axvspan(2000, 5000, alpha=0.5, color=colors_ranges[1]) # DESI VIS
    # ax1.axvspan(5400, 17400, alpha=0.5, color=colors_ranges[0])  # XSHOOTER VIS

    # Phenomena arrows
    ax1.annotate('', xy=(2400, 1.02), xytext=(8500,  1.02), xycoords=('data', 'axes fraction'),
                 arrowprops=dict(arrowstyle='<->', color=theme.colors['fg'], lw=1))
    ax1.annotate('', xy=(7600, 1.02), xytext=(34800,  1.02), xycoords=('data', 'axes fraction'),
                 arrowprops=dict(arrowstyle='<->', color=theme.colors['fg'], lw=1))
    ax1.annotate('', xy=(33900, 1.02), xytext=(80500,  1.02), xycoords=('data', 'axes fraction'),
                 arrowprops=dict(arrowstyle='<->', color=theme.colors['fg'], lw=1))

    ax1.annotate(text_sigma_ranges[0], xy=(57200, 1.04), xytext=(52000,  1.04), xycoords=('data', 'axes fraction'),
                 fontsize=7)
    ax1.annotate(text_sigma_ranges[1], xy=(57200, 1.04), xytext=(20000,  1.04), xycoords=('data', 'axes fraction'),
                 fontsize=7, horizontalalignment='center')

    ax1.annotate(text_sigma_ranges[2], xy=(57200, 1.04), xytext=(5300,  1.04), xycoords=('data', 'axes fraction'),
                 fontsize=7, horizontalalignment='center')

    # Set new ticks
    ax1.set_ylim(y_lims)
    ax1.set_yticks(pixel_range)
    # ticks_lims = ax1.get_ylim()
    ticks_values = ax1.get_yticks()
    ticks_labels = [f'{tick:.1f}' for tick in pixel_range]

    # delta_inst_range = delta_inst_conv(np.array(ticks_values), c_KMpS)
    ax2 = ax1.twinx()
    ticks_labels = [f'{tick:.0f}' for tick in ticks_values/6]
    ax2.set_ylim(np.array(y_lims)/6)
    ax2.set_yticks(ticks_values/6)
    ax2.set_yticklabels(ticks_labels)


    x_label = r'Resolving power, $R = \frac{\lambda}{\Delta \lambda}$'

    ax1.set_xlabel(x_label, x=0.25)
    ax1.text(35000, -5.95, 'NIRSPEC PRISM-G293M,', color='salmon', fontsize=8)
    ax1.text(54500, -5.95, 'SLOAN-DESI,', color='#ffdb68', fontsize=8)
    ax1.text(65000, -5.95, 'XSHOOTER', color='forestgreen', fontsize=8)

    # loc = 'left', labelpad = 10)
    #
    ax1.set_ylabel(r'$b_{pixels}$ (detection box width in pixels)')
    ax2.set_ylabel('$\sigma_{pixels}$ (Gaussian sigma in pixels)')

    # Formula
    formula = r'$b_{pixels} =  n_{\sigma} \cdot \sigma_{pixels} = n_{\sigma} \, \frac{\sigma_{line}}{c} R$'
    ax1.text(0.30, 0.08, formula, transform=ax1.transAxes, fontsize=9)

    # Inset axis
    range_sigma_pixels = np.arange(-4, 5, 1)
    axin_ticks_labels = [r'{}$\sigma$'.format(tick) for tick in range_sigma_pixels]
    axin_ticks_labels[4] = '0'
    axins = ax1.inset_axes([0.65, 0.08, 0.30, 0.50], xticklabels=axin_ticks_labels, yticklabels=[], transform=ax1.transAxes)
    for idx_amp, amp in enumerate(amp_tiers):
        y_gauss = gaussian_model(x_gauss, amp, 0, sigma_shape) + cont + noise_gauss
        axins.step(x_gauss, y_gauss, where='mid', color=cmap(norm(idx_amp)))
    axins.axhline(cont, color=theme.colors['fg'], linestyle='--')

        # text_curve = r'$\frac{A}{\sigma_{noise}}=$' + r'${}$'.format(amp)
        # axins.text(0, (amp + cont) * 1.30, text_curve, fontsize=4, horizontalalignment='center')

    # Pointing arrow
    axins.annotate('', xy=(-3.25, 7000), xytext=(3.25, 7000), arrowprops=dict(arrowstyle='<->', color=theme.colors['fg'], lw=1))
    axins.text((-3.25 + 3.25) / 2, 7000-1200, r'$n_{\sigma} = 6$', ha='center', va='bottom', fontsize=5, color=theme.colors['fg'],
               bbox=dict(facecolor=theme.colors['bg'], alpha=1, edgecolor='none', boxstyle='round,pad=0.01'))
    axins.set_yscale('log')
    axins.set_xticks(range_sigma_pixels)
    axins.set_xticklabels(axin_ticks_labels)
    axins.grid(axis='x', color=theme.colors['fg'], linewidth=0.1)
    axins.set_ylim(bottom=0, top=10000)
    axins.tick_params(axis='y', labelsize=5)
    axins.tick_params(axis='x', labelsize=5)
    axins.set_ylabel(r'$\frac{A_{line}}{\sigma_{noise}}$')
    # axins.set_xlabel(r'$\mu (\sigma_{line})$')


    # axins.set_yticklabels(axins.get_yticks(), fontsize=5)

    ax1.grid(axis='x', color=theme.colors['fg'], linewidth=0.25)
    ax1.grid(axis='y', color=theme.colors['fg'], linewidth=0.25)
    # ax1.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(output_folder/'box_selector.png')


# delta_lamb = deltaLamb_formula(k_line, b_range, sigma)
# ax1.plot(R_range, b_range, label=f'$\sigma = {sigma} km/s$')
# ax2.plot(delta_lamb, b_range, linestyle=':')