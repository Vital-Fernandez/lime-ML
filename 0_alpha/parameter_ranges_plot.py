import numpy as np
from lime.model import gaussian_model
from matplotlib import pyplot as plt
import mplcursors

latex_dict = dict(amp=r'$A_g$', mu=r'$\mu$', sigma=r'$\sigma$',
                  lambda_step=r'$\Delta \lambda$', noise=r'$\sigma_{noise}$')


def normal_curve(amp, mu, sigma, lambda_step, noise):

    w_b, w_r = 2, 5
    w_lim = (-3 * sigma - w_b, 3 * sigma + w_r)

    rnd = np.random.RandomState()

    x = np.arange(w_lim[0], w_lim[1], lambda_step)
    cont = rnd.normal(0, noise, x.size)
    y = gaussian_model(x, amp, mu, sigma) + cont

    y_norm = y / np.max(y)

    return x, y_norm


def sn_calculation(amp, mu, sigma, lambda_step, noise):

    SN = (amp/noise) * np.sqrt(np.pi * sigma / 3)

    return SN


def grid_distribution(params_dict, **kwargs):

    param_j, param_i = params_dict.keys()
    j_array, i_array = params_dict[param_j], params_dict[param_i]

    fig, ax_array = plt.subplots(nrows=len(j_array), ncols=len(i_array), figsize=(len(i_array)*2, len(j_array)*2))

    for j, value_j in enumerate(j_array):
        for i, value_i in enumerate(i_array):

            ax = ax_array[j, i]

            # Gaussian curve
            gaussian_params = {param_j: value_j, param_i:  value_i, **kwargs}
            x_array, y_array = normal_curve(**gaussian_params)

            # Limits checks
            SN = sn_calculation(**gaussian_params)
            n_pix = (6.0 * gaussian_params['sigma']) / gaussian_params['lambda_step']
            Ag_Noise = gaussian_params['amp']/gaussian_params['noise']
            total_pix = x_array.size

            condition_1 = (SN >= 4) and (n_pix > 4.0)
            condition_2 = (SN >= 2.9) and (SN < 4) and (n_pix > 4.0) and (total_pix > 2 * n_pix)

            valid_line = condition_1 or condition_2

            # Plot
            label_curve = f'{latex_dict[param_i]} = {value_i:.2f}\n' \
                          f'{latex_dict[param_j]} = {value_j:.2f}\n' \
                          f'cond1 = {condition_1}, cond2 = {condition_2}\n' \
                          f'S/N = {SN:.2f}\n' \
                          f'Line width = {n_pix:.2f} pixels\n' \
                          f'Total pix = {total_pix:.2f}'

            ax.step(x_array, y_array, where='mid', label=label_curve)

            #Clear axis everywhere but edges
            if j == 0:
                ax.set_title(f'{latex_dict[param_i]} = {value_i:.2f}')

            if i == 0:
                ax.set_ylabel(f'{latex_dict[param_j]} = {value_j:.2f}')
            else:
                ax.axes.yaxis.set_visible(False)

            if (condition_1 == True) and (condition_2 == True):
                ax.set_facecolor('violet')

            else:
                if (condition_1 == True and condition_2 == False):
                    ax.set_facecolor('lightskyblue')

                elif (condition_2 == True and condition_1 == False):
                    ax.set_facecolor('palegreen')

                else:
                    ax.set_facecolor('xkcd:salmon')

            ax.yaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_major_locator(plt.NullLocator())


    mplcursors.cursor().connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))

    # Grid guides
    bbox_props = dict(boxstyle="rarrow, pad=0.3", fc="white", lw=1)
    arrow_props = dict(facecolor='black', shrink=0.05)

    # Columns arrow
    ax_0 = ax_array[0, 0]
    ax_0.annotate('', xy=(1.1 * len(i_array), 1.5), xycoords='axes fraction',
                    xytext=(0.2, 1.5), textcoords='axes fraction', arrowprops=arrow_props)
    ax_0.text(1.2 * len(i_array)/2, 1.7, latex_dict[param_i], ha="center", va="center", size=15, transform=ax_0.transAxes)


    # Rows arrow
    ax_0 = ax_array[0, 0]
    ax_0.annotate('', xy=(-0.3, -len(j_array)), xycoords='axes fraction',
                    xytext=(-0.3, 0.5), textcoords='axes fraction', arrowprops=arrow_props)
    ax_0.text(-0.5, -len(j_array)/2, latex_dict[param_j], ha="center", va="center", size=15, transform=ax_0.transAxes,
              rotation=90)

    plt.show()

    return


# First param for rows, second columns
for lambda_step in np.array([0.1, 0.2, 0.5, 0.75, 1, 1.2, 1.5, 2]):

    amp_array = np.array([0.1, 0.25, 0.75, 1, 2, 3, 4, 5, 10, 100, 1000])


    array_params = {'noise': np.linspace(0.05, 2, 5),
                    'amp':   amp_array}

    const_params = {'mu': 0,
                    'sigma': 1.0,
                    'lambda_step': lambda_step}


    # amp_array = np.array([0.1, 0.25, 0.50, 0.75, 1, 1.5,  2, 3, 4, 5, 10, 20, 100, 1000, 10000])
    #
    #
    # array_params = {'noise': np.linspace(0.05, 2, 10),
    #                 'amp':   amp_array}
    #
    # const_params = {'mu': 0,
    #                 'sigma': 1.0,
    #                 'lambda_step': lambda_step}


    print(f'Sigma_g = {const_params["sigma"]}, Delta_lambda = {const_params["lambda_step"]}, sigma/delta = {const_params["sigma"]/const_params["lambda_step"]}')
    grid_distribution(array_params, **const_params)


# # First param for rows, second columns
# array_params = {'sigma': np.linspace(0.03, 12.50, 8),
#                 'lambda_step':   np.linspace(0.01, 10, 20)}
#
# const_params = {'mu': 0,
#                 'amp': 2.0,
#                 'noise': 0.5}


# # First param for rows, second columns
# array_params = {'noise': [0.1, 0.25, 0.5, 0.75, 1.0],
#                 'amp':   [0.5, 1, 5, 7.5, 10]}
#
# const_params = {'mu': 0,
#                 'sigma': 1.5,
#                 'lambda_step': 0.25}

# # First param for rows, second columns
# array_params = {'lambda_step': np.linspace(0.1, 1.5, 8),
#                 'amp':   np.linspace(0.5, 10, 20)}
#
# const_params = {'mu': 0,
#                 'sigma': 1.0,
#                 'noise': 1.0}


# Plot the grid
# grid_distribution(array_params, **const_params)