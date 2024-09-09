import numpy as np
from lime.model import gaussian_model
from matplotlib import pyplot as plt
import mplcursors

latex_dict = dict(amp=r'$A_g$', mu=r'$\mu$', sigma=r'$\sigma$',
                  lambda_step=r'$\Delda \lambda$', noise=r'$\sigma_{noise}$')


def normal_curve(amp, mu, sigma, lambda_step, noise):

    w_b, w_r = 2, 5
    w_lim = (-3 * sigma - w_b, 3 * sigma + w_r)

    rnd = np.random.RandomState()

    x = np.arange(w_lim[0], w_lim[1], lambda_step)
    cont = rnd.normal(0, noise, x.size)
    y = gaussian_model(x, amp, mu, sigma) + cont

    y_norm = y / np.max(y)

    return x, y_norm


def grid_distribution(params_dict, **kwargs):

    param_j, param_i = params_dict.keys()
    j_array, i_array = params_dict[param_j], params_dict[param_i]

    print(len(i_array))

    fig, ax_array = plt.subplots(nrows=len(j_array), ncols=len(i_array), figsize=(len(i_array)*2, len(j_array)*2))

    for j, value_j in enumerate(j_array):
        for i, value_i in enumerate(i_array):

            ax = ax_array[j, i]

            gaussian_params = {param_j: value_j, param_i:  value_i, **kwargs}
            x_array, y_array = normal_curve(**gaussian_params)

            label_curve = f'{latex_dict[param_i]} = {value_i}\n{latex_dict[param_j]} = {value_j}'
            ax.step(x_array, y_array, where='mid', label=label_curve)

            #Clear axis everywhere but edges
            if j == 0:
                ax.set_title(f'{latex_dict[param_i]} = {value_i}')

            if i == 0:
                ax.set_ylabel(f'{latex_dict[param_j]} = {value_j}')
            else:
                ax.axes.yaxis.set_visible(False)

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
array_params = {'noise': [0.1, 0.25, 0.5, 0.75, 1.0, 1.2],
                'amp':   [0.5, 1, 2.5, 5.0, 7.5, 10.0]}

const_params = {'mu': 0,
                'sigma': 1.5,
                'lambda_step': 0.25}

grid_distribution(array_params, **const_params)



# # First param for rows, second columns
# array_params = {'noise': [0.1, 0.25, 0.5, 0.75, 1.0],
#                 'amp':   [0.5, 1, 5, 7.5, 10]}
#
# const_params = {'mu': 0,
#                 'sigma': 1.5,
#                 'lambda_step': 0.25}
#
# grid_distribution(array_params, **const_params)