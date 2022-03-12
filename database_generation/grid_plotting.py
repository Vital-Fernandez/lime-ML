import numpy as np
from lime.model import gaussian_model
from matplotlib import pyplot as plt


def normal_curve(amp, mu, sigma, lambda_step, noise):

    w_b, w_r = 2, 5
    w_lim = (-3 * sigma - w_b, 3 * sigma + w_r)

    rnd = np.random.RandomState()

    x = np.arange(w_lim[0], w_lim[1], lambda_step)
    cont = rnd.normal(0, noise, x.size)
    y = gaussian_model(x, amp, mu, sigma) + cont

    y_norm = y / np.max(y)

    return x, y_norm


A_g = 5
delta_lam = 0.25
sigma_g = 1.5
noise = 0.25

# x_array, y_array = normal_curve(A_g, 0, sigma_g, delta_lam, noise)
# x_array2, y_array2 = normal_curve(A_g, 0, 3, delta_lam, noise)
#
# fig, ax = plt.subplots()
# ax.step(x_array, y_array, where='mid')
# ax.step(x_array2, y_array2, where='mid')
# plt.show()


n_rows, n_columns = 5, 5

params_dict = {'noise': [0.1, 0.25, 0.5, 0.75],
               'A_g': [0.5, 1, 5, 7.5, 10]}


param_j, param_i = params_dict.keys()
j_array, i_array = params_dict[param_j], params_dict[param_i]

fig, ax_array = plt.subplots(nrows=len(j_array), ncols=len(i_array), figsize=(12, 12))

for j, value_j in enumerate(j_array):
    for i, value_i in enumerate(i_array):

        x_array, y_array = normal_curve(value_i, 0, sigma_g, delta_lam, value_j)

        ax = ax_array[j, i]
        ax.step(x_array, y_array, where='mid')

        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.axes.yaxis.set_visible(False)

        ax.set_title(f'{param_j} = {value_j}, {param_i} = {value_i}')

plt.show()
