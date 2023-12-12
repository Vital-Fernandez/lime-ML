import numpy as np

SAMPLE_SIZE = 100000
SEED_VALUE = 1234
SAMPLE_LINE_PERCENTAGE = 50
PARMS_FILE = 'sample_parameter_table.csv'
MASK_MAX_SIZE = 750

from matplotlib import pyplot as plt, rcParams


def sn_calculation(amp, mu, sigma, lambda_step, noise):

    SN = (amp/noise) * np.sqrt(np.pi * sigma / 3)

    return SN


def plot_ratio_distribution(y_array, x_label=None, label=None, title=None, verbose=True):

    if verbose:
        fig, ax = plt.subplots()
        ax.hist(y_array, histtype='stepfilled', alpha=0.2, bins=1000, label=label)
        ax.update({'xlabel': x_label, 'ylabel': 'Variable value count', 'title': title})
        ax.legend(loc='upper center')
        plt.show()

    return


def plot_distribution(x_array, y_array, density_function, dist_label=None, title=None, x_label=None, verbose=True):

    if verbose:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(x_array, density_function, label=dist_label)
        ax.hist(y_array, density=True, histtype='stepfilled', alpha=0.2, bins=50)
        ax.legend(loc='lower center')
        ax.update({'xlabel': x_label, 'ylabel': 'Variable value count', 'title': title})
        plt.show()

    return