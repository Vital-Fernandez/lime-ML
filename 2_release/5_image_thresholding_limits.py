import numpy as np
import pandas as pd
from pathlib import Path
import lime
from lime.model import gaussian_model
from lime.plots import STANDARD_PLOT
from matplotlib import pyplot as plt, rc_context

STANDARD_PLOT.update({'axes.labelsize': 30, 'legend.fontsize': 20, 'figure.figsize': (8, 8)})

base = 10000
inst_ratio_break = 100
inst_ratio_break_log = np.emath.logn(base, inst_ratio_break)

x1 = np.arange(1, 27, 1)
range_log1 = np.logspace(0, 0.5, x1.size, base=10000)

x2 = np.arange(27, 34, 1)
range_log2 = np.logspace(0.55, 1, x2.size, base=10000)

a1, a2 =  np.emath.logn(base, range_log1),  np.emath.logn(base, range_log2)
trans_array = np.round(np.hstack((a1, a2)), 6)
print(list(trans_array))
print(np.power(base, trans_array))

with rc_context(STANDARD_PLOT):

    fig, ax = plt.subplots()
    # ax.scatter(a1, x1)
    # ax.scatter(a2, x2)

    ax.scatter(np.power(base, a1), x1)
    ax.scatter(np.power(base, a2), x2)

    ax.update({'xlabel': r'$\frac{A_{gas}}{\sigma_{noise}}$', 'ylabel': r'Number of pixels'})
    ax.set_xscale('log')
    plt.show()




# import numpy as np
#
# # Assuming X is your original array of length 100
# X = np.arange(100)
#
# # Calculate the number of rows to include up to the last digit of X
# num_rows = len(X) - 10 + 1  # Adjust as needed
#
# # Define the number of columns for the matrix Y
# num_columns = 11  # Adjust as needed
#
# # Create the matrix Y using fancy indexing
# Y = X[np.arange(num_rows)[:, np.newaxis] + np.arange(num_columns) <= len(X) - 1]
#
# # Print the resulting matrix Y
# print(Y)