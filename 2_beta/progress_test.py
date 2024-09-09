# import numpy as np
# from scipy.signal import convolve2d
#
# AA = np.array([[False, True, False, False, False, True, False, False, False, False, False, True, False],
#                [False, True, False, False, False, True, False, False, False, False, False, True, False]])
#
# propagation = 2
#
# # generate 2D kernel
# k = np.r_[np.zeros(propagation), np.ones(propagation+1)][None]
#
# out = convolve2d(AA, k, mode='same').astype(bool)
#
# print(out)

import numpy as np

# Number of positions to propagate the array
propagation = 2

# Original array
AA = np.array([[False, True, False, False, False, True, False, False, False, False, False, True, False],
             [False, True, False, False, False, True, False, False, False, False, False, True, False]])
AA = np.transpose(AA)

def prop_func(A):

    B = np.zeros(A.shape, dtype=bool)

    # Compute the indices of the True values and make the two next to it True as well
    idcs_true = np.argwhere(A) + np.arange(propagation + 1)
    idcs_true = idcs_true.flatten()
    idcs_true = idcs_true[idcs_true < A.size] # in case the error propagation gives a great
    B[idcs_true] = True
    return B

# Apply prop_func along the rows (axis=1)
BB = np.apply_along_axis(prop_func, 0, AA)

# Array
print(f'Original array     AA = {AA}')
print(f'New array (2 true) BB = {BB}')

for i in range(AA.shape[0]):
    print(i, AA[i, 0], BB[i, 0])

# orig_detect = np.argwhere(pred_array[:, 0]) + range_box
# orig_detect = orig_detect.flatten()
# orig_detect = orig_detect[orig_detect <= pred_array[:, 0].size]
# maskOrig[orig_detect] = True



# import numpy as np
#
# # data
# n_data = 8
# x_data = np.arange(n_data)
# n_mc = 4
#
# # I varied the 2nd and 3rd list to better visualize the functionality
# y_true = np.array([[0.2, 2.3, 5.9, 7.0, 6.2, 4.7, 2.9],
#                   [1.2, 1.3, 4.9, 6.9, 4.2, 4.7, 2.9],
#                   [2.2, 2.3, 7.9, 6.2, 5.2, 3.7, 1.9]])
# noise = np.random.normal(0, 0.05, size=(y_true.shape[0], y_true.shape[1], n_mc))
# # y_1d = y_true[:, :, np.newaxis] + noise
#
# y_image = np.tile(y_true[:, None, :], (1, n_data, 1))
# y_image_mc = np.tile(y_image[..., np.newaxis], (1, 1, 1, n_mc))
# y_image_mc = y_image_mc + noise[:, np.newaxis, :, :]
# y_image_binary = y_image_mc > x_data[::-1, None, None]


# I have an array A with dimensions (3, 8, 7, 4) and an array B with dimensions (3, 7, 4) I would like to add to each of
# columns in array (its second axis) the corresponding value the values from array B. How do I do that?

# I have an array A with dimensions (3, 8, 7) and I would like to repeat the values in this arry 4 times in a new array
# using the function np.tile. how can i do that?
# # with dimensions (3,8,8,4)
# ()
# which corresponds to all the
# y_data = y_true[:, :, np.newaxis] + noise
#

# I have a numpy array A with dimensions (3, 8) I would like to create an array B with dimensions (3, 8, 8) numpy tile function where
# so that each rows from the tile A is repeated 8 times

# Reshape A to add a singleton dimension, making its shape (3, 8, 1)

# Use numpy.tile to repeat A_reshaped 8 times along the new axis, resulting in shape (3, 8, 8)
# images_gpt = np.tile(y_true[:, :, np.newaxis], (1, 1, 8))
#
# images = np.tile(y_true[:, None, :], (1, n_data, 1))
# noise = np.random.rand(3, 8, 4) - 0.5
# np.random.normal(0, 0.05, size=(3, 8, 4))
#
# images = images > x_data[::-1, None]
# images = images.astype(float)
#
# print(images)

# I have 2D numpy array with shape (3,8) I would link to add a random amount of noise to each row 4 times hence generating a
# 3,8,4 array. Could you suggest me how to do this with numpy fancy indexing?


# import numpy as np
#
# # Example 2D array with some NaN entries
# array_2d = np.array([[9, 10, 11, 12],
#                      [1, 2, np.nan, 4],
#                      [9, 10, 11, 12],
#                      [5, np.nan, 7, 8],
#                      [9, 10, 11, 12]])
#
# # Identify rows that contain NaN values
# idcs_nan_rows = np.isnan(array_2d).any(axis=1)
# array_no_nan = array_2d[~idcs_nan_rows, :]
#
# print(array_no_nan)

# import numpy as np
#
# # Example 1D array
# X = np.arange(1, 26)  # Example array with 20 elements
#
# # Parameters
# n_cols = 7
# offsets = np.arange(n_cols)  # Shape (n_cols,)
# n_rows = len(X) - 7 + 1  # Number of rows in the output array
#
# # Create an array of starting indices for each row
# idx = np.arange(n_rows)[:, None] + np.arange(n_cols)
# flux_2D = X
# result = X[idx]
#
# print(X)
# print(result)


# import numpy as np
#
# # Example arrays
# Y = np.arange(20)  # Target array
# X = np.array([2, 5, 10])  # Starting indices for masks
#
# # Create a mask array filled with False
# mask = np.zeros(Y.shape, dtype=bool)
#
# # Create a range of 5 elements (0 to 4) for broadcasting
# mask_range = np.arange(3)
#
# # Calculate the indices for the mask by broadcasting the addition
# mask_indices = X[:, np.newaxis] + mask_range
#
# # Flatten the mask indices and remove those that exceed the bounds of Y
# mask_indices = mask_indices.flatten()
# mask_indices = mask_indices[mask_indices < Y.size]
# mask[mask_indices] = True
#
# # Set the corresponding indices in the mask array to True
# for i, x_entry in enumerate(mask):
#     print(i, x_entry)

import pip._vendor.rich.progress as progress_bar
# from pip._internal.cli.progress_bars import get_download_progress_renderer
# from tqdm import tqdm, trange
#
# from itertools import product
#
# # Your two arrays
# array1 = [1, 2, 3]
# array2 = ['a', 'b', 'c', 'f', 'g']
#
# # Using itertools.product and enumerate to iterate over combinations with index
# for index, (element1, element2) in enumerate(product(array1, array2)):
#     print(f"Index: {index}, Elements: {element1}, {element2}")
#
# import time
#
# # # Create a list of items to iterate over
# # items = list(range(10))
# #
# # # Use tqdm to create a progress bar
# # for item in tqdm(items, desc="Item", mininterval=0.2, unit=" combinations"):
# #     # Simulate some work being done
# #     time.sleep(0.1)
#
# items = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
#
# # for item in tqdm(items, desc="Processing item {item}", unit="item"):
# #     # Simulate some work being done
#
# pbar = tqdm(items, unit=" line")
# for char in pbar:
#     pbar.set_description(desc=f'Line {char}')
