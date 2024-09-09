import numpy as np
from tools import normalization_1d

# data
n_data = 13
x_data = np.arange(n_data)
# I varied the 2nd and 3rd list to better visualize the functionality
y_data = np.array([[0.2, 2.3, 5.9, 7.2, 6.2, 4.7, 2.9, 0.9],
                  [1.2, 1.3, 4.9, 7.2, 4.2, 4.7, 2.9, 0.9],
                  [2.2, 2.3, 7.9, 6.2, 5.2, 3.7, 1.9, 0.9]])


images = np.tile(y_data[:, None, :], (1, n_data, 1))
images = images > x_data[::-1, None]
images = images.astype(float)

print(images)
images.reshape((3, 1, 64))