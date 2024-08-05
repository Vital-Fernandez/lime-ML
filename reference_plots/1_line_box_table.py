import numpy as np

n_sigmas = 8
c_KMpS = 299792.458
R_array = np.array([30, 420, 636, 1500, 2000, 2500, 3750, 5000])
sigma_array = np.array([30, 60, 120, 240, 480])

for n_sigmas in [1, 6]:
    for R_value in R_array:
        b_array = n_sigmas * sigma_array/c_KMpS * R_value
        print(R_value, np.ceil(b_array))
    print('\n', '\n')

