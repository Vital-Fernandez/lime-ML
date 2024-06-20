import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import lime


address = r'/home/vital/PycharmProjects_backUp/ceers-data/data/spectra/CEERs_DR0.7/nirspec4/prism/hlsp_ceers_jwst_nirspec_nirspec4-001027_prism_v0.7_x1d-masked.fits'

spec = lime.Spectrum.from_file(address, 'nirspec', redshift=7.8334)
spec.unit_conversion(wave_units_out='Angstrom', flux_units_out='FLAM')

# spec.plot.spectrum(rest_frame=True)
wave_obs = spec.wave.data
deltalamb_arr = np.diff(wave_obs)
R_arr = wave_obs[1:]/deltalamb_arr
FWHM_arr = wave_obs[1:]/ R_arr

fig, ax = plt.subplots()
# ax.plot(wave_obs[1:], R)
ax.plot(wave_obs[1:], FWHM_arr)
plt.show()

# # Example data: binary array with peaks
# data = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
#
# # Desired width for broadening the delta function (total width)
# desired_width = 3  # This will add 1 '1' to each side of the original '1'
#
# # Create a kernel of ones with the desired width
# kernel = np.ones(desired_width)
#
# # Convolve the data with the kernel
# # mode='same' ensures the output array is the same size as the input array
# broadened_data = convolve(data, kernel)
#
# # Since convolution can increase counts above 1, threshold the result to get binary output
# # broadened_data = (broadened_data > 0).astype(int)
#
# # Plot the original and broadened data
# plt.figure(figsize=(10, 4))
# plt.plot(data, 'o-', label='Original Delta Functions')
# plt.plot(broadened_data, label='Broadened Peaks')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.legend()
# plt.show()

# /home/vital/PycharmProjects_backUp/ceers-data/data/spectra/CEERs_DR0.7/nirspec4/prism/hlsp_ceers_jwst_nirspec_nirspec4-001027_prism_v0.7_x1d-masked.fits
# /home/vital/PycharmProjects_backUp/ceers-data/data/spectra/CEERs_DR0.7/nirspec11/prism/hlsp_ceers_jwst_nirspec_nirspec11-001027_prism_v0.7_x1d.fits