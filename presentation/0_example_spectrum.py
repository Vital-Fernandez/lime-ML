import numpy as np
import lime


wave_array, flux_array, err_array = np.loadtxt('manga_spectrum.txt', unpack=True)
manga_spec = lime.Spectrum(wave_array, flux_array, err_array, redshift=0.0475, norm_flux=1e-17, pixel_mask=np.isnan(err_array))
manga_spec.plot.spectrum(rest_frame=True, show_masks=False)




