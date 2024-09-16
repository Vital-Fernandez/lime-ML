import numpy as np
import lime
from pathlib import Path
from astropy.cosmology import Planck18 as cosmo

# Calculate the lookback time and the age of the universe at the given redshift
redshift = 4.299
lookback_time = cosmo.lookback_time(redshift)
age_of_universe = cosmo.age(redshift)

# Print the results
print(f"Lookback time: {lookback_time:.2f}")
print(f"Age of the universe at redshift {redshift}: {age_of_universe:.2f}")


lime.theme.set_style(style='dark')

# wave_array, flux_array, err_array = np.loadtxt('manga_spectrum.txt', unpack=True)
# manga_spec = lime.Spectrum(wave_array, flux_array, err_array, redshift=0.0475, norm_flux=1e-17, pixel_mask=np.isnan(err_array))
# manga_spec.plot.spectrum(rest_frame=True, show_masks=False)

fig_cfg = {"figure.dpi" : 2000, "figure.figsize" : (8, 2)}
output_folder=Path('/home/vital/Dropbox/Astrophysics/Seminars/BootCamp2025')

spec_address = '/home/vital/PycharmProjects/ceers-data/data/spectra/CEERs_DR0.9/nirspecDDT/prism/hlsp_ceers_jwst_nirspec_nirspecDDT-001586_prism_dr0.9_x1d.fits'
spec = lime.Spectrum.from_file(spec_address, instrument='nirspec', redshift=redshift, crop_waves=(0.75, 5.2))
# spec.plot.spectrum()
spec.unit_conversion('AA', 'FLAM')
ax_cfg = {'title': f'Galaxy MSA1586         , at z = {redshift} (lookback time {lookback_time:.2f})'}
spec.plot.spectrum(rest_frame=True, fig_cfg=fig_cfg, ax_cfg=ax_cfg, output_address=output_folder/'MSA1586.svg')





