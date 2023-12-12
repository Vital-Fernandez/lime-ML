import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

# Generate data: for N=1e6, the triangulation hogs 1 GB of memory
N = 1000000
x, y = 10 * np.random.random((2, N))
rho = np.sin(3*x) + np.cos(7*y)**3

# Set up a regular grid of interpolation points
xi, yi = np.linspace(x.min(), x.max(), 300), np.linspace(y.min(), y.max(), 300)
xi, yi = np.meshgrid(xi, yi)

# Interpolate; there's also method='cubic' for 2-D data such as here
zi = scipy.interpolate.griddata((x, y), rho, (xi, yi), method='linear')

plt.imshow(zi, vmin=rho.min(), vmax=rho.max(), origin='lower',
           extent=[x.min(), x.max(), y.min(), y.max()])
plt.colorbar()
plt.show()