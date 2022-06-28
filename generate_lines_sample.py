import numpy as np
import pandas as pd
from scipy import stats
from lime.model import gaussian_model
from functions import SAMPLE_SIZE, SEED_VALUE, MASK_MAX_SIZE, PARMS_FILE
from matplotlib import rcParams

STANDARD_PLOT = {'figure.figsize': (10, 8),
                 'axes.titlesize': 16,
                 'axes.labelsize': 16,
                 'legend.fontsize': 14,
                 'xtick.labelsize': 14,
                 'ytick.labelsize': 14}

rcParams.update(STANDARD_PLOT)


# Recomver the line sample parameters
df = pd.read_csv(PARMS_FILE)

amp_array = df['amp']
noise_array = df['noise']

sigma_g_array = df['sigma_g']
lambda_step_array = df['lambda_step']

w_blue_array = df['w_blue']
w_red_array = df['w_red']

line_check_array = df['line_check']

# Define the line array limits
w_low = -3 * sigma_g_array + w_blue_array
w_high = 3 * sigma_g_array + w_red_array

# Continuum values
m = 0
n = 1

# Data container
data_matrix_x = np.full((SAMPLE_SIZE, MASK_MAX_SIZE), 0.0)
data_matrix_y = np.full((SAMPLE_SIZE, MASK_MAX_SIZE), 0.0)
array_lengths = np.full(SAMPLE_SIZE, np.nan)

# Compute the wavelength intervals
i_range = np.arange(SAMPLE_SIZE)
for i in i_range:

    # Random generator
    rnd = np.random.RandomState(seed=1)

    # Compute the line
    x = np.arange(w_low[i], w_high[i], lambda_step_array[i])
    cont_noise = rnd.normal(0, noise_array[i], x.size)
    y = cont_noise + (m*x + n)

    if line_check_array[i]:
        y = gaussian_model(x, amp_array[i], 0, sigma_g_array[i]) + cont_noise

    # Normalized the flux
    norm = np.sqrt((y[0]**2+y[-1]**2)/2)
    # y = y / np.max(y)
    y = (y - norm)/norm

    # Set the wavelength 0 at blue end
    x = x - w_low[i]

    # Save the data
    data_matrix_x[i, : x.size] = x
    data_matrix_y[i, : y.size] = y

    # Store the lengths as a security check
    array_lengths[i] = x.size

# Save to a text file
column_names = np.full(MASK_MAX_SIZE, 'Pixel')
column_names = np.char.add(column_names, np.arange(MASK_MAX_SIZE).astype(str))

df = pd.DataFrame(data=data_matrix_y, columns=column_names)
df.to_csv('sample_flux_table.csv', index=False)

df = pd.DataFrame(data=data_matrix_x, columns=column_names)
df.to_csv('sample_wave_table.csv', index=False)

print('Finished')
print(f'The largest is {np.max(array_lengths)}, the smallest {np.min(array_lengths)}')

