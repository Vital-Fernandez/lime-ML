import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rc_context
from matplotlib.widgets import Cursor
import lime
from pathlib import Path
from lime.plots import theme
from plots import SampleReviewer

# def update_feature_plot(event, df, x_arr):
#     if event.inaxes == ax1:
#
#         # Get the position of the cursor
#         x_mouse, y_mouse = event.xdata, event.ydata
#
#         # Calculate distances from the cursor to all data points
#         distances = np.sqrt((df['res_ratio'] - x_mouse) ** 2 + (df['int_ratio'] - y_mouse) ** 2)
#
#         # Find the index of the nearest point
#         ind = np.argmin(distances)
#
#         # Update feature plot based on the nearest point
#         feature_values = df.iloc[ind, 3:].values
#         feature_lines.set_data(x_arr, feature_values)
#         ax2.set_xlim(0, len(feature_values) - 1)
#         ax2.set_ylim(feature_values.min(), feature_values.max())
#         fig.canvas.draw_idle()


# Figure format
fig_cfg = theme.fig_defaults({'axes.labelsize': 10,
                              'axes.titlesize': 10,
                              'figure.figsize': (3, 3),
                              'hatch.linewidth': 0.3,
                              "legend.fontsize": 8})

# Read sample configuration
cfg_file = 'training_sample_v3.toml'
sample_params = lime.load_cfg(cfg_file)

version = sample_params['data_labels']['version']
cfg = sample_params[f'training_data_{version}']
inverted_class_dict = {value: key for key, value in cfg['classes'].items()}

# Read the sample database
output_folder = Path(sample_params['data_labels']['output_folder'])/version
sample_database = f'{output_folder}/training_multi_sample_{version}.csv'
database_df = pd.read_csv(sample_database)


n_points = 5000
list_features = [1, 2, 3, 4]
idcs = database_df['spectral_number'].isin(list_features)
database_df = database_df.loc[idcs].sample(n_points)
database_df['spectral_number'] = database_df['spectral_number'].astype(int)

diag = SampleReviewer(database_df, id_dict=inverted_class_dict, color_dict=sample_params['colors'])
diag.interactive_plot()

# wave_arr = np.arange(cfg['box_pixels'])
#
# # Creat the figure
# with rc_context(fig_cfg):
#
#     # Create the figure and two subplots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#
#     # Diagnostic axis
#     for label_feature, number_feature in cfg['classes'].items():
#         if number_feature > 0 and (number_feature != 5.5):
#
#             idcs_class = database_df['spectral_number'] == number_feature
#             int_ratio = database_df.loc[idcs_class, 'int_ratio'].to_numpy()
#             res_ratio = database_df.loc[idcs_class, 'res_ratio'].to_numpy()
#             color = cfg[label_feature]['color']
#
#             ax1.scatter(res_ratio, int_ratio, color=color, label=label_feature, alpha=0.5, edgecolor='none')
#
#     # Connect the event handler for motion events
#     fig.canvas.mpl_connect('motion_notify_event', update_feature_plot)
#
#
#     # Line axis
#     y_0 = database_df.iloc[0, 3:].to_numpy()
#     feature_lines, = ax2.step(wave_arr, y_0, where='mid')
#
#     # Wording
#     ax1.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}} = \sigma_{pixels}$ (Gaussian sigma in pixels)',
#          'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$ (Signal-to-noise)'})
#     ax1.legend(loc='lower center', ncol=2, framealpha=0.95)
#
#     ax2.update({'xlabel': 'Feature Number', 'ylabel': 'value'})
#
#     # Axis format
#     ax1.set_yscale('log')
#
#     plt.tight_layout()
#     plt.show()

# # Sample DataFrame creation
# np.random.seed(0)
# df = pd.DataFrame({
#     'Class': np.random.randint(0, 2, size=100),
#     'X': np.random.rand(100),
#     'Y': np.random.rand(100),
#     'Feature1': np.random.rand(100),
#     'Feature2': np.random.rand(100),
#     'Feature3': np.random.rand(100)
# })
#
# # Create the figure and two subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#
# # Scatter plot on the left
# sc = ax1.scatter(df['X'], df['Y'], c=df['Class'], cmap='viridis')
# ax1.set_xlabel('X Coordinate')
# ax1.set_ylabel('Y Coordinate')
# ax1.set_title('Scatter Plot')
#
# # Feature plot on the right
# ax2.set_xlabel('Feature Number')
# ax2.set_ylabel('Feature Value')
# ax2.set_title('Feature Values')
# feature_lines, = ax2.step([], [], where='mid')
#
#
# def update_feature_plot(event):
#     if event.inaxes == ax1:
#         # Get the position of the cursor
#         x_mouse, y_mouse = event.xdata, event.ydata
#
#         # Calculate distances from the cursor to all data points
#         distances = np.sqrt((df['X'] - x_mouse) ** 2 + (df['Y'] - y_mouse) ** 2)
#
#         # Find the index of the nearest point
#         ind = np.argmin(distances)
#
#         # Update feature plot based on the nearest point
#         feature_values = df.iloc[ind, 3:].values
#         x = np.arange(len(feature_values))
#         feature_lines.set_data(x, feature_values)
#         ax2.set_xlim(0, len(feature_values) - 1)
#         ax2.set_ylim(min(feature_values), max(feature_values))
#         fig.canvas.draw_idle()
#
#
# # Connect the event handler for motion events
# fig.canvas.mpl_connect('motion_notify_event', update_feature_plot)
#
# plt.show()
