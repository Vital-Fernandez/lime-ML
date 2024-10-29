import numpy as np
import pandas as pd

import lime
from pathlib import Path
from lime.plots import theme
from plots import SampleReviewer
from model_tools import stratified_train_test_split


# Figure format
fig_cfg = theme.fig_defaults({'axes.labelsize': 10,
                              'axes.titlesize': 10,
                              'figure.figsize': (3, 3),
                              'hatch.linewidth': 0.3,
                              "legend.fontsize": 8})

# Read sample configuration
cfg_file = 'training_sample_v4.toml'
sample_params = lime.load_cfg(cfg_file)

version = sample_params['data_labels']['version']
scale = sample_params['data_labels']['scale']
sample_prefix = sample_params['data_labels']['sample_prefix']
cfg = sample_params[f'training_data_{version}']

# Read the sample database
output_folder = Path(sample_params['data_labels']['output_folder'])/version
sample_database = f'{output_folder}/{sample_prefix}_{version}.csv'
database_df = pd.read_csv(sample_database)

n_points = 5000
list_features = ['white-noise', 'continuum', 'dead-pixel', 'absorption', 'doublet']
res_range = np.linspace(cfg["res-ratio_min"], cfg["box_pixels"]/cfg["n_sigma"], num=100)

df_train, df_test = stratified_train_test_split(database_df, list_features, n_points, test_size=0.2)
diag = SampleReviewer(df_train, color_dict=sample_params['colors'], sample_size=n_points)
diag.interactive_plot()

# diag = SampleReviewer(df_test, color_dict=sample_params['colors'], detection_range=res_range)
# diag.interactive_plot()