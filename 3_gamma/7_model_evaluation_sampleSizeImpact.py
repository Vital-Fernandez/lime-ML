
import lime
import joblib
from pathlib import Path
from matplotlib import pyplot as plt, rc_context, cm
from training import run_loaddb_and_training
from model_tools import read_sample_database, stratified_train_test_split
from plots import SampleReviewer, diagnostics_plot
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import tomllib
import numpy as np
from lime.plots import theme

format_figure = {'figure.figsize': (4, 4), 'figure.dpi': 200, "xtick.labelsize" : 8, "ytick.labelsize" : 10,
                 "legend.fontsize": 10}


# Read sample configuration
cfg_file = 'training_sample_v3.toml'
sample_params = lime.load_cfg(cfg_file)

version = sample_params['data_labels']['version']
scale = sample_params['data_labels']['scale']
sample_prefix = sample_params['data_labels']['sample_prefix']

# sampleSizeTests_2categories_v2_nSamples250000_FitSummary.toml

# Read the file list
parent_folder = Path('/home/vital/Astrodata/LiMe_ml/v3/results/')
extension_list = ['2categories_v2', '3categories_v2', '4categories_v2']
file_list = list(parent_folder.glob('*sampleSizeTests_*.toml'))

with rc_context(theme.fig_defaults(format_figure)):

    fig, ax = plt.subplots()
    x_cord, y_cord = 'n_samples','f1'

    for i, ext in enumerate(extension_list):

        x_arr, y_arr = [], []

        for file_i in file_list:
            if ext in file_i.as_posix():

                label = file_i.stem[file_i.stem.find('v2_'):file_i.stem.find('_FitSummary')]
                n_samples = float(label[11:])
                with open(file_i, "rb") as f:
                    data = tomllib.load(f)

                x_arr.append(n_samples)
                y_arr.append(float(data['resuts'][y_cord]))

        # Sort and plot
        x_arr, y_arr, data_name = np.array(x_arr), np.array(y_arr), f'{ext[0]} categories data set'
        idcs_sorted = np.argsort(x_arr)
        ax.plot(x_arr[idcs_sorted], y_arr[idcs_sorted], label=data_name)
        ax.update({'xlabel': 'Training sample size per category', 'ylabel': 'f1 score'})

    ax.legend()
    plt.tight_layout()
    plt.show()



