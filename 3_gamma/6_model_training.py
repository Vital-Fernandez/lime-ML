import numpy as np
import pandas as pd
import lime
import joblib
from pathlib import Path
from time import time
import json

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from model_tools import algorithm_dict

# Read configuration
cfg_file = 'training_sample_v3.toml'
sample_params = lime.load_cfg(cfg_file)
version = sample_params['data_labels']['version']
scale = sample_params['data_labels']['scale']
data_folder = Path(sample_params['data_labels']['output_folder']) / version
sample_prefix = sample_params['data_labels']['sample_prefix']
suffix = 'no_max_features'

cfg = sample_params[f'training_data_{version}']

# CPUs to use
n_cpus = joblib.cpu_count(only_physical_cores=True)

flux_file = data_folder / f'{scale}_{sample_prefix}_{version}.txt'
image_file = data_folder / f'{scale}_{sample_prefix}_{version}_image.txt'
class_list = ['white-noise', 'emission', 'cosmic-ray', 'pixel-line']
id_list = [cfg['classes'][label] for label in class_list]

for data_file in [image_file]:
    print(0)

    # load the datase
    data_df = pd.read_csv(data_file, index_col= 0)
    data_df['spectral_number'] = data_df['spectral_number']
    n_data_points = data_df.iloc[:,3:].shape
    data_type = '1D' if n_data_points[1] == cfg['box_pixels'] else '2D'
    print(1)

    # Divide into test and train sets
    idcs_types = data_df['spectral_number'].isin(id_list)
    data_df = data_df.loc[idcs_types]
    n_train = int(np.round(data_df.shape[0] * cfg['train_faction']))
    print(2)

    flux_sample, type_sample = data_df.iloc[:,3:].to_numpy(), data_df['spectral_number'].to_numpy(int)
    flux_train, type_train = flux_sample[:n_train], type_sample[:n_train]
    flux_test, type_test = flux_sample[n_train:], type_sample[n_train:]

    for algorithm, ml_function in algorithm_dict.items():

        # Algorithm i_th configuration
        ml_conf = cfg[algorithm]

        # Machine output file path
        ml_path = f'{data_folder}/{data_type}_{scale}_{algorithm}_{version}_{suffix}.joblib'

        print(f'\n Training {data_type} dataset with {algorithm}: {type_train.size} points ({suffix}) ')
        start_time = time()
        ml_function_i = ml_function(**ml_conf)
        ml_function_i.fit(flux_train, type_train)
        end_time = time()
        print(f'- completed ({(end_time-start_time)/60:0.2f} minutes)')

        # Saving to a file
        print(f'\nSaving model:')
        joblib.dump(ml_function_i, ml_path)
        print(f'- completed ({ml_path})')

        # Saving configuration
        with open(f'{data_folder}/{data_type}_{scale}_{algorithm}_{version}_{suffix}_parameters.txt', 'w') as file:
            for key, value in ml_conf.items():
                file.write(f'{key}: {value}\n')
            file.write(f'n_points: {type_train.size}\n')
            file.write(f'time: {(end_time-start_time)/60:0.2f} minutes\n')