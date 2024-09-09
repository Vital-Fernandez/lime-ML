import numpy as np
import pandas as pd
import lime
import joblib
from pathlib import Path

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

cfg = sample_params[f'training_data_{version}']

# CPUs to use
n_cpus = joblib.cpu_count(only_physical_cores=True)

flux_file = data_folder / f'{scale}_{sample_prefix}_{version}.txt'
image_file = data_folder / f'{scale}_{sample_prefix}_{version}_image.txt'
class_list = ['white-noise', 'emission', 'cosmic-ray', 'pixel-line']
id_list = [cfg['classes'][label] for label in class_list]

for data_file in [flux_file]:

    # load the datase
    data_df = pd.read_csv(data_file, index_col= 0)
    data_df['spectral_number'] = data_df['spectral_number'].astype(int)

    # Divide into test and train sets
    idcs_types = data_df['spectral_number'].isin(id_list)
    data_df = data_df.loc[idcs_types]
    n_train = int(np.round(data_df.shape[0] * cfg['train_faction']))

    flux_sample, type_sample = data_df.iloc[:,3:].to_numpy(), data_df['spectral_number'].to_numpy(int)
    flux_train, type_train = flux_sample[:n_train], type_sample[:n_train]
    flux_test, type_test = flux_sample[n_train:], type_sample[n_train:]
    data_type = '1D' if flux_sample.shape[1] == cfg['box_pixels'] else '2D'

    for algorithm, ml_function in algorithm_dict.items():
        ml_conf = cfg[algorithm]
        ml_conf = cfg["RandomForestClassifier"]

        # Machine output file path
        ml_path = f'{data_folder}/{data_type}_{scale}_{algorithm}_{version}.joblib'

        print(f'Training the model {algorithm}: {type_train.size} points')
        ml_function_i = ml_function(**ml_conf)
        ml_function_i.fit(flux_train, type_train)

        # Saving to a file
        print(f'\n Saving model to: {ml_path}')
        joblib.dump(ml_function_i, ml_path)

        # Load the model
        print(f'\n Loading model from: {ml_path}')
        bin_clf = joblib.load(ml_path)

        # Train Confusion matrix
        print(f'\n Confusion Train matrix')
        detect_pred_train = cross_val_predict(bin_clf, flux_train, type_train, cv=2, n_jobs=-1)
        conf_matrix_out = confusion_matrix(type_train, detect_pred_train, normalize="all", n_jobs=-1)

        # Testing confussion matrix
        print(f'\n Confusion test matrix')
        detect_pred_test = cross_val_predict(bin_clf, flux_test, type_test, cv=2, n_jobs=-1)
        conf_matrix_test = confusion_matrix(type_test, detect_pred_test, normalize="all", cv=2, n_jobs=-1)

        # Precision and recall:
        print(f'\n Precision and recall matrix')
        pres = precision_score(type_test, detect_pred_test)
        recall = recall_score(type_test, detect_pred_test)

        print(f'- Training confusion matrix: \n {conf_matrix_out}')
        print(f'- Testing confusion matrix: \n {conf_matrix_test}')
        print(f'- Precision: \n {pres}')
        print(f'- Recall: \n {recall}')
