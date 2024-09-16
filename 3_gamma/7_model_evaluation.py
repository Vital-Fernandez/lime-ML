import numpy as np
import pandas as pd
import lime
import joblib
from pathlib import Path
from time import time

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
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

for data_file in [flux_file, image_file]:

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

        # Algorithm i_th configuration
        ml_conf = cfg[algorithm]

        # Machine output file path
        ml_path = f'{data_folder}/{data_type}_{scale}_{algorithm}_{version}_{suffix}.joblib'

        # Load the model
        print(f'\n Loading model from: {ml_path}')
        start_time = time()
        bin_clf = joblib.load(ml_path)
        print(f'- completed ({(time()-start_time)/60:0.2f} minutes)')

        # Check on one entry
        start_time = time()
        model_pred, true_value = bin_clf.predict(flux_train[0, :].reshape(1, -1))[0], type_train[0]
        print(f'Model output: {model_pred}, real type {true_value} ({model_pred == true_value}), ({(time()-start_time)/60:0.7f} minutes)')

        # Train Confusion matrix
        print(f'\n Runing cross_val_predict in train set ({type_train.size} points) ')
        start_time = time()
        detect_pred_train = cross_val_predict(bin_clf, flux_train, type_train, cv=2, n_jobs=-1)
        print(f'- completed ({(time()-start_time)/60:0.2f} minutes)')

        print(f'\n Runing cross_val_predict in test set ({type_test.size} points)')
        start_time = time()
        detect_pred_test = cross_val_predict(bin_clf, flux_test, type_test, cv=2, n_jobs=-1)
        print(f'- completed ({(time()-start_time)/60:0.2f} minutes)')

        # Testing confussion matrix
        print(f'\n Confusion matrix in train set ({type_train.size} points)')
        start_time = time()
        conf_matrix_out = confusion_matrix(type_train, detect_pred_train, normalize="all")
        print(f'- completed ({(time()-start_time)/60:0.2f} minutes)')

        print(f'\n Confusion matrix in test set ({type_test.size} points)')
        start_time = time()
        conf_matrix_test = confusion_matrix(type_test, detect_pred_test, normalize="all")
        print(f'- completed ({(time()-start_time)/60:0.2f} minutes)')

        # Precision and recall:
        print(f'\n F1, Precision and recall matrix ({type_test.size} points)')
        start_time = time()
        pres = precision_score(type_test, detect_pred_test, average='macro')
        recall = recall_score(type_test, detect_pred_test, average='macro')
        f1 = f1_score(type_test, detect_pred_test, average='macro')

        print(f'- Training confusion matrix: \n {conf_matrix_out}')
        print(f'- Testing confusion matrix: \n {conf_matrix_test}')
        print(f'- F1: \n {f1}')
        print(f'- Precision: \n {pres}')
        print(f'- Recall: \n {recall}')
        print(f'- completed ({(time()-start_time)/60:0.2f} minutes)')




