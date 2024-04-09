import numpy as np
import pandas as pd
import lime
import joblib
from pathlib import Path

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from tools import algorithm_dict

# Read configuration
n_cpus = joblib.cpu_count(only_physical_cores=True)
cfg_file = 'config_file.toml'
cfg = lime.load_cfg(cfg_file)
output_folder = Path(cfg['data_location']['output_folder'])
version = cfg['data_grid']['version']
train_frac = cfg['data_grid']['training_fraction']

# IDs for the dataset
label = 'small_box_min_max'
version = cfg['data_grid']['version']

# Load the data sets
subfolder = output_folder/f'_results_{label}_{version}'
database_file = subfolder/f'database_{label}_{version}.csv'
detect_file = subfolder/f'sample_detection_{label}_{version}.csv'
flux_file = subfolder/f'sample_flux_{label}_{version}.csv'
image_file = subfolder/f'sample_image_{label}_{version}.csv'

flux_df = pd.read_csv(flux_file)
image_df = pd.read_csv(image_file)
line_bool_arr = np.loadtxt(detect_file)
database_df = lime.load_log(database_file)


# Divide into test and train sets
n_train = int(np.round(line_bool_arr.size * train_frac))
flux_sample, image_sample, detect_sample = flux_df.to_numpy(), image_df.to_numpy(), line_bool_arr.astype(bool)
flux_sample_train, image_train, detect_train = flux_sample[:n_train], image_df[:n_train], detect_sample[:n_train]
flux_sample_test, image_test, detect_test = flux_sample[n_train:], image_df[n_train:], detect_sample[n_train:]

# Confirm that the samples have the same pattern than in the database
db_detect = database_df['detection'].to_numpy(dtype=bool)
db_detect_train, db_detect_test = db_detect[:n_train], db_detect[n_train:]
assert np.all(db_detect_train == detect_train)
assert np.all(db_detect_test == detect_test)

# Training algorithms
train_sample = {'1D': flux_sample_train, '2D': image_train}
test_sample = {'1D': flux_sample_test, '2D': image_test}

# Generate the clasifiers with different algorithms
for model_name, model in algorithm_dict.items():

    # Test results of the fits
    print(f'\n-- Results for {model_name}:')

    for data_type in ['1D', '2D']:

        data_train = train_sample[data_type]
        data_test =  test_sample[data_type]

        print(f'\n------ {data_type} data)')

        # Machine output file path
        ml_path = f'{subfolder}/_{data_type}_{label}_{version}_{model_name}.joblib'

        # Fit the sample
        model.fit(data_train, detect_train)

        # Saving to a file
        joblib.dump(model, ml_path)

        # Load the model
        bin_clf = joblib.load(ml_path)

        # Train Confusion matrix
        detect_pred_train = cross_val_predict(bin_clf, data_train, detect_train, cv=5)
        conf_matrix_out = confusion_matrix(detect_train, detect_pred_train, normalize="all")

        # Testing confussion matrix
        detect_pred_test = cross_val_predict(bin_clf, data_test, detect_test, cv=5)
        conf_matrix_test = confusion_matrix(detect_test, detect_pred_test, normalize="all")

        # Precision and recall:
        pres = precision_score(detect_test, detect_pred_test)
        recall = recall_score(detect_test, detect_pred_test)

        print(f'- Training confusion matrix: \n {conf_matrix_out}')
        print(f'- Testing confusion matrix: \n {conf_matrix_test}')
        print(f'- Precision: \n {pres}')
        print(f'- Recall: \n {recall}')



# /home/vital/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
# STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
#
# Increase the number of iterations (max_iter) or scale the data as shown in:
#     https://scikit-learn.org/stable/modules/preprocessing.html
# Please also refer to the documentation for alternative solver options:
#     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
#   n_iter_i = _check_optimize_result(