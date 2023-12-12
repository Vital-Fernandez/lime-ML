import numpy as np
import pandas as pd
import lime
import joblib

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score


# Load the configuration
cfg = lime.load_cfg('config_file.ini')
output_folder = cfg['data_location']['output_folder']
TRAINING_SIZE = int(cfg['ml_grid_design']['training_size'])
rnd = np.random.RandomState()
version = 'v2_cost1_logNorm'

# Load the data
line_bool = np.loadtxt(f'{output_folder}/sample_detection_training_{version}.csv', dtype=bool)
wave_df = pd.read_csv(f'{output_folder}/sample_wave_training_{version}.csv', header=0)
flux_df = pd.read_csv(f'{output_folder}/sample_flux_training_{version}.csv', header=0)

# Divide into test and train sets
flux_sample, lineCheck_sample = flux_df.to_numpy(), line_bool.astype(bool)
flux_sample_train, lineCheck_train = flux_sample[:TRAINING_SIZE], lineCheck_sample[:TRAINING_SIZE],
flux_sample_test, lineCheck_test = flux_sample[TRAINING_SIZE:], lineCheck_sample[TRAINING_SIZE:]

# Generate the clasifiers with different algorithms
algorithm_dict = {'GradientDescent': SGDClassifier, 'LogitistRegression': LogisticRegression}
for model_name, model in algorithm_dict.items():

    # Machine output file path
    ml_path = f'{output_folder}/{model_name}_{version}.joblib'

    # Fit the sample
    bin_clf = model(max_iter=1000)
    bin_clf.fit(flux_sample_train, lineCheck_train)

    # Saving to a file
    joblib.dump(bin_clf, ml_path)

    # Load the model
    bin_clf = joblib.load(ml_path)

    # Test results of the fits
    print(f'\n----- Results for {model_name}:')

    # Train Confusion matrix
    lineCheck_pred = cross_val_predict(bin_clf, flux_sample_train, lineCheck_train, cv=3)
    conf_matrix_out = confusion_matrix(lineCheck_train, lineCheck_pred, normalize="all")

    # Testing confussion matrix
    lineCheck_pred = cross_val_predict(bin_clf, flux_sample_test, lineCheck_test, cv=3)
    conf_matrix_test = confusion_matrix(lineCheck_test, lineCheck_pred, normalize="all")

    # Precision and recall:
    pres = precision_score(lineCheck_test, lineCheck_pred)
    recall = recall_score(lineCheck_test, lineCheck_pred)

    print(f'- Training confusion matrix: \n {conf_matrix_out}')
    print(f'- Testing confusion matrix: \n {conf_matrix_test}')
    print(f'- Precision: \n {pres}')
    print(f'- Recall: \n {recall}')

