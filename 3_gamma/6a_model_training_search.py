import numpy as np
import pandas as pd
from xarray.core.formatting import first_n_items

import lime
import joblib
from pathlib import Path
import importlib
from time import time
import json
import toml

from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from model_tools import algorithm_dict, read_sample_database
from model_tools import stratified_train_test_split, prepare_search_algorithm, save_search_results
from plots import SampleReviewer
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV

# Read sample configuration
cfg_file = 'training_sample_v3.toml'
sample_params = lime.load_cfg(cfg_file)

version = sample_params['data_labels']['version']
scale = sample_params['data_labels']['scale']
sample_prefix = sample_params['data_labels']['sample_prefix']

# Read the sample database
data_folder = Path(sample_params['data_labels']['output_folder'])/version
sample1D_database_file = data_folder/f'{sample_prefix}_{version}_{scale}.csv'

fit_label = '4_categories_v1_search_80000points_min_samples_1-5-10-15percent_mult_12_features'
cfg = sample_params[fit_label]

# Recover predictor algorithm from sklearn
estimator_class = getattr(importlib.import_module(cfg['estimator']["module"]), cfg['estimator']["class"])
estimator = estimator_class(**cfg['estimator_params'])

for sample_file in [sample1D_database_file]:

    # Load the training sample
    db_df = read_sample_database(sample_file, cfg)

    # Prepare training and testing sets
    df_train, df_test = stratified_train_test_split(db_df, cfg['categories'], cfg['sample_size'],
                                                    test_size=cfg['test_sample_size_fraction'])
    x, y = df_train.iloc[:, 3:], df_train.iloc[:, 0]

    # Define the searchg algorithm
    search = GridSearchCV(estimator=estimator, param_grid=cfg['param_distributions'])

    # Run the fit
    search.fit(x, y)

    # Save the results
    x, y = df_test.iloc[:, 3:], df_test.iloc[:, 0]
    output_root = data_folder/'results'/ f'{sample_prefix}_{version}_{scale}_{fit_label}'
    save_search_results(search, sample_params, fit_label, x, y, output_root)

