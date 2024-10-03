import numpy as np
import pandas as pd
import importlib
import joblib
import toml
from model_tools import stratified_train_test_split, read_sample_database
from time import time
from pathlib import Path
from plots import SampleReviewer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def run_training(x_arr, y_arr, estimator_id, estimator_params, fit_label):

    estimator = getattr(importlib.import_module(estimator_id[0]), estimator_id[1])

    # Run the training
    print(f'\nTraining: {fit_label}')
    print(f'- Settings: {str(estimator_params)}\n')
    start_time = time()
    ml_function = estimator(**estimator_params)
    ml_function.fit(x_arr, y_arr)
    end_time = np.round((time() - start_time) / 60, 2)
    print(f'- completed ({end_time} minutes)')

    return ml_function


def save_model(trained_model, x_arr, y_arr, output_root, fit_label, fit_cfg):
    # cv_results = search_algorithm.cv_results_

    print(f'\nRuning prediction on test set ({y_arr.size} points)')
    start_time = time()
    y_pred = trained_model.predict(x_arr)
    print(f'- completed ({(time() - start_time) / 60:0.2f} minutes)')

    # Testing confussion matrix
    print(f'\nConfusion matrix in test set ({y_arr.size} points)')
    start_time = time()
    conf_matrix_test = confusion_matrix(y_arr, y_pred, normalize="all")
    print(f'- completed ({(time() - start_time) / 60:0.2f} minutes)')

    # Precision, recall and f1:
    print(f'\nF1, Precision and recall diagnostics ({y_arr.size} points)')
    start_time = time()
    pres = precision_score(y_arr, y_pred, average='macro')
    recall = recall_score(y_arr, y_pred, average='macro')
    f1 = f1_score(y_arr, y_pred, average='macro')
    print(f'- completed ({(time() - start_time) / 60:0.2f} minutes)')

    print(f'\nModel outputs')
    print(f'- F1: \n {f1}')
    print(f'- Precision: \n {pres}')
    print(f'- Recall: \n {recall}')
    print(f'- Testing confusion matrix: \n {conf_matrix_test}')
    # print(f'- Fitting time: \n {fit_time}')

    # Save the trained model and configuration
    model_address = f'{output_root}_model.joblib'
    joblib.dump(trained_model, model_address)

    # Save results into a TOML file
    toml_path = f'{output_root}_FitSummary.toml'
    output_dict = {'resuts': {'f1': f1,
                              'precision': pres,
                              'Recall': recall,
                              'confusion_matrix': conf_matrix_test,
                              'fit_time': None},
                   fit_label: fit_cfg}

    with open(toml_path, 'w') as f:
        toml.dump(output_dict, f)

    return


def run_loaddb_and_training(database_address, model_cfg, list_labels, review_sample=False):

    # Prepare list of cfgs and databases
    list_databases = [database_address] if not isinstance(database_address, list) else database_address
    list_labels = [list_labels] if not isinstance(list_labels, list) else list_labels

    for i, database_file in enumerate(list_databases):

        # Set the configuration for the set:
        fitting_label = list_labels[i]
        print(f'\nLoading configuration: {fitting_label}')
        fit_cfg = model_cfg[fitting_label]

        # Load the training sample
        db_df = read_sample_database(database_file, fit_cfg)

        # Prepare training and testing sets
        df_train, df_test = stratified_train_test_split(db_df, fit_cfg['categories'], fit_cfg['sample_size'],
                                                        test_size=fit_cfg['test_sample_size_fraction'])
        x, y = df_train.iloc[:, 3:], df_train.iloc[:, 0]

        # if review_sample:
        #     category_sample_plot = 5000 if y.size < (len(fit_cfg['categories']) * 5000) else np.floor(y.size/len(fit_cfg['categories']))
        #     res_range = np.linspace(df_test['res_ratio'].min(), df_test['res_ratio'].max(), 100)
        #     diag = SampleReviewer(df_test, color_dict=model_cfg['colors'], detection_range=res_range, sample_size=category_sample_plot)
        #     diag.interactive_plot()

        # Review the training sample
        if review_sample:
            res_range = np.linspace(df_train['res_ratio'].min(), df_train['res_ratio'].max(), 100)
            diag = SampleReviewer(df_train, color_dict=model_cfg['colors'], detection_range=res_range)
            diag.interactive_plot()

        # Preparing the estimator:
        estimator = getattr(importlib.import_module(fit_cfg['estimator']["module"]), fit_cfg['estimator']["class"])
        estimator_params = fit_cfg.get('estimator_params', {})
        print(f'\nLoading estimator: {fit_cfg["estimator"]["class"]}')

        # Run the training
        print(f'\nTraining: {y.size} points ({fitting_label})')
        print(f'- Settings: {fit_cfg["estimator_params"]}\n')
        start_time = time()
        ml_function = estimator(**estimator_params)
        ml_function.fit(x, y)
        end_time = np.round((time()-start_time)/60, 2)
        print(f'- completed ({end_time} minutes)')

        # Save the trained model and configuration
        output_folder = Path(database_file).parent/'results'
        output_folder.mkdir(parents=True, exist_ok=True)
        file_stem = f'{fit_cfg["sample_prefix"]}_{fit_cfg["version"]}_{fit_cfg["scale"]}_{fitting_label}'

        model_address = output_folder/f'{file_stem}_model.joblib'
        joblib.dump(ml_function, model_address)

        # Run initial diagnostics
        print(f'\nReloading model from: {model_address}')
        start_time = time()
        ml_function = joblib.load(model_address)
        fit_time = np.round((time()-start_time)/60, 3)
        print(f'- completed ({fit_time} minutes)')

        # Setting test set
        x, y = df_test.iloc[:, 3:], df_test.iloc[:, 0]

        # # if review_sample:
        # if review_sample:
        #     category_sample_plot = 5000 if y.size < len(fit_cfg['categories']) * 5000 else y.size
        #     res_range = np.linspace(df_test['res_ratio'].min(), df_test['res_ratio'].max(), 100)
        #     diag = SampleReviewer(df_test, color_dict=model_cfg['colors'], detection_range=res_range)
        #     diag.interactive_plot()

        print(f'\nRuning prediction on test set ({y.size} points)')
        start_time = time()
        y_pred = ml_function.predict(x)
        print(f'- completed ({(time()-start_time)/60:0.2f} minutes)')

        # Testing confussion matrix
        print(f'\nConfusion matrix in test set ({y.size} points)')
        start_time = time()
        conf_matrix_test = confusion_matrix(y, y_pred, normalize="all")
        print(f'- completed ({(time()-start_time)/60:0.2f} minutes)')

        # Precision, recall and f1:
        print(f'\nF1, Precision and recall diagnostics ({y.size} points)')
        start_time = time()
        pres = precision_score(y, y_pred, average='macro')
        recall = recall_score(y, y_pred, average='macro')
        f1 = f1_score(y, y_pred, average='macro')
        print(f'- completed ({(time()-start_time)/60:0.2f} minutes)')

        print(f'\nModel outputs')
        print(f'- F1: \n {f1}')
        print(f'- Precision: \n {pres}')
        print(f'- Recall: \n {recall}')
        print(f'- Testing confusion matrix: \n {conf_matrix_test}')
        print(f'- Fitting time: \n {fit_time}')

        # Save results into a TOML file
        toml_path = output_folder/f'{file_stem}_FitSummary.toml'
        output_dict = {'resuts': {'f1':f1, 'precision':pres, 'Recall':recall, 'confusion_matrix':conf_matrix_test,
                                  'fit_time': fit_time}, fitting_label: fit_cfg,}
        with open(toml_path, 'w') as f:
            toml.dump(output_dict, f)

    return


