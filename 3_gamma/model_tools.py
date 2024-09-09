import numpy as np
import pandas as pd
import joblib
from lime import load_frame, save_frame, load_cfg
from lime import detection_function
from lime.plots import theme
from matplotlib import pyplot as plt, rc_context
from pathlib import Path

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


STANDARD_PLOT = theme.fig_defaults()
STANDARD_PLOT['axes.labelsize'] = 20
STANDARD_PLOT['figure.figsize'] = (11, 11)

c_kmpers = 299792.458  # Km/s


# Training algorithms
algorithm_dict = {'RandomForestClassifier': RandomForestClassifier}


def feature_scaling(data, transformation='min-max', log_base=None, axis=1):

    if transformation == 'min-max':
        data_min_array = data.min(axis=axis, keepdims=True)
        data_max_array = data.max(axis=axis, keepdims=True)
        data_norm = (data - data_min_array) / (data_max_array - data_min_array)

    elif transformation == 'log':
        y_cont = data - data.min(axis=1, keepdims=True) + 1
        data_norm = np.emath.logn(log_base, y_cont)

    elif transformation == 'log-min-max':
        data_cont = data - data.min(axis=1, keepdims=True) + 1
        log_data = np.emath.logn(log_base, data_cont)
        log_min_array, log_max_array = log_data.min(axis=axis, keepdims=True), log_data.max(axis=axis, keepdims=True)
        data_norm = (log_data - log_min_array) / (log_max_array - log_min_array)

    else:
        raise KeyError(f'Input scaling "{transformation}" is not recognized.')

    return data_norm

def analysis_check(conf, version, label, output_folder):

    output_folder = Path(output_folder)

    subfolder = output_folder/f'_results_{label}_{version}'
    database_file = subfolder / f'database_{label}_{version}.csv'
    detect_file = subfolder / f'sample_detection_{label}_{version}.csv'
    flux_file = subfolder / f'sample_flux_{label}_{version}.csv'
    image_file = subfolder / f'sample_image_{label}_{version}.csv'

    line_bool_arr = np.loadtxt(detect_file)
    database_df = load_log(database_file)
    flux_df = pd.read_csv(flux_file)
    image_df = pd.read_csv(image_file)

    # Read the configuration
    train_frac = conf['data_grid']['training_fraction']
    box_limit = conf['data_grid']['small_box_reslimit']

    # Divide into test and train sets
    n_train = int(np.round(line_bool_arr.size * train_frac))
    flux_sample, image_sample, detect_sample = flux_df.to_numpy(), image_df.to_numpy(), line_bool_arr.astype(bool)
    flux_sample_train, image_train, detect_train = flux_sample[:n_train], image_df[:n_train], detect_sample[:n_train]
    flux_sample_test, image_test, detect_test = flux_sample[n_train:], image_df[n_train:], detect_sample[n_train:]
    db_sample, db_test = database_df.iloc[:n_train,:], database_df.iloc[n_train:,:]

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
            data_test = test_sample[data_type]

            print(f'\n------ {data_type} data)')

            # Machine output file path
            ml_path = f'{subfolder}/_{data_type}_{label}_{version}_{model_name}.joblib'

            # Load the model
            bin_clf = joblib.load(ml_path)

            # Train Confusion matrix
            detect_pred_train = cross_val_predict(bin_clf, data_train, detect_train, cv=5)
            conf_matrix_out = confusion_matrix(detect_train, detect_pred_train, normalize="all")

            # Testing confussion matrix
            detect_pred_test = cross_val_predict(bin_clf, data_test, detect_test, cv=5)
            conf_matrix_test = confusion_matrix(detect_test, detect_pred_test, normalize="all")
            TN, FP, FN, TP = conf_matrix_test.ravel()

            incorrect_predictions = detect_pred_test != detect_test
            fp_mask = (detect_pred_test == 1) & incorrect_predictions
            idcs_FP = np.where(fp_mask)[0]

            fn_mask = (detect_pred_test == 0) & incorrect_predictions
            idcs_FN = np.where(fn_mask)[0]

            # Precision and recall:
            pres = precision_score(detect_test, detect_pred_test)
            recall = recall_score(detect_test, detect_pred_test)

            # Compare prediction and tests
            idcs_fail = detect_test != detect_pred_test
            x_fail = db_test['sigma_lambda_ratio'].to_numpy()
            y_fail = db_test['amp_noise_ratio'].to_numpy()

            # Plot the sample
            x_ratios = database_df['sigma_lambda_ratio'].to_numpy()
            y_ratios = database_df['amp_noise_ratio'].to_numpy()
            detect_values = line_bool_arr.astype(bool)

            x_detection = np.linspace(x_ratios.min(), x_ratios.max(), 100)
            y_detection = detection_function(x_detection)

            with rc_context(STANDARD_PLOT):

                fig, ax = plt.subplots()

                ax.axvline(0.3, label='Cosmic ray boundary', linestyle='--', color='purple')

                ax.axvline(box_limit, label='Selection limit', linestyle='--', color='orange')

                ax.plot(x_detection, y_detection, color='black', label='Detection boundary')

                ax.scatter(x_ratios[detect_values], y_ratios[detect_values], color='palegreen', label='Positive detection')
                ax.scatter(x_ratios[~detect_values], y_ratios[~detect_values], color='xkcd:salmon', label='Negative detection')

                ax.scatter(x_fail[fn_mask], y_fail[fn_mask], color='black', label=f'FN (Recall ={recall:.2f})', alpha=0.1)
                ax.scatter(x_fail[fp_mask], y_fail[fp_mask], color='blue', label=f'FP (Precision ={pres:.2f})', alpha=0.1)

                ax.axvspan(0.10, 3.60, alpha=0.2, color='tab:blue', label='DESI range')

                # Confusion matrix
                inset_ax = ax.inset_axes([0.70, 0.70, 0.2, 0.2]) #[x, y, width, height] in fractions of fig size
                cax = inset_ax.matshow(conf_matrix_test, cmap='Blues')

                for (i, j), val in np.ndenumerate(conf_matrix_test):
                    inset_ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=12)

                # Setting up the labels for x and y axis
                inset_ax.set_xticks([0, 1])
                inset_ax.set_yticks([0, 1])
                inset_ax.set_xticklabels(['Pred Neg', 'Pred Pos'], fontsize=12)
                inset_ax.set_yticklabels(['Act Neg', 'Act Pos'], fontsize=12)

                title = f'Grid size = {detect_values.size} points ({(detect_values.sum() / detect_values.size) * 100:0.1f} % True)'

                ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}}$',
                           'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$',
                           'title': title})

                ax.set_yscale('log')
                ax.legend(loc='lower right')

                plt.tight_layout()
                #plt.show()

                plot_results = f'{subfolder}/_{data_type}_{label}_{version}_{model_name}_plot_results.png'
                plt.savefig(plot_results)

    return

def normalization_1d(y, log_base):

    y_norm = np.emath.logn(log_base, y - y.min() + 1)

    return y_norm


class TrainingSampleScaler:

    def __init__(self, cfg_file):

        # Load the parameters
        sample_params = load_cfg(cfg_file)
        self.version = sample_params['data_labels']['version']
        self.scale = sample_params['data_labels']['scale']
        self.sample_prefix = sample_params['data_labels']['sample_prefix']
        self.data_folder = Path(sample_params['data_labels']['output_folder'])/self.version

        self.cfg = sample_params[f'training_data_{self.version}']

        # Config
        self.res_limits = [self.cfg["res-ratio_min"], self.cfg["box_pixels"]/self.cfg["n_sigma"]]

        # Load the database
        database_file = self.data_folder/f'{self.sample_prefix}_{self.version}.csv'
        print(f'Loading training database at: {database_file}')
        self.sample_db = pd.read_csv(database_file)
        print('-- complete')

        # Slice the data
        self.type_array = self.sample_db["spectral_number"].to_numpy(int)
        self.int_ratio, self.res_ratio = self.sample_db['int_ratio'].to_numpy(), self.sample_db['res_ratio'].to_numpy()

        # Container for the image version
        self.image_db = None

        # Shuffle the data
        self.shuffle_databases()

        assert self.sample_db.iloc[:, 3:].columns.size == self.cfg["box_pixels"], (f'The configuration pixel number '
                                                                                   f'(self.cfg["box_pixels"]) is different '
                                                                                   f'from the number of pixel columns '
                                                                                   f'({self.sample_db.iloc[:, 3:].columns.size})')

        return

    def biBox_1D_sample(self, output_folder, limit_box, box_size, norm_base, approximation=None, label=None):

        # Set target region positive
        idcs_detect = (self.params_db.sigma_lambda_ratio < limit_box) & (self.detect_array)
        self.params_db.loc[idcs_detect, 'detection'] = True
        self.params_db.loc[~idcs_detect, 'detection'] = False
        self.detect_array[idcs_detect] = True
        self.detect_array[~idcs_detect] = False

        # Crop to an even number of true and false cases
        idcs_detect = self.detect_array == 1
        if (idcs_detect).sum() > (~idcs_detect).sum():
            idcs_false = self.detect_array == 0
            idcs_true_crop =  np.where(~idcs_false)[0][:idcs_false.sum()]
            idcs_true = np.zeros_like(idcs_false)
            idcs_true[idcs_true_crop] = True
        else:
            idcs_true = self.detect_array == 1
            idcs_false_crop =  np.where(~idcs_true)[0][:idcs_true.sum()]
            idcs_false = np.zeros_like(idcs_true)
            idcs_false[idcs_false_crop] = True

        idcs_sample = idcs_true | idcs_false

        self.params_db = self.params_db.loc[idcs_sample]
        self.params_db = self.params_db.reset_index(drop=True)

        self.wave_db = self.wave_db.loc[idcs_sample]
        self.wave_db = self.wave_db.reset_index(drop=True)

        self.flux_db = self.flux_db.loc[idcs_sample]
        self.flux_db = self.flux_db.reset_index(drop=True)

        self.detect_array = self.detect_array[idcs_sample]

        # Crop the grid
        mu_idx = self.params_db['mu_index'].unique()
        assert mu_idx.size == 1
        mu_idx = mu_idx[0]

        start_index = int(max(0, mu_idx - box_size // 2))
        end_index = int(start_index + box_size)

        self.wave_db = self.wave_db.iloc[:, start_index:end_index]
        self.flux_db = self.flux_db.iloc[:, start_index:end_index]

        # Normalize the data
        array_1D = self.flux_db.to_numpy()
        array_1D = self.normalization_1d(array_1D, norm_base)
        self.flux_db = pd.DataFrame(data=array_1D, columns=self.flux_db.columns)

        # Generate the 2D data
        if approximation is not None:
            approximation = np.atleast_1d(approximation)
            array_2D = np.tile(array_1D[:, None, :], (1, approximation.size, 1))
            array_2D = array_2D > approximation[::-1, None]
            array_2D = array_2D.astype(int)
            array_2D = array_2D.reshape((array_1D.shape[0], 1, -1))
            array_2D = array_2D.squeeze()

            hdrs = np.full(approximation.size * box_size, 'Pixel')
            hdrs = np.char.add(hdrs, np.arange(approximation.size * box_size).astype(str))
            self.image_db = pd.DataFrame(data=array_2D, columns=hdrs)

        # Reshuffle
        self.shuffle_databases()

        # Save the 1D
        subfolder = Path(output_folder)/f'_results_{label}'
        subfolder.mkdir(parents=True, exist_ok=True)
        self.wave_db.to_csv(f'{subfolder}/sample_wave_{label}.csv', index=False)
        self.flux_db.to_csv(f'{subfolder}/sample_flux_{label}.csv', index=False)
        np.savetxt(f'{subfolder}/sample_detection_{label}.csv', self.detect_array, fmt='%i')
        save_log(self.params_db, f'{subfolder}/database_{label}.csv')

        # Save the 2D array
        if approximation is not None:
            self.image_db.to_csv(f'{subfolder}/sample_image_{label}.csv', index=False)

        # Save the plot with the data points
        self.plot_training_sample(subfolder, label, limit_box)

        return


    def run_scale(self):

        # Normalize the data
        array_1D = feature_scaling(self.sample_db.iloc[:,3:].to_numpy(), transformation=self.scale)
        self.sample_db.iloc[:, 3:] = array_1D

        # Generate the 2D data
        approximation = self.cfg.get('conversion_array_min_max')
        if approximation is not None:
            approximation = np.atleast_1d(approximation)
            array_2D = np.tile(array_1D[:, None, :], (1, approximation.size, 1))
            array_2D = array_2D > approximation[::-1, None]
            array_2D = array_2D.astype(int)
            array_2D = array_2D.reshape((array_1D.shape[0], 1, -1))
            array_2D = array_2D.squeeze()

            hdrs = np.full(approximation.size * self.cfg["box_pixels"], 'Pixel')
            hdrs = np.char.add(hdrs, np.arange(approximation.size * self.cfg["box_pixels"]).astype(str))

            self.image_db = pd.DataFrame(data=array_2D, columns=hdrs)
            self.image_db.insert(loc=0, column=f'spectral_number', value=self.type_array)
            self.image_db.insert(loc=1, column=f'int_ratio', value=self.int_ratio)
            self.image_db.insert(loc=2, column=f'res_ratio', value=self.res_ratio)

        # Reshuffle
        # self.shuffle_databases()

        # Save the 1D
        self.sample_db.to_csv(self.data_folder/f'{self.scale}_{self.sample_prefix}_{self.version}.txt')

        # Save the 2D array
        if self.image_db is not None:
            print('\nSaving 2D database')
            self.image_db.to_csv(self.data_folder / f'{self.scale}_{self.sample_prefix}_{self.version}_image.txt')
            print('-- saved')

        return

    def biBox_1D_sample_log_min_max(self, output_folder, limit_box, box_size, norm_base, approximation=None, label=None):

        # Set target region positive
        idcs_detect = (self.params_db.sigma_lambda_ratio < limit_box) & (self.detect_array)
        self.params_db.loc[idcs_detect, 'detection'] = True
        self.params_db.loc[~idcs_detect, 'detection'] = False
        self.detect_array[idcs_detect] = True
        self.detect_array[~idcs_detect] = False

        # Crop to an even number of true and false cases
        idcs_detect = self.detect_array == 1
        if (idcs_detect).sum() > (~idcs_detect).sum():
            idcs_false = self.detect_array == 0
            idcs_true_crop =  np.where(~idcs_false)[0][:idcs_false.sum()]
            idcs_true = np.zeros_like(idcs_false)
            idcs_true[idcs_true_crop] = True
        else:
            idcs_true = self.detect_array == 1
            idcs_false_crop =  np.where(~idcs_true)[0][:idcs_true.sum()]
            idcs_false = np.zeros_like(idcs_true)
            idcs_false[idcs_false_crop] = True

        idcs_sample = idcs_true | idcs_false

        self.params_db = self.params_db.loc[idcs_sample]
        self.params_db = self.params_db.reset_index(drop=True)

        self.wave_db = self.wave_db.loc[idcs_sample]
        self.wave_db = self.wave_db.reset_index(drop=True)

        self.flux_db = self.flux_db.loc[idcs_sample]
        self.flux_db = self.flux_db.reset_index(drop=True)

        self.detect_array = self.detect_array[idcs_sample]

        # Crop the grid
        mu_idx = self.params_db['mu_index'].unique()
        assert mu_idx.size == 1
        mu_idx = mu_idx[0]

        start_index = int(max(0, mu_idx - box_size // 2))
        end_index = int(start_index + box_size)

        self.wave_db = self.wave_db.iloc[:, start_index:end_index]
        self.flux_db = self.flux_db.iloc[:, start_index:end_index]

        # Normalize the data
        array_1D = self.flux_db.to_numpy()
        array_1D = feature_scaling(array_1D, transformation='log-min-max', log_base=norm_base)
        self.flux_db = pd.DataFrame(data=array_1D, columns=self.flux_db.columns)

        # Generate the 2D data
        if approximation is not None:
            approximation = np.atleast_1d(approximation)
            array_2D = np.tile(array_1D[:, None, :], (1, approximation.size, 1))
            array_2D = array_2D > approximation[::-1, None]
            array_2D = array_2D.astype(int)
            array_2D = array_2D.reshape((array_1D.shape[0], 1, -1))
            array_2D = array_2D.squeeze()

            hdrs = np.full(approximation.size * box_size, 'Pixel')
            hdrs = np.char.add(hdrs, np.arange(approximation.size * box_size).astype(str))
            self.image_db = pd.DataFrame(data=array_2D, columns=hdrs)

        # Reshuffle
        self.shuffle_databases()

        # Save the 1D
        subfolder = Path(output_folder)/f'_results_{label}'
        subfolder.mkdir(parents=True, exist_ok=True)
        self.wave_db.to_csv(f'{subfolder}/sample_wave_{label}.csv', index=False)
        self.flux_db.to_csv(f'{subfolder}/sample_flux_{label}.csv', index=False)
        np.savetxt(f'{subfolder}/sample_detection_{label}.csv', self.detect_array, fmt='%i')
        save_log(self.params_db, f'{subfolder}/database_{label}.csv')

        # Save the 2D array
        if approximation is not None:
            self.image_db.to_csv(f'{subfolder}/sample_image_{label}.csv', index=False)

        # Save the plot with the data points
        self.plot_training_sample(subfolder, label, limit_box)

        return

    def normalization_1d(self, y, log_base):

        y_cont = y - y.min(axis=1, keepdims=True) + 1
        y_norm = np.emath.logn(log_base, y_cont)

        return y_norm

    def shuffle_databases(self, shuffle_seed=42):

        idcs_shuffle = np.random.default_rng(seed=shuffle_seed).permutation(np.arange(self.type_array.size))

        self.sample_db = self.sample_db.loc[idcs_shuffle]
        self.sample_db = self.sample_db.reset_index(drop=True)

        self.type_array = self.type_array[idcs_shuffle]

        self.int_ratio = self.int_ratio[idcs_shuffle]
        self.res_ratio = self.res_ratio[idcs_shuffle]

        if self.image_db is not None:
            self.image_db = self.image_db.loc[idcs_shuffle]
            self.image_db = self.image_db.reset_index(drop=True)

        print('Shuffling data')

        # # Slice the data
        # self.type_array
        # self.int_ratio, self.res_ratio

        return

    def plot_training_sample(self, output_folder, label, limit_box=None, idcs_plot=None):

        # Plot the sample
        idcs_plot = idcs_plot if idcs_plot is not None else self.params_db.index
        x_ratios = self.params_db.loc[idcs_plot, 'sigma_lambda_ratio'].to_numpy()
        y_ratios = self.params_db.loc[idcs_plot, 'amp_noise_ratio'].to_numpy()
        detect_values = self.detect_array[idcs_plot]

        x_detection = np.linspace(self.res_limits[0],  self.res_limits[1], 100)
        y_detection = detection_function(x_detection)

        with rc_context(STANDARD_PLOT):
            fig, ax = plt.subplots()

            ax.axvline(0.3, label='Cosmic ray boundary', linestyle='--', color='purple')

            ax.axvline(limit_box, label='Selection limit', linestyle='--', color='orange')

            ax.plot(x_detection, y_detection, color='black', label='Detection boundary')

            # ax.scatter(x_ratios, y_ratios)
            ax.scatter(x_ratios[detect_values], y_ratios[detect_values], color='palegreen', label='Positive detection')
            ax.scatter(x_ratios[~detect_values], y_ratios[~detect_values], color='xkcd:salmon',
                       label='Negative detection')

            # Desi range
            ax.axvspan(0.10, 3.60, alpha=0.2, color='tab:blue', label='DESI range')

            # fraction_text = r'$\textcolor{{{}}}{{{}}}\,\textcolor{{{}}}{{{}}}$'.format('palegreen',
            #                                                                             x_ratios[detect_values].size,
            #                                                                             'xkcd:salmon',
            #                                                                             x_ratios[~detect_values].size)


            title = f'Grid size = {detect_values.size} points ({(detect_values.sum()/detect_values.size)*100:0.1f} % True)'

            ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}}$',
                       'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$',
                       'title': title})

            ax.set_yscale('log')
            ax.legend(loc='lower right')

            plt.tight_layout()
            # plt.show()
            plt.savefig(output_folder/f'{label}_training_sample.png')

        return