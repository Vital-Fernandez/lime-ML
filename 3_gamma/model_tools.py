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
        print(f'\nLoading training database at: {database_file}')
        self.sample_db = pd.read_csv(database_file)
        print('- complete')

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

    def run_scale(self):

        # Normalize the data
        array_1D = feature_scaling(self.sample_db.iloc[:,3:].to_numpy(), transformation=self.scale)
        self.sample_db.iloc[:, 3:] = array_1D

        # Generate the 2D data
        approximation = None # self.cfg.get('conversion_array_min_max')
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

        # Make plot to review the training sample
        plot_address = self.data_folder / f'{self.scale}_{self.sample_prefix}_{self.version}_diagnostic_plot.png'
        print(f'\nMaking review plot: {plot_address}')
        self.plot_training_sample(self.cfg, self.sample_db, plot_address)
        print(f'- complete')

        # Save the 1D
        database_address = self.data_folder/f'{self.scale}_{self.sample_prefix}_{self.version}.txt'
        print(f'\nSaving 1D database: {database_address}')
        self.sample_db.to_csv(database_address)
        print(f'- complete')

        # Save the 2D array
        if self.image_db is not None:
            file_address = self.data_folder / f'{self.scale}_{self.sample_prefix}_{self.version}_image.txt'
            print(f'\nSaving 2D database: {file_address}')
            self.image_db.to_csv(file_address)
            print('- saved')

        return

    def normalization_1d(self, y, log_base):

        y_cont = y - y.min(axis=1, keepdims=True) + 1
        y_norm = np.emath.logn(log_base, y_cont)

        return y_norm

    def shuffle_databases(self, shuffle_seed=42):

        print('\nShuffling data')

        idcs_shuffle = np.random.default_rng(seed=shuffle_seed).permutation(np.arange(self.type_array.size))

        self.sample_db = self.sample_db.loc[idcs_shuffle]
        self.sample_db = self.sample_db.reset_index(drop=True)

        self.type_array = self.type_array[idcs_shuffle]

        self.int_ratio = self.int_ratio[idcs_shuffle]
        self.res_ratio = self.res_ratio[idcs_shuffle]

        if self.image_db is not None:
            self.image_db = self.image_db.loc[idcs_shuffle]
            self.image_db = self.image_db.reset_index(drop=True)

        print('- complete')

        # # Slice the data
        # self.type_array
        # self.int_ratio, self.res_ratio

        return

    def plot_training_sample(self, cfg, database_df, output_address):

        # Figure format
        fig_cfg = theme.fig_defaults({'axes.labelsize': 10,
                                      'axes.titlesize': 10,
                                      'figure.figsize': (3, 3),
                                      'hatch.linewidth': 0.3,
                                      "legend.fontsize": 8})

        # Creat the figure
        n_points = 500
        with rc_context(fig_cfg):

            # Loop throught the sample and generate the figures
            fig, ax = plt.subplots()
            for label_feature, number_feature in cfg['classes'].items():
                if number_feature > 0:

                    # Filter the DataFrame by the category
                    idcs_feature = database_df['spectral_number'] == number_feature

                    if idcs_feature.sum() > 0:
                        feature_df = database_df.loc[database_df['spectral_number'] == number_feature].sample(n=n_points)
                        # feature_df = database_df.iloc[:500, :]

                        int_ratio = feature_df.loc[:, 'int_ratio'].to_numpy()
                        res_ratio = feature_df.loc[:, 'res_ratio'].to_numpy()
                        color = cfg[label_feature]['color']

                        ax.scatter(res_ratio, int_ratio, color=color, label=label_feature, alpha=0.5, edgecolor='none')

                        # Narrow component case
                        if label_feature == 'broad':
                            feature_df = database_df.loc[database_df['spectral_number'] == number_feature + 0.5].sample(n=n_points)

                            int_ratio = feature_df.loc[:, 'int_ratio'].to_numpy()
                            res_ratio = feature_df.loc[:, 'res_ratio'].to_numpy()
                            color = cfg[label_feature]['color']

                            ax.scatter(res_ratio, int_ratio, marker='x', color='black', label='narrow', alpha=1)

            # Wording
            ax.update(
                {'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}} = \sigma_{pixels}$ (Gaussian sigma in pixels)',
                 'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$ (Signal-to-noise)'})
            ax.legend(loc='lower center', ncol=2, framealpha=0.95)

            # Axis format
            ax.set_yscale('log')
            # ax.set_xlim(0, 10)
            # ax.set_ylim(0.01, 10000)

            # Upper axis
            ax2 = ax.twiny()
            ticks_values = ax.get_xticks()
            ticks_labels = [f'{tick:.0f}' for tick in ticks_values * 6]
            ax2.set_xticks(ticks_values)  # Set the tick positions
            ax2.set_xticklabels(ticks_labels)
            ax2.set_xlabel(r'$b_{pixels}$ (detection box width in pixels)')

            # Grid
            ax.grid(axis='x', color='0.95', zorder=1)
            ax.grid(axis='y', color='0.95', zorder=1)

            plt.tight_layout()
            # plt.show()
            plt.savefig(output_address)


        return