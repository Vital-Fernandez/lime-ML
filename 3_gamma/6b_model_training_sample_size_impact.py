import lime
from pathlib import Path
from training import run_training, save_model
from model_tools import read_sample_database, stratified_train_test_split
from plots import SampleReviewer


# Read sample configuration
cfg_file = 'training_sample_v3_old.toml'
sample_params = lime.load_cfg(cfg_file)

version = sample_params['data_labels']['version']
scale = sample_params['data_labels']['scale']
sample_prefix = sample_params['data_labels']['sample_prefix']

# Run the training
label = '2categories_v2'
# samples_size_list = [70000]
samples_size_list = [5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000,
                     100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000, 180000, 190000, 200000, 210000,
                     220000, 230000, 240000, 250000]

fit_cfg = sample_params[label]
estimator_id = (fit_cfg['estimator']['module'], fit_cfg['estimator']['class'])

# Read the sample database
data_folder = Path(sample_params['data_labels']['output_folder'])/version
sample1D_database_file = data_folder/f'{sample_prefix}_{version}_{scale}.csv'
db_df = read_sample_database(sample1D_database_file, fit_cfg)

for i, n_samples in enumerate(samples_size_list):

    # Prepare training and testing sets
    print(f'\n{i}) Configuration 1')
    df_train, df_test = stratified_train_test_split(db_df, fit_cfg['categories'], n_samples,
                                                    test_size=fit_cfg['test_sample_size_fraction'])
    x, y = df_train.iloc[:, 3:], df_train.iloc[:, 0]

    # diag = SampleReviewer(df_train, color_dict=sample_params['colors'])
    # diag.interactive_plot()

    # Run training
    label_i = f'{label}_nSamples{n_samples}'
    model = run_training(x, y, estimator_id, fit_cfg['estimator_params'], label_i)

    # Save the results
    x, y = df_test.iloc[:, 3:], df_test.iloc[:, 0]
    output_root = f'{data_folder}/results/sampleSizeTests_{label_i}'
    save_model(model, x, y, output_root, label_i, fit_cfg)



