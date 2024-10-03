import lime
from pathlib import Path
from training import run_loaddb_and_training


# Read sample configuration
cfg_file = 'training_sample_v3.toml'
sample_params = lime.load_cfg(cfg_file)

version = sample_params['data_labels']['version']
scale = sample_params['data_labels']['scale']
sample_prefix = sample_params['data_labels']['sample_prefix']

# Read the sample database
data_folder = Path(sample_params['data_labels']['output_folder'])/version
sample1D_database_file = data_folder/f'{sample_prefix}_{version}_{scale}.csv'

# Run the training
label = '6categories_v2_80000points'
run_loaddb_and_training(sample1D_database_file, sample_params, label, review_sample=False)
