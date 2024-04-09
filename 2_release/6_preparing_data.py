import numpy as np
import pandas as pd
import lime
from pathlib import Path
from tools import TrainingSampleGenerator


# Read configuration
cfg_file = 'config_file.toml'
cfg = lime.load_cfg(cfg_file)
output_folder = Path(cfg['data_location']['output_folder'])

# Recover configuration data
res_sample_size = cfg['data_grid']['res_sample_size']
res_sample_limts = cfg['data_grid']['res_sample_limits']
small_res_limit =  cfg['data_grid']['small_box_reslimit']
large_res_limit =  cfg['data_grid']['large_box_reslimit']
small_box_size = cfg['data_grid']['small_box_size']
large_box_size = cfg['data_grid']['large_box_size']
conversion_array_log = cfg['data_grid']['conversion_array_log']
conversion_array_min_max = cfg['data_grid']['conversion_array_min_max']
norm_base = cfg['data_grid']['norm_base']

# IDs for the data set
version = cfg['data_grid']['version']


# Input output files
sample_database = output_folder/f'sample_database_{version}.txt'

# # Smmall box training
# label = 'small_box'
# samplGen = TrainingSampleGenerator(version, output_folder, sample_database)
# samplGen.biBox_1D_sample(output_folder, small_res_limit, small_box_size, norm_base,
#                          conversion_array_log, label=f'{label}_{version}')

# # Small box training with min max
# label = 'small_box_min_max'
# samplGen = TrainingSampleGenerator(version, output_folder, sample_database)
# samplGen.biBox_1D_sample_min_max(output_folder, small_res_limit, small_box_size, norm_base,
#                                  conversion_array_min_max, label=f'{label}_{version}')

# Small box training with min max
label = 'small_box_logMinMax'
samplGen = TrainingSampleGenerator(version, output_folder, sample_database)
samplGen.biBox_1D_sample_log_min_max(output_folder, small_res_limit, small_box_size, norm_base,
                                     conversion_array_min_max, label=f'{label}_{version}')
