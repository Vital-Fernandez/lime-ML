import numpy as np
import pandas as pd
import lime
from pathlib import Path
from model_tools import TrainingSampleScaler


# Read sample configuration
cfg_file = 'training_sample_v3.toml'

# Small box training with min max
samplGen = TrainingSampleScaler(cfg_file)
samplGen.run_scale()



# # Recover configuration data
# res_sample_size = cfg['data_grid']['res_sample_size']
# res_sample_limts = cfg['data_grid']['res_sample_limits']
# small_res_limit =  cfg['data_grid']['small_box_reslimit']
# large_res_limit =  cfg['data_grid']['large_box_reslimit']
# small_box_size = cfg['data_grid']['small_box_size']
# large_box_size = cfg['data_grid']['large_box_size']
# conversion_array_log = cfg['data_grid']['conversion_array_log']
# conversion_array_min_max = cfg['data_grid']['conversion_array_min_max']
# norm_base = cfg['data_grid']['norm_base']