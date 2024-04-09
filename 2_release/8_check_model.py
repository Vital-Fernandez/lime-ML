import numpy as np
import pandas as pd
import lime
import joblib
from pathlib import Path
from tools import analysis_check

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score


# Read configuration
cfg_file = 'config_file.toml'
cfg = lime.load_cfg(cfg_file)
output_folder = Path(cfg['data_location']['output_folder'])
version = cfg['data_grid']['version']

# Recover configuration entries
version = cfg['data_grid']['version']
train_frac = cfg['data_grid']['training_fraction']
# label = 'small_box_logMinMax'
label = 'small_box_min_max'

analysis_check(cfg, version, label, output_folder)

lime.Spectrum