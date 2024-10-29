import numpy as np
import pandas as pd
import lime
from pathlib import Path
from model_tools import TrainingSampleScaler


# Read sample configuration
cfg_file = 'training_sample_v4.toml'

# Small box training with min max
samplGen = TrainingSampleScaler(cfg_file)
samplGen.run_scale()
