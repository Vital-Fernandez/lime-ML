import lime
import joblib
from pathlib import Path
from model_tools import read_sample_database, stratified_train_test_split
from plots import diagnostics_plot
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

# Read sample configuration
cfg_file = 'training_sample_v3.toml'
sample_params = lime.load_cfg(cfg_file)

version = sample_params['data_labels']['version']
scale = sample_params['data_labels']['scale']
sample_prefix = sample_params['data_labels']['sample_prefix']

# Load the configuration
label_model = 'training_data_v3'
data_cfg = sample_params[label_model]

# Read the sample database
data_folder = Path(sample_params['data_labels']['output_folder'])/version
sample1D_database_file = data_folder/f'{sample_prefix}_{version}_{scale}.csv'
results_folder = Path(sample1D_database_file).parent / 'results'

# Load the training sample
db_df = read_sample_database(sample1D_database_file, data_cfg)

# Prepare training and testing sets
label_fit = '5categories_v2_80000points'
fit_cfg = sample_params[label_fit]
df_train, df_test = stratified_train_test_split(db_df, fit_cfg['categories'], fit_cfg['sample_size'],
                                                test_size=fit_cfg['test_sample_size_fraction'])

# Load the model
model_address = results_folder/f'{fit_cfg["sample_prefix"]}_{fit_cfg["version"]}_{fit_cfg["scale"]}_{label_fit}_model.joblib'
ml_function = joblib.load(model_address)

# Get predictions on test data
x_test, y_test = df_test.iloc[:, 3:], df_test.iloc[:, 0]
y_pred = ml_function.predict(x_test)


# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(cm.shape)
plt.figure(figsize=(7,7))
sns.heatmap(cm, annot=True, fmt='d', cbar=False, xticklabels=fit_cfg['categories'],
            yticklabels=fit_cfg['categories'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Multiclass Confusion Matrix')
plt.show()

# # Load the model
# model_address = results_folder/f'{fit_cfg["sample_prefix"]}_{fit_cfg["version"]}_{fit_cfg["scale"]}_{label_fit}_model.joblib'
# ml_function = joblib.load(model_address)
#
#
#
# # Get predictions on test data
# x_test, y_test = df_test.iloc[:, 3:], df_test.iloc[:, 0]
# pred_test = ml_function.predict(x_test)
#
# idcs_missmatch = pred_test != y_test
# missmatch_df = df_test.loc[idcs_missmatch]
# missmatch_df['pred_values'] = pred_test[idcs_missmatch]
#
# # Get the diagnostics:
# pres = precision_score(y_test, pred_test, average='weighted')
# recall = recall_score(y_test, pred_test, average='weighted')
# f1 = f1_score(y_test, pred_test, average='weighted')
#
# ax_cfg = {'title': f'Precision: {pres:0.2f}, Recall: {recall:0.2f}, F1 score: {f1:0.2f} ({idcs_missmatch.sum()}/{y_test.size})'}
#
# # Display plot
# categories = sample_params[label_fit]['categories']
# diagnostics_plot(sample_params[label_model], categories=categories, missmatch_df=missmatch_df, ax_cfg=ax_cfg)
