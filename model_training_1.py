import numpy as np
import pandas as pd
import pickle
from functions import PARMS_FILE, SEED_VALUE
from sklearn.linear_model import SGDClassifier
import joblib
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

parameters_DF = pd.read_csv(PARMS_FILE)

flux_table = 'sample_flux_table.csv'
flux_data_DF = pd.read_csv(flux_table)

X, y = flux_data_DF.to_numpy(), parameters_DF['line_check'].values

# Convert boolean to float
y = y.astype(np.uint8)

# Divide into test and train sets
X_train, y_train = X[:80000], y[:80000],
X_test, y_test = X[80000:], y[80000:]

# True cases with emission line
y_train_line = (y_train == 1)
y_test_line = (y_test == 1)

# # Stochastic gradient descent classifier
# sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train_line)
#
# print(f'Entrenado, prediction of line 1: {sgd_clf.predict([X_train[0]])}')
#
# # Saving to a file
# filename = 'sgd_v1.pickle'
# pickle.dump(sgd_clf, open(filename, 'wb'))
#
# filename = 'sgd_v1.joblib'
# joblib.dump(sgd_clf, filename)

# Load the model
filename = 'sgd_v1.joblib'
sgd_clf = joblib.load(filename)
print(f'Loaded model, prediction for line 1: {sgd_clf.predict([X_train[0]])}')

# ------------------- Testing the accuracy
tests_3_folds = cross_val_score(sgd_clf, X_train, y_train_line, cv=3, scoring="accuracy")
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_line, cv=3)

conf_matrix = confusion_matrix(y_train_line, y_train_pred)
print(conf_matrix)

# Perfect analysis
y_train_perfect_predictions = y_train_line
print(confusion_matrix(y_train_line, y_train_perfect_predictions))

# Precision and recall:
pres = precision_score(y_train_line, y_train_pred)
recall = recall_score(y_train_line, y_train_pred)

# Compare with the test case
prediction_test = sgd_clf.predict(X_test)


tests_3_folds = cross_val_score(sgd_clf, X_test, y_test_line, cv=3, scoring="accuracy")
y_train_pred = cross_val_predict(sgd_clf, X_test, y_test_line, cv=3)

conf_matrix = confusion_matrix(y_test_line, y_train_pred)
print(conf_matrix)

# # some time later...
#
# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)
#
# model = LogisticRegression()
# model.fit(X_train, Y_train)
# # save the model to disk
# filename = 'finalized_model.sav'
# joblib.dump(model, filename)
#
# # some time later...
#
# # load the model from disk
# loaded_model = joblib.load(filename)
# result = loaded_model.score(X_test, Y_test)
# print(result)
