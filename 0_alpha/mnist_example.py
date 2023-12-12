# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"
from sklearn.datasets import fetch_openml
import sklearn
assert sklearn.__version__ >= "0.20"
from sklearn.datasets import fetch_openml
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

# Is this notebook running on Colab or Kaggle?
IS_COLAB = "google.colab" in sys.modules
IS_KAGGLE = "kaggle_secrets" in sys.modules

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = ".."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print(f'Database keys: {mnist.keys()}')

# mnist_df = fetch_openml('mnist_784', version=1)

# The number images are stored 1D arrays
X, y = mnist["data"], mnist["target"]

# Changing the label type to a number (the same as the image number)
y = y.astype(np.uint8)

# The first entry is a 5
image_0 = X[0].reshape(28, 28)
plt.imshow(image_0, cmap=mpl.cm.binary)
plt.title(f'Dataset label: {y[0]}')
plt.show()


# Tratining and testing sets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Just those with the number 5
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# Binary classifier
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)

# Use the trained model
sgd_clf.predict([X[0]])

# Random forest
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")