# pipeline.py
import mlflow
import pandas as pd
import numpy as np
import sklearn.ensemble as rf
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data, data['Amount'])

def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(l) for l in lengths]
        )

def _num_samples(X):
    return len(X) if isinstance(X, (list, tuple, pd.DataFrame)) else X.shape[0]

def train_model(X_train, y_train):
    with mlflow.start_run():
        try:
            model = rf.RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            mlflow.log_artifact(f'model_{mlflow.active_run().info.run_id}.pkl', artifact_location='models')
            return model
        except ConvergenceWarning:
            mlflow.log_warning('ConvergenceWarning during model training.')
