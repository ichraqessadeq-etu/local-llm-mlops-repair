import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, track, pyfunc, run_local

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps like filling missing values, normalization, etc.
    # For simplicity, I'm assuming there are no missing values in the dataset
    return data.drop('Amount', axis=1), data['Amount']

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    mlflow_model = pyfunc.PyFuncTrackingEstimator(
        module=__name__,
        init_code="""
            import pandas as pd
            import mlflow.pyfunc
            import numpy as np
            def init(model):
                return model
            def predict(model, X):
                return model.predict(X)
        """,
        source_code="""
            def predict(X):
                return model.predict(X)
        """,
        init=init_code,
        source_code=source_code,
        model=model,
    )
    mlflow.log_artifact(artifact=mlflow_model, artifact_path="models")
    track(mlflow_model.model_name)
    return mlflow_model
