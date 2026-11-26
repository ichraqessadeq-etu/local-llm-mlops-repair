import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import PyFuncModel, log_model, log_artifact, set_tracking_uri

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Add any necessary preprocessing steps here
    X = data.drop('Class', axis=1)
    y = data['Class']
    return X, y

def train_model(X_train, y_train, random_state=42):
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def set_mlflow_tracking_uri(uri):
    set_tracking_uri(uri)

def save_model(model, model_name):
    with PyFuncModel.create_context() as ctx:
        model = PyFuncModel(model, input_schema=ctx.get_schema(X_train), output_schema=ctx.get_schema(y_train))
        log_model(model, "model", registered_model_name=model_name)
        log_artifact(model.model_uri, f"models/{model_name}")
