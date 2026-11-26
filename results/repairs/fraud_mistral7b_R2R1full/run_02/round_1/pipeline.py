import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import PyFuncModel, log_artifact, log_model

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    X = data.drop('Class', axis=1)
    y = data['Class']
    return (X, y)

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, model_name):
    with PyFuncModel.create_context() as ctx:
        model = PyFuncModel(model, input_schema=ctx.get_schema(X_train), output_schema=ctx.get_schema(y_train))
        log_model(model, 'model', registered_model_name=model_name)
        log_artifact(model.model_uri, f'models/{model_name}')
