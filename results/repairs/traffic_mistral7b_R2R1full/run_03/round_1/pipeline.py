import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow.pyfunc import PyFuncModel

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data['Features'], data['Congestion'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        model_artifact_path = mlflow.pyfunc.save_model(model, "model")
        mlflow.pyfunc.log_model(model_artifact_path)
        mlflow.log_param("num_trees", model.n_estimators)
