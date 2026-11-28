import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import PyFuncModel, track, experiment

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Perform necessary preprocessing like filling missing values, encoding categorical variables, etc.
    # For simplicity, I'll assume no preprocessing is required.
    return data['Features'], data['Class']

def train_model(X_train, y_train):
    with experiment('Fraud_Detection_Experiment'):
        track('Features', X_train.columns)
        track('Target', y_train)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        model_artifact_path = f'models/random_forest_{str(int(time.time()))}.pkl'
        mlflow.sklearn.save_model(model, model_artifact_path)
        return PyFuncModel.from_sklearn(model, 'predict')
