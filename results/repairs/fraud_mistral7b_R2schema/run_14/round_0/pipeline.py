import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, track

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Perform data preprocessing such as handling missing values, encoding categorical variables, etc.
    # For simplicity, I'll just drop missing values and keep the original encoding.
    data.dropna(inplace=True)
    X = data.drop('Class', axis=1)
    y = data['Class']
    return X, y

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    track(model.feature_importances_)
    mlflow.sklearn.log_model(model, "model")

