import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Perform necessary preprocessing like filling missing values, encoding categorical variables, etc.
    # For simplicity, I'll just drop missing values.
    data.dropna(inplace=True)
    X = data.drop('Class', axis=1)
    y = data['Class']
    return X, y

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random-forest")
