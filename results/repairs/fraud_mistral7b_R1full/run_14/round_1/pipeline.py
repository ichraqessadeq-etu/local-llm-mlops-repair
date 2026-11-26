import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, tracking

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Perform data preprocessing such as handling missing values, encoding categorical variables, etc.
    # For simplicity, I'll just drop missing values and keep the original encoding.
    data.dropna(inplace=True)
    X = data.drop('Class', axis=1)
    y = data['Class']
    return X, y

def train_model(X_train, y_train, random_state=42):
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    tracking.set_global_context(experiment_id=mlflow.get_experiment_by_name("Fraud Detection").experiment_id)
    log_artifact(model, "model")

