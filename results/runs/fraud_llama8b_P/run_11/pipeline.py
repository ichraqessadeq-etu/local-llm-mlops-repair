import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow.sklearn
import mlflow

def load_data(file_path):
    """Load data from CSV file"""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        mlflow.log_error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocess data by splitting into features and target, and scaling features"""
    try:
        X = data.drop(['Class'], axis=1)  # features
        y = data['Class']  # target
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        mlflow.log_error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train a RandomForest classifier"""
    try:
        with mlflow.start_run():
            mlflow.set_tag("model_type", "RandomForest")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 5)
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "random_forest")
            return model
    except Exception as e:
        mlflow.log_error(f"Error training model: {e}")
        raise
