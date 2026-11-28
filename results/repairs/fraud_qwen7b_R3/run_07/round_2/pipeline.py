import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data by separating features and target."""
    X = data.drop(columns=['Class'])
    y = data['Class']
    return (X, y)

def train_model(X_train, y_train):
    """Train a RandomForest model and log it using MLflow."""
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, 'random_forest_model')
        mlflow.log_param('n_estimators', 100)
        mlflow.log_param('max_depth', None)
        mlflow.log_metric('accuracy', accuracy_score(y_train, model.predict(X_train)))
        return model