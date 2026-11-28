import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError(f'File {file_path} not found.')
    except pd.errors.EmptyDataError:
        raise ValueError(f'File {file_path} is empty.')

def preprocess_data(data):
    """Preprocess the data."""
    data = data.dropna()
    data['weather_main'] = data['weather_main'].astype('category').cat.codes
    X = data.drop('Congestion', axis=1)
    y = data['Congestion']
    return (X, y)

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, 'random_forest_model')
            y_pred = model.predict(X_train)
            accuracy = accuracy_score(y_train, y_pred)
            mlflow.log_metric('accuracy', accuracy)
            return model
    except Exception as e:
        mlflow.log_exception(e)
        raise e