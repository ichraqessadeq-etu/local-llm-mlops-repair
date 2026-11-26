import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f'Error loading data: {e}')

def preprocess_data(data):
    """Preprocess data to split into features (X) and target (y)."""
    try:
        # Encode categorical variables
        encoder = LabelEncoder()
        data['weather_main'] = encoder.fit_transform(data['weather_main'])
        data['IsHoliday'] = encoder.fit_transform(data['IsHoliday'])
        # Split into features (X) and target (y)
        X = data.drop(columns=['Congestion'])
        y = data['Congestion']
        return (X, y)
    except Exception as e:
        raise Exception(f'Error preprocessing data: {e}')

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    try:
        with mlflow.start_run():
            mlflow.log_param('model', 'RandomForestClassifier')
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, 'model')
            mlflow.log_metric('accuracy', model.score(X_train, y_train))
            return model
    except Exception as e:
        raise Exception(f'Error training model: {e}')
