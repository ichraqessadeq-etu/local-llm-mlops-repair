import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("The file is empty or does not contain any data.")

def preprocess_data(data):
    """Preprocess the data by splitting into features and target, and scaling the features."""
    try:
        # Convert 'rain_1h' and 'snow_1h' to numeric
        data['rain_1h'] = pd.to_numeric(data['rain_1h'], errors='coerce')
        data['snow_1h'] = pd.to_numeric(data['snow_1h'], errors='coerce')

        # Drop rows with NaN values in 'rain_1h' or 'snow_1h'
        data = data.dropna(subset=['rain_1h', 'snow_1h'])

        # Convert 'weather_main' to numeric if possible
        data['weather_main'] = pd.to_numeric(data['weather_main'], errors='coerce')

        # Drop rows with NaN values in 'weather_main'
        data = data.dropna(subset=['weather_main'])

        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise Exception(f"Error preprocessing data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForest model and log the model to MLflow."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "random_forest_model")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            return model
    except Exception as e:
        raise Exception(f"Error training model: {e}")
