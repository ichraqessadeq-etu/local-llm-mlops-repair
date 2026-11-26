import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ValueError, _value_error
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError("File not found.")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty.")
    except pd.errors.ParserError:
        raise ValueError("Error parsing the CSV file.")

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Convert 'IsHoliday' to binary
        data['IsHoliday'] = data['IsHoliday'].astype(int)
        
        # Encode categorical variables
        data = pd.get_dummies(data, columns=['weather_main', 'DayOfWeek', 'Month'])
        
        # Split features and target
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        return X, y
    except Exception as e:
        raise ValueError(f"Error preprocessing data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise ValueError(f"Error training model: {e}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    except Exception as e:
        raise ValueError(f"Error evaluating model: {e}")
