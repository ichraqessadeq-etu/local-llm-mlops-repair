import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise MlflowException(f"File {file_path} not found.")
    except pd.errors.ParserError:
        raise MlflowException("Error parsing CSV file.")

def preprocess_data(data):
    try:
        # Define features and target
        features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']
        target = 'Congestion'
        
        # Convert categorical variables to numerical
        data['weather_main'] = data['weather_main'].map({'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Snow': 3})
        
        # Split data into features and target
        X = data[features]
        y = data[target]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise MlflowException(f"Preprocessing error: {e}")

def train_model(X_train, y_train):
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise MlflowException(f"Model training error: {e}")

def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    except Exception as e:
        raise MlflowException(f"Evaluation error: {e}")
