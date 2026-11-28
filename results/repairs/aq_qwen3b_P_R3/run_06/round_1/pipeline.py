import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError(f'File not found: {file_path}')
    except pd.errors.EmptyDataError:
        raise ValueError('The file is empty.')
    except pd.errors.ParserError:
        raise ValueError('Error parsing the CSV file.')

def preprocess_data(data):
    """Preprocess the data by splitting it into features and target."""
    features = data.columns.drop('Anomaly')
    X = data[features]
    y = data['Anomaly']
    return (X, y)

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, 'model')
            return model
    except Exception as e:
        pass  # R3: API inexistante retirée
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)
        return accuracy
    except Exception as e:
        pass  # R3: API inexistante retirée
        raise