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
        raise ValueError("File not found: {}".format(file_path))
    except pd.errors.EmptyDataError:
        raise ValueError("Empty file: {}".format(file_path))

def preprocess_data(data):
    """Preprocess data by splitting features and target, and splitting into training and test sets."""
    try:
        features = data.drop('Congestion', axis=1)
        target = data['Congestion']
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise ValueError("Error preprocessing data: {}".format(e))

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    try:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise ValueError("Error training model: {}".format(e))

def predict(model, X_test):
    """Predict congestion using the trained model."""
    try:
        predictions = model.predict(X_test)
        return predictions
    except Exception as e:
        raise ValueError("Error predicting congestion: {}".format(e))

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's accuracy."""
    try:
        predictions = predict(model, X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy
    except Exception as e:
        raise ValueError("Error evaluating model: {}".format(e))

def run_pipeline(file_path):
    """Run the entire pipeline."""
    try:
        data = load_data(file_path)
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        print("Model accuracy:", accuracy)
        return model
    except Exception as e:
        print("Error running pipeline:", e)
        return None

if __name__ == "__main__":
    file_path = 'data/traffic.csv'
    run_pipeline(file_path)
