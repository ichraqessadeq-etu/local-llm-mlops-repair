import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow import log_param, log_metric, log_model

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f"Failed to load data: {e}")

def preprocess_data(data):
    """Preprocess data for model training."""
    try:
        X = data.drop(columns=['Congestion'])
        y = data['Congestion']
        return X, y
    except Exception as e:
        raise Exception(f"Failed to preprocess data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise Exception(f"Failed to train model: {e}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    except Exception as e:
        raise Exception(f"Failed to evaluate model: {e}")

def main():
    """Main function to run the pipeline."""
    try:
        file_path = 'data/traffic.csv'
        data = load_data(file_path)
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = train_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        
        log_param("n_estimators", model.n_estimators)
        log_param("max_depth", model.max_depth)
        log_metric("accuracy", accuracy)
        log_model(model, "random_forest_model")
        
        print(f"Model trained with accuracy: {accuracy}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
