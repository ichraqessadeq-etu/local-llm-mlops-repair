import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError(f"File {file_path} not found.")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty or does not contain any data.")

def preprocess_data(data):
    """Preprocess the dataset."""
    try:
        # Assuming 'Class' is the target column
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except KeyError:
        raise ValueError("The target column 'Class' is not found in the dataset.")

def train_model(X_train, y_train):
    """Train the RandomForest model."""
    try:
        # Initialize the model
        model = RandomForestClassifier(random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        return model
    except Exception as e:
        raise Exception(f"An error occurred while training the model: {e}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.2f}")
        
        return accuracy
    except Exception as e:
        raise Exception(f"An error occurred while evaluating the model: {e}")
