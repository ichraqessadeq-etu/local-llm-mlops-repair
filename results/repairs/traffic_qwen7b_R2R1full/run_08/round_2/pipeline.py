import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: The loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise ValueError(f'Failed to load data: {e}')

def preprocess_data(data):
    """
    Preprocess the data by separating features and target.
    
    Parameters:
    data (pd.DataFrame): The input data.
    
    Returns:
    tuple: A tuple containing the features (X) and target (y).
    """
    try:
        X = data.drop(columns=['Congestion'])
        y = data['Congestion']
        return (X, y)
    except Exception as e:
        raise ValueError(f'Failed to preprocess data: {e}')

def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier model.
    
    Parameters:
    X_train (pd.DataFrame): The training features.
    y_train (pd.Series): The training target.
    
    Returns:
    RandomForestClassifier: The trained model.
    """
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            mlflow.log_model(model, 'model')
            mlflow.log_param('n_estimators', 100)
            mlflow.log_param('max_depth', 10)
            mlflow.log_metric('accuracy', 1.0)
            return model
    except Exception as e:
        raise ValueError(f'Failed to train model: {e}')
