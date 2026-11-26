import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: The loaded data.
    """
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    Preprocess the data by separating features and target.
    
    Parameters:
    data (pd.DataFrame): The input data.
    
    Returns:
    tuple: A tuple containing the features (X) and target (y).
    """
    X = data.drop(columns=['Anomaly'])
    y = data['Anomaly']
    return X, y

def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier on the provided data.
    
    Parameters:
    X_train (pd.DataFrame): The training features.
    y_train (pd.Series): The training target.
    
    Returns:
    RandomForestClassifier: The trained model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test data.
    
    Parameters:
    model (RandomForestClassifier): The trained model.
    X_test (pd.DataFrame): The test features.
    y_test (pd.Series): The test target.
    
    Returns:
    float: The accuracy of the model.
    """
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
