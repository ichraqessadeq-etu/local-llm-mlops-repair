import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the dataset."""
    # Example preprocessing: drop columns with high cardinality
    data = data.drop(['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'], axis=1)
    # Example preprocessing: scale the 'Amount' column
    data['Amount'] = (data['Amount'] - data['Amount'].mean()) / data['Amount'].std()
    # Split the data into features and target
    X = data.drop('Class', axis=1)
    y = data['Class']
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
