import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise ValueError(f'Failed to load data: {e}')

def preprocess_data(data):
    """
    Preprocess the data by separating features and target.

    Args:
        data (pd.DataFrame): Input data.

    Returns:
        tuple: Tuple containing features (X) and target (y).
    """
    try:
        X = data.drop(columns=['Class'])
        y = data['Class']
        return (X, y)
    except Exception as e:
        raise ValueError(f'Failed to preprocess data: {e}')

def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        RandomForestClassifier: Trained model.
    """
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, 'model')
            mlflow.log_metric('accuracy', accuracy_score(y_train, model.predict(X_train)))
            mlflow.log_params(model.get_params())
            return model
    except Exception as e:
        raise ValueError(f'Failed to train model: {e}')
if __name__ == '__main__':
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
