import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError('The file does not exist.')
    except pd.errors.EmptyDataError:
        raise ValueError('The file is empty.')

def preprocess_data(data):
    """Preprocess the data by splitting it into features and target."""
    X = data.drop(columns=['Class'])
    y = data['Class']
    return (X, y)

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's accuracy."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def main():
    file_path = 'data/fraud.csv'
    data = load_data(file_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    with mlflow.start_run():
        model = train_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_artifact(data, artifact_path='data')
if __name__ == '__main__':
    main()