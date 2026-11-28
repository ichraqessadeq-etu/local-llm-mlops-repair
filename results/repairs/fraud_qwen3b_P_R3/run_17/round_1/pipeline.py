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
        raise ValueError('File not found.')
    except pd.errors.EmptyDataError:
        raise ValueError('Empty data in the file.')

def preprocess_data(data):
    """Preprocess the data by splitting it into features and target."""
    X = data.drop('Class', axis=1)
    y = data['Class']
    return (X, y)

def train_model(X_train, y_train):
    """Train a RandomForestClassifier and log the model using MLflow."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    with mlflow.start_run():
        mlflow.log_param('model', 'RandomForestClassifier')
        mlflow.sklearn.log_model(model, 'model')
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        mlflow.log_metric('accuracy', accuracy)
        return model

def main():
    file_path = 'data/fraud.csv'
    data = load_data(file_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy}')
if __name__ == '__main__':
    main()