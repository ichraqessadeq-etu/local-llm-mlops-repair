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
        raise FileNotFoundError(f'File {file_path} not found.')
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError('The file is empty or contains no data.')

def preprocess_data(data):
    """Preprocess the data by splitting it into features and target."""
    X = data.drop('Class', axis=1)
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (X_train, y_train, X_test, y_test)

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, 'random_forest_model')
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)
        return model

def main():
    file_path = 'data/fraud.csv'
    data = load_data(file_path)
    X_train, y_train, X_test, y_test = preprocess_data(data)
    model = train_model(X_train, y_test)
    print('Model trained and logged to MLflow.')
if __name__ == '__main__':
    main()