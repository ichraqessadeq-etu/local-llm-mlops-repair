import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data by splitting into features and target."""
    X = data.drop('Class', axis=1)
    y = data['Class']
    return (X, y)

def train_model(X_train, y_train):
    """Train a RandomForest classifier."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def run_pipeline(file_path):
    """Run the entire pipeline."""
    data = load_data(file_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    with mlflow.start_run():
        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_param('model', 'RandomForestClassifier')
        mlflow.log_metric('accuracy', accuracy)
        mlflow.sklearn.log_model(model, 'model')
        print(f'Model accuracy: {accuracy:.2f}')
if __name__ == '__main__':
    run_pipeline('data/fraud.csv')