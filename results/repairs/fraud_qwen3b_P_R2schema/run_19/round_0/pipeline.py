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
    """Preprocess the data by splitting it into features and target."""
    X = data.drop('Class', axis=1)
    y = data['Class']
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, experiment_name, run_name):
    """Save the trained model using MLflow."""
    with mlflow.start_run(experiment_name=experiment_name, run_name=run_name):
        mlflow.sklearn.log_model(model, 'model')
        mlflow.log_metric('accuracy', accuracy_score(y_train, model.predict(X_train)))

def main():
    file_path = 'data/fraud.csv'
    data = load_data(file_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    save_model(model, 'fraud_detection', 'random_forest')

if __name__ == '__main__':
    main()
