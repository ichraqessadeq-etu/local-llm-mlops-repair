import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def load_data(file_path):
    return pd.read_csv(file_path, index_col=False)

def preprocess_data(data):
    data.rename(columns={'Features': 'Time', 'Class': 'target'}, inplace=True)
    data['Time'] = data['Time'].values.reshape(-1, 1)
    return (data['Time'], data['target'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, 'random_forest')
        run_id = mlflow.get_run_id()
        print(f'Training run ID: {run_id}')
