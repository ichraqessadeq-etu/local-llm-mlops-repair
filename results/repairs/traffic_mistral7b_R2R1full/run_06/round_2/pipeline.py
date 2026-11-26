import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow

def load_data(file_path):
    return pd.read_csv(file_path, index_col=False)

def preprocess_data(data):
    data = data.copy()
    data = data.rename(columns={'Features': 'X', 'Congestion': 'y'})
    return (data['X'], data['y'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.log_artifact(f'model', local_model=model)
        mlflow.log_param('num_trees', model.n_estimators)
        mlflow.log_metric('train_loss', model.fit_score(X_train, y_train))
        return model
