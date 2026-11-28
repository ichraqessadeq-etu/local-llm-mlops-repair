import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow import trace_experiment, pyfunc, log_param, log_artifact

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps like filling NaN, encoding categorical variables, etc.
    # For simplicity, I'm assuming that the data is already preprocessed.
    return data['Features'], data['Congestion']

def train_model(X_train, y_train):
    with mlflow.start_run():
        tracked_model = pyfunc.track_experiment('RandomForestCongestionDetector')
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        tracked_model.log_model(model)
        log_param('num_trees', model.n_estimators)
        log_artifact('model.pkl', model)
