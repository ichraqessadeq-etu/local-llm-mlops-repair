import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import mlflow

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps like filling NaN, encoding categorical variables, scaling features, etc.
    # For simplicity, I'm assuming that the data is already preprocessed.
    data = pd.get_dummies(data)  # One-hot encoding for categorical variables
    return data['Features'], data['Congestion']

def train_model(X_train, y_train):
    random_state = np.random.randint(0, 100)  # Seed the random number generator
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        mlflow.log_artifact(f'model', local_model=model)
        mlflow.log_param('num_trees', model.n_estimators)
        mlflow.log_metric('train_loss', model.fit_score(X_train, y_train))
    return model
