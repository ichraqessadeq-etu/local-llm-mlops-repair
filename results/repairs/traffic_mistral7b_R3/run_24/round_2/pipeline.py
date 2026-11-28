import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_param, log_artifact, log_metric

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data['Features'], data['Congestion'])

def train_model(X_train, y_train):
    with PyFuncExperiment(experiment_name='Congestion_Detector') as exp:
        try:
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            log_param(exp, 'learning_rate', model.get_params()['learning_rate'])
            log_artifact(exp, 'model.pkl', pickle.dumps(model))
            log_metric(exp, 'train_loss', model.fit(X_train, y_train).loss_)
        except Exception as e:
            print(f'Error occurred: {e}')
    pass  # R3: API inexistante retirée