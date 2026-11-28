import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f'Error loading data: {e}')
        raise

def preprocess_data(data):
    try:
        X = data.drop(['Class', 'Amount'], axis=1)
        y = data['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return (X_train, X_test, y_train, y_test)
    except Exception as e:
        print(f'Error preprocessing data: {e}')
        raise

def train_model(X_train, y_train):
    try:
        mlflow.log_param('n_estimators', 100)
        mlflow.log_param('max_depth', 5)
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, 'random_forest_model')
        return model
    except Exception as e:
        print(f'Error training model: {e}')
        raise