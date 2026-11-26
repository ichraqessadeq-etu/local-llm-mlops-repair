import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Define the features
features = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'NOx(GT)', 'C6H6(GT)', 'T', 'RH', 'AH', 'Hour', 'DayOfWeek', 'Month']

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps like filling missing values, encoding categorical variables, etc.
    # For simplicity, I'm assuming no preprocessing is needed
    return data[features], data['Anomaly']

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)  # Seed the random number generator
        model.fit(X_train, y_train)
        mlflow.log_artifact('model', model)
        mlflow.log_metric('train_loss', model.fit_params['loss'])
    return model
