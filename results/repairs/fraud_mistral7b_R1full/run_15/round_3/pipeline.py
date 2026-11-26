import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import mlflow
import mlflow.sklearn

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data['Class'] = data['Class'].apply(lambda x: [x])  # Convert integer to list
    y = data['Class']
    data, data['Amount'] = preprocess_data_binary(data)
    return data, y

def preprocess_data_binary(data):
    # Add any necessary binary preprocessing steps here, such as normalization, encoding, etc.
    # For simplicity, I'll just return the data as it is.
    return data, data['Amount']

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random_forest")

    return model
