# pipeline.py
def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Add any preprocessing steps like normalization, encoding, etc.
    # Return X (features) and y (target)

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "anomaly_detector")

    return model

