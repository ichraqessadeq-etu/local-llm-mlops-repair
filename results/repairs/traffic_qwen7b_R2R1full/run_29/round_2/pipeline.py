import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.models import ModelSignature, ModelInputExample
from mlflow.types.schema import Schema, ColSpec

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess data for model training."""
    # Handle missing values
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)

    # Convert categorical variables to numerical
    data['weather_main'] = data['weather_main'].astype('category').cat.codes
    data['IsHoliday'] = data['IsHoliday'].astype('category').cat.codes

    # Split features and target
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']

    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier."""
    with mlflow.start_run():
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("features", list(X_train.columns))
        mlflow.log_param("target", "Congestion")

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log metrics
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        mlflow.log_metric("train_accuracy", accuracy)

        # Infer model signature
        input_example = X_train.iloc[:5]
        signature = infer_signature(input_example, model.predict(input_example))

        # Log model signature
        mlflow.log_model(model, "model", signature=signature)

        return model

