# test_pipeline.py
from sklearn.model_selection import train_test_split
from pipeline import load_data, preprocess_data, train_model
import pytest
import mlflow

@pytest.fixture(scope="module")
def data():
    return load_data("data/traffic.csv")

def test_preprocess_data(data):
    X, y = preprocess_data(data)
    assert len(X.columns) == 8

def test_train_model(data):
    X, y = preprocess_data(data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)

    # Add assertions to check the model's performance and parameters
