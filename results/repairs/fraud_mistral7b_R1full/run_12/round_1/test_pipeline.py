import pytest
from pipeline import load_data, preprocess_data, train_model

@pytest.fixture(scope="module")
def data():
    return load_data("data/fraud.csv")

def test_pipeline(data):
    X, y = preprocess_data(data)
    train_model(X_train=X, y_train=y)
