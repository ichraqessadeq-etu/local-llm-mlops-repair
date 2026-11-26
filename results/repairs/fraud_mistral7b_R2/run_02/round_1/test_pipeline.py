import pytest
from pipeline import load_data, preprocess_data, train_model, save_model

@pytest.fixture(scope="module")
def data():
    return load_data("data/fraud.csv")

def test_preprocess_data(data):
    X, y = preprocess_data(data)
    assert len(X.shape) == 2
    assert len(y) == data.shape[0]

def test_train_model(data):
    X, y = preprocess_data(data)
    model = train_model(X_train=X, y_train=y)
    # Add your custom assertions here to check the model's performance

def test_save_model(data):
    X, y = preprocess_data(data)
    model = train_model(X_train=X, y_train=y)
    save_model(model, "random_forest_fraud_detector")
    # Add your custom assertions here to check the saved model
