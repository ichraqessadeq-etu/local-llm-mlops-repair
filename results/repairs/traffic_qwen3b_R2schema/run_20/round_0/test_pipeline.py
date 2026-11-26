import pytest
from pipeline import load_data, preprocess_data, train_model

@pytest.fixture
def data():
    file_path = 'data/traffic.csv'
    return load_data(file_path)

def test_load_data(data):
    assert data is not None

def test_preprocess_data(data):
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None

def test_train_model(data):
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert model is not None
