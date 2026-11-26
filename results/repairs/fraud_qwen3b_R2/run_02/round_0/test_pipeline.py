import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

@pytest.fixture
def data():
    """Fixture to load and preprocess data."""
    file_path = 'data/fraud.csv'
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    return X_train, X_test, y_train, y_test

def test_load_data():
    """Test loading data."""
    with pytest.raises(Exception):
        load_data('non_existent_file.csv')

def test_preprocess_data(data):
    """Test preprocessing data."""
    X_train, X_test, y_train, y_test = data
    assert X_train.shape[0] > 0
    assert y_train.shape[0] > 0

def test_train_model(data):
    """Test training the model."""
    X_train, X_test, y_train, y_test = data
    model = train_model(X_train, y_train)
    assert model is not None

def test_evaluate_model(data):
    """Test evaluating the model."""
    X_train, X_test, y_train, y_test = data
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert accuracy > 0
