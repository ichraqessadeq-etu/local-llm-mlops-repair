import pytest
from pipeline import load_data, preprocess_data, train_model, save_model

@pytest.fixture
def data():
    """Fixture to load and preprocess data."""
    try:
        data = load_data('data/traffic.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise Exception(f"Error loading and preprocessing data: {e}")

def test_load_data():
    """Test loading data."""
    try:
        load_data('data/traffic.csv')
    except Exception as e:
        pytest.fail(f"Failed to load data: {e}")

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

def test_save_model(data):
    """Test saving the model."""
    X_train, X_test, y_train, y_test = data
    model = train_model(X_train, y_train)
    save_model(model, "TrafficCongestionDetector", "Run1")
