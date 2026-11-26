import pytest
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test loading data."""
    try:
        data = load_data('data/traffic.csv')
        assert data is not None
    except Exception as e:
        pytest.fail(f"Test failed: {e}")

def test_preprocess_data():
    """Test preprocessing data."""
    try:
        data = load_data('data/traffic.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        assert X_train is not None and y_train is not None
    except Exception as e:
        pytest.fail(f"Test failed: {e}")

def test_train_model():
    """Test training the model."""
    try:
        data = load_data('data/traffic.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        assert model is not None and accuracy is not None
    except Exception as e:
        pytest.fail(f"Test failed: {e}")
