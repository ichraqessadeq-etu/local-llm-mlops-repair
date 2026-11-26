import pytest
import pandas as pd
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test the load_data function."""
    try:
        data = load_data('data/traffic.csv')
        assert isinstance(data, pd.DataFrame)
        logging.info("Test load_data passed.")
    except Exception as e:
        logging.error(f"Test load_data failed: {e}")
        pytest.fail()

def test_preprocess_data():
    """Test the preprocess_data function."""
    try:
        data = load_data('data/traffic.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        logging.info("Test preprocess_data passed.")
    except Exception as e:
        logging.error(f"Test preprocess_data failed: {e}")
        pytest.fail()

def test_train_model():
    """Test the train_model function."""
    try:
        data = load_data('data/traffic.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        rf = train_model(X_train, y_train)
        assert isinstance(rf, RandomForestClassifier)
        logging.info("Test train_model passed.")
    except Exception as e:
        logging.error(f"Test train_model failed: {e}")
        pytest.fail()
