import pytest
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    try:
        data = load_data('data/traffic.csv')
        assert isinstance(data, pd.DataFrame)
        assert 'Congestion' in data.columns
    except Exception as e:
        pytest.fail(f"Test failed: {e}")

def test_preprocess_data():
    try:
        data = load_data('data/traffic.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
    except Exception as e:
        pytest.fail(f"Test failed: {e}")

def test_train_model():
    try:
        data = load_data('data/traffic.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        assert isinstance(model, RandomForestClassifier)
    except Exception as e:
        pytest.fail(f"Test failed: {e}")
