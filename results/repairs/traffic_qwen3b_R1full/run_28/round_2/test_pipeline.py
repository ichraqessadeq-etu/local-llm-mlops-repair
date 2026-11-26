import pytest
from pipeline import load_data, preprocess_data, train_model
import pandas as pd

@pytest.fixture
def data():
    # Load a sample dataset
    data = pd.DataFrame({
        'temp': [20, 22, 18, 25, 30],
        'rain_1h': [0, 1, 0, 1, 0],
        'snow_1h': [0, 0, 1, 0, 0],
        'clouds_all': [50, 70, 40, 90, 60],
        'weather_main': ['Clouds', 'Rain', 'Snow', 'Clouds', 'Rain'],
        'IsHoliday': [False, True, False, True, False],
        'Hour': [9, 15, 12, 18, 10],
        'DayOfWeek': ['Monday', 'Friday', 'Wednesday', 'Thursday', 'Tuesday'],
        'Month': ['January', 'March', 'December', 'February', 'November']
    }, columns=['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month', 'Congestion'])
    return data

def test_load_data(data):
    """Test data loading."""
    try:
        loaded_data = load_data(data)
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(data)
    except Exception as e:
        assert False, f"Test failed: {e}"

def test_preprocess_data(data):
    """Test data preprocessing."""
    try:
        X, y = preprocess_data(data)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
    except Exception as e:
        assert False, f"Test failed: {e}"

def test_train_model(data):
    """Test model training."""
    try:
        X, y = preprocess_data(data)
        model = train_model(X, y)
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 100
        assert model.random_state == 42
    except Exception as e:
        assert False, f"Test failed: {e}"
