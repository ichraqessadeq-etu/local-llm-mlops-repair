from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test data loading."""
    data = load_data('data/traffic.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test data preprocessing."""
    data = pd.DataFrame({
        'temp': [20, 22, 18],
        'rain_1h': [0.5, 0.0, 0.3],
        'snow_1h': [0.0, 0.2, 0.1],
        'clouds_all': [70, 60, 80],
        'weather_main': ['Clear', 'Clouds', 'Rain'],
        'IsHoliday': [False, True, False],
        'Hour': [10, 14, 18],
        'DayOfWeek': [1, 3, 5],
        'Month': [1, 2, 3]
    })
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model():
    """Test model training."""
    data = pd.DataFrame({
        'temp': [20, 22, 18],
        'rain_1h': [0.5, 0.0, 0.3],
        'snow_1h': [0.0, 0.2, 0.1],
        'clouds_all': [70, 60, 80],
        'weather_main': ['Clear', 'Clouds', 'Rain'],
        'IsHoliday': [False, True, False],
        'Hour': [10, 14, 18],
        'DayOfWeek': [1, 3, 5],
        'Month': [1, 2, 3],
        'Congestion': [0, 1, 0]
    })
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
