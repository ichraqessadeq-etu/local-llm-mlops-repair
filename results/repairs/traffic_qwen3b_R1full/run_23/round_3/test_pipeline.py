import pytest
import pandas as pd
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test loading data."""
    file_path = 'data/traffic.csv'
    data = load_data(file_path)
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test preprocessing data."""
    data = pd.DataFrame({
        'temp': [10, 20, 30],
        'rain_1h': [0, 1, 0],
        'snow_1h': [0, 0, 1],
        'clouds_all': [50, 60, 70],
        'weather_main': ['Clear', 'Clouds', 'Snow'],
        'IsHoliday': [False, True, False],
        'Hour': [1, 2, 3],
        'DayOfWeek': [1, 2, 3],
        'Month': [1, 2, 3],
        'Congestion': [0, 1, 0]
    })
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model():
    """Test training the model."""
    data = pd.DataFrame({
        'temp': [10, 20, 30],
        'rain_1h': [0, 1, 0],
        'snow_1h': [0, 0, 1],
        'clouds_all': [50, 60, 70],
        'weather_main': ['Clear', 'Clouds', 'Snow'],
        'IsHoliday': [False, True, False],
        'Hour': [1, 2, 3],
        'DayOfWeek': [1, 2, 3],
        'Month': [1, 2, 3],
        'Congestion': [0, 1, 0]
    })
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
