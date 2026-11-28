from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

def test_load_data():
    """Test loading data."""
    file_path = 'data/traffic.csv'
    data = load_data(file_path)
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test preprocessing data."""
    data = pd.DataFrame({
        'temp': [20, 22, 18],
        'rain_1h': [0.1, 0.2, 0.0],
        'snow_1h': [0.0, 0.0, 0.0],
        'clouds_all': [50, 60, 70],
        'weather_main': ['sunny', 'rainy', 'cloudy'],
        'IsHoliday': [False, True, False],
        'Hour': [9, 14, 18],
        'DayOfWeek': [1, 2, 3],
        'Month': [1, 2, 3],
        'Congestion': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_train_model():
    """Test training the model."""
    data = pd.DataFrame({
        'temp': [20, 22, 18],
        'rain_1h': [0.1, 0.2, 0.0],
        'snow_1h': [0.0, 0.0, 0.0],
        'clouds_all': [50, 60, 70],
        'weather_main': ['sunny', 'rainy', 'cloudy'],
        'IsHoliday': [False, True, False],
        'Hour': [9, 14, 18],
        'DayOfWeek': [1, 2, 3],
        'Month': [1, 2, 3],
        'Congestion': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    """Test evaluating the model."""
    data = pd.DataFrame({
        'temp': [20, 22, 18],
        'rain_1h': [0.1, 0.2, 0.0],
        'snow_1h': [0.0, 0.0, 0.0],
        'clouds_all': [50, 60, 70],
        'weather_main': ['sunny', 'rainy', 'cloudy'],
        'IsHoliday': [False, True, False],
        'Hour': [9, 14, 18],
        'DayOfWeek': [1, 2, 3],
        'Month': [1, 2, 3],
        'Congestion': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    model = train_model(X, y)
    accuracy = evaluate_model(model, X, y)
    assert isinstance(accuracy, float)
