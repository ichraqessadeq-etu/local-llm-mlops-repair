import pytest
import pandas as pd
from pipeline import load_data, preprocess_data, train_model, evaluate_model, run_pipeline

def test_load_data():
    """Test loading data."""
    data = load_data('data/traffic.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test preprocessing data."""
    data = pd.DataFrame({
        'temp': [20, 22, 21],
        'rain_1h': [0.5, 0.3, 0.2],
        'snow_1h': [0, 0.1, 0],
        'clouds_all': [50, 60, 70],
        'weather_main': ['Clouds', 'Rain', 'Snow'],
        'IsHoliday': [False, True, False],
        'Hour': [10, 14, 18],
        'DayOfWeek': ['Monday', 'Wednesday', 'Friday'],
        'Month': [1, 4, 7],
        'Congestion': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (3, 8)
    assert y.shape == (3,)

def test_train_model():
    """Test training the model."""
    data = pd.DataFrame({
        'temp': [20, 22, 21],
        'rain_1h': [0.5, 0.3, 0.2],
        'snow_1h': [0, 0.1, 0],
        'clouds_all': [50, 60, 70],
        'weather_main': ['Clouds', 'Rain', 'Snow'],
        'IsHoliday': [False, True, False],
        'Hour': [10, 14, 18],
        'DayOfWeek': ['Monday', 'Wednesday', 'Friday'],
        'Month': [1, 4, 7],
        'Congestion': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    """Test evaluating the model."""
    data = pd.DataFrame({
        'temp': [20, 22, 21],
        'rain_1h': [0.5, 0.3, 0.2],
        'snow_1h': [0, 0.1, 0],
        'clouds_all': [50, 60, 70],
        'weather_main': ['Clouds', 'Rain', 'Snow'],
        'IsHoliday': [False, True, False],
        'Hour': [10, 14, 18],
        'DayOfWeek': ['Monday', 'Wednesday', 'Friday'],
        'Month': [1, 4, 7],
        'Congestion': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    model = train_model(X, y)
    evaluate_model(model, X, y)

def test_run_pipeline():
    """Test running the entire pipeline."""
    run_pipeline('data/traffic.csv')
