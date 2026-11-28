from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
import pipeline

def test_load_data():
    """Test loading data from the CSV file."""
    data = pipeline.load_data('data/traffic.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test preprocessing the data."""
    data = pd.DataFrame({
        'temp': [20, 22, 18],
        'rain_1h': [0.5, 0.3, 0.0],
        'snow_1h': [0.0, 0.1, 0.0],
        'clouds_all': [50, 70, 30],
        'weather_main': ['sunny', 'rainy', 'sunny'],
        'IsHoliday': [False, True, False],
        'Hour': [12, 15, 8],
        'DayOfWeek': [1, 3, 5],
        'Month': [10, 12, 1]
    })
    X, y = pipeline.preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_train_model():
    """Test training the model."""
    data = pd.DataFrame({
        'temp': [20, 22, 18],
        'rain_1h': [0.5, 0.3, 0.0],
        'snow_1h': [0.0, 0.1, 0.0],
        'clouds_all': [50, 70, 30],
        'weather_main': ['sunny', 'rainy', 'sunny'],
        'IsHoliday': [False, True, False],
        'Hour': [12, 15, 8],
        'DayOfWeek': [1, 3, 5],
        'Month': [10, 12, 1],
        'Congestion': [0, 1, 0]
    })
    X, y = pipeline.preprocess_data(data)
    model = pipeline.train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    """Test evaluating the model."""
    data = pd.DataFrame({
        'temp': [20, 22, 18],
        'rain_1h': [0.5, 0.3, 0.0],
        'snow_1h': [0.0, 0.1, 0.0],
        'clouds_all': [50, 70, 30],
        'weather_main': ['sunny', 'rainy', 'sunny'],
        'IsHoliday': [False, True, False],
        'Hour': [12, 15, 8],
        'DayOfWeek': [1, 3, 5],
        'Month': [10, 12, 1],
        'Congestion': [0, 1, 0]
    })
    X, y = pipeline.preprocess_data(data)
    model = pipeline.train_model(X, y)
    accuracy = pipeline.evaluate_model(model, X, y)
    assert isinstance(accuracy, float)
