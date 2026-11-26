from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

def test_load_data():
    """Test loading the dataset."""
    data = load_data('data/traffic.csv')
    assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame."

def test_preprocess_data():
    """Test preprocessing the data."""
    data = pd.DataFrame({
        'temp': [20, 22, 18],
        'rain_1h': [0.1, 0.2, 0.0],
        'snow_1h': [0.0, 0.0, 0.1],
        'clouds_all': [50, 60, 40],
        'weather_main': ['Clouds', 'Rain', 'Snow'],
        'IsHoliday': [False, True, False],
        'Hour': [9, 15, 21],
        'DayOfWeek': ['Monday', 'Wednesday', 'Friday'],
        'Month': ['March', 'May', 'August']
    })
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame), "X should be a pandas DataFrame."
    assert isinstance(y, pd.Series), "y should be a pandas Series."

def test_train_model():
    """Test training the model."""
    data = pd.DataFrame({
        'temp': [20, 22, 18],
        'rain_1h': [0.1, 0.2, 0.0],
        'snow_1h': [0.0, 0.0, 0.1],
        'clouds_all': [50, 60, 40],
        'weather_main': ['Clouds', 'Rain', 'Snow'],
        'IsHoliday': [False, True, False],
        'Hour': [9, 15, 21],
        'DayOfWeek': ['Monday', 'Wednesday', 'Friday'],
        'Month': ['March', 'May', 'August'],
        'Congestion': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "Model should be a RandomForestClassifier."

def test_evaluate_model():
    """Test evaluating the model."""
    data = pd.DataFrame({
        'temp': [20, 22, 18],
        'rain_1h': [0.1, 0.2, 0.0],
        'snow_1h': [0.0, 0.0, 0.1],
        'clouds_all': [50, 60, 40],
        'weather_main': ['Clouds', 'Rain', 'Snow'],
        'IsHoliday': [False, True, False],
        'Hour': [9, 15, 21],
        'DayOfWeek': ['Monday', 'Wednesday', 'Friday'],
        'Month': ['March', 'May', 'August'],
        'Congestion': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    model = train_model(X, y)
    accuracy, report = evaluate_model(model, X, y)
    assert isinstance(accuracy, float), "Accuracy should be a float."
    assert isinstance(report, str), "Classification report should be a string."
