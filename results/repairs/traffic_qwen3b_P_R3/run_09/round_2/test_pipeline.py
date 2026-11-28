from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

def test_load_data():
    data = load_data('data/traffic.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    data = pd.DataFrame({
        'temp': [20, 22, 18],
        'rain_1h': [0.1, 0.2, 0.0],
        'snow_1h': [0.0, 0.0, 0.0],
        'clouds_all': [50, 60, 70],
        'weather_main': ['Sunny', 'Cloudy', 'Sunny'],
        'IsHoliday': [False, True, False],
        'Hour': [10, 14, 18],
        'DayOfWeek': ['Monday', 'Wednesday', 'Friday'],
        'Month': [5, 7, 9],
        'Congestion': [0, 1, 0]
    })
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model():
    data = pd.DataFrame({
        'temp': [20, 22, 18],
        'rain_1h': [0.1, 0.2, 0.0],
        'snow_1h': [0.0, 0.0, 0.0],
        'clouds_all': [50, 60, 70],
        'weather_main': ['Sunny', 'Cloudy', 'Sunny'],
        'IsHoliday': [False, True, False],
        'Hour': [10, 14, 18],
        'DayOfWeek': ['Monday', 'Wednesday', 'Friday'],
        'Month': [5, 7, 9],
        'Congestion': [0, 1, 0]
    })
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    data = pd.DataFrame({
        'temp': [20, 22, 18],
        'rain_1h': [0.1, 0.2, 0.0],
        'snow_1h': [0.0, 0.0, 0.0],
        'clouds_all': [50, 60, 70],
        'weather_main': ['Sunny', 'Cloudy', 'Sunny'],
        'IsHoliday': [False, True, False],
        'Hour': [10, 14, 18],
        'DayOfWeek': ['Monday', 'Wednesday', 'Friday'],
        'Month': [5, 7, 9],
        'Congestion': [0, 1, 0]
    })
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float)
