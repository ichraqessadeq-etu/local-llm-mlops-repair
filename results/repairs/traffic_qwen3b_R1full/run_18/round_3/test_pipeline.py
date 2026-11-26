import pytest
import pandas as pd
from pipeline import load_data, preprocess_data, train_model, evaluate_model

def test_load_data():
    file_path = 'data/traffic.csv'
    data = load_data(file_path)
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    data = pd.DataFrame({
        'temp': [20, 22, 18, 25],
        'rain_1h': [0, 1, 0, 1],
        'snow_1h': [0, 0, 1, 0],
        'clouds_all': [50, 70, 30, 60],
        'weather_main': ['Clear', 'Clouds', 'Rain', 'Snow'],
        'IsHoliday': [False, True, False, True],
        'Hour': [10, 14, 18, 22],
        'DayOfWeek': [1, 2, 3, 4],
        'Month': [1, 2, 3, 4],
        'Congestion': [0, 1, 0, 1]
    })
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model():
    data = pd.DataFrame({
        'temp': [20, 22, 18, 25],
        'rain_1h': [0, 1, 0, 1],
        'snow_1h': [0, 0, 1, 0],
        'clouds_all': [50, 70, 30, 60],
        'weather_main': ['Clear', 'Clouds', 'Rain', 'Snow'],
        'IsHoliday': [False, True, False, True],
        'Hour': [10, 14, 18, 22],
        'DayOfWeek': [1, 2, 3, 4],
        'Month': [1, 2, 3, 4],
        'Congestion': [0, 1, 0, 1]
    })
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    data = pd.DataFrame({
        'temp': [20, 22, 18, 25],
        'rain_1h': [0, 1, 0, 1],
        'snow_1h': [0, 0, 1, 0],
        'clouds_all': [50, 70, 30, 60],
        'weather_main': ['Clear', 'Clouds', 'Rain', 'Snow'],
        'IsHoliday': [False, True, False, True],
        'Hour': [10, 14, 18, 22],
        'DayOfWeek': [1, 2, 3, 4],
        'Month': [1, 2, 3, 4],
        'Congestion': [0, 1, 0, 1]
    })
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float)
