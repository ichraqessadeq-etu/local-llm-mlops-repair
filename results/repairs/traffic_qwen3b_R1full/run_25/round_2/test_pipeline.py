import pytest
import pandas as pd
from pipeline import load_data, preprocess_data, train_model, main
from sklearn.preprocessing import LabelEncoder

def test_load_data():
    data = load_data('data/traffic.csv')
    assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame."

def test_preprocess_data():
    data = pd.DataFrame({
        'temp': [22, 18],
        'rain_1h': [0.2, 0.0],
        'snow_1h': [0.0, 0.1],
        'clouds_all': [60, 70],
        'weather_main': ['Rain', 'Snow'],
        'IsHoliday': [False, True],
        'Hour': [14, 18],
        'DayOfWeek': [2, 3],
        'Month': [3, 3]
    })
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame), "X_train should be a pandas DataFrame."
    assert isinstance(X_test, pd.DataFrame), "X_test should be a pandas DataFrame."
    assert isinstance(y_train, pd.Series), "y_train should be a pandas Series."
    assert isinstance(y_test, pd.Series), "y_test should be a pandas Series."

def test_train_model():
    data = pd.DataFrame({
        'temp': [22, 18],
        'rain_1h': [0.2, 0.0],
        'snow_1h': [0.0, 0.1],
        'clouds_all': [60, 70],
        'weather_main': ['Rain', 'Snow'],
        'IsHoliday': [False, True],
        'Hour': [14, 18],
        'DayOfWeek': [2, 3],
        'Month': [3, 3]
    })
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model should be a RandomForestClassifier."

def test_main():
    main()
    # Add more assertions to verify the output of the main function
