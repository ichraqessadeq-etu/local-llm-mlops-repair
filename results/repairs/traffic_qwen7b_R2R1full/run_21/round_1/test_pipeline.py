from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pipeline import load_data, preprocess_data, train_model

Features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']

def test_load_data():
    """Test data loading function."""
    data = load_data("data/traffic.csv")
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

def test_preprocess_data():
    """Test data preprocessing function."""
    data = load_data("data/traffic.csv")
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert not X_train.empty
    assert not X_test.empty
    assert not y_train.empty
    assert not y_test.empty

def test_train_model():
    """Test model training function."""
    data = load_data("data/traffic.csv")
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100

