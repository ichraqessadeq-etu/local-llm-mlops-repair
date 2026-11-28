from sklearn.model_selection import train_test_split
import pytest
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    data = load_data('data/traffic.csv')
    assert not data.empty
    assert 'Congestion' in data.columns

def test_preprocess_data():
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    assert X.shape[1] == len(['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month'])
    assert y.shape[0] == X.shape[0]

def test_train_model():
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert model is not None
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test)
