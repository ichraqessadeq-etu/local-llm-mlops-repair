# test_pipeline.py
from pipeline import load_data, preprocess_data, train_model
import pytest
import os

Features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']

@pytest.fixture(scope='module')
def data():
    return load_data(os.path.join('data', 'traffic.csv'))

def test_train_model(data):
    X, y = preprocess_data(data)
    model = train_model(X_train=X, y_train=y)
    # Add assertions to check if the model is trained correctly
