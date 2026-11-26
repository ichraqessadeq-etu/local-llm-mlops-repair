import pytest
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test loading data."""
    data = load_data('data/traffic.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test preprocessing data."""
    data = pd.DataFrame({
        'temp': [20, 22, 18],
        'rain_1h': [0.1, 0.2, 0.0],
        'snow_1h': [0.0, 0.0, 0.1],
        'clouds_all': [50, 70, 30],
        'weather_main': ['sunny', 'rainy', 'snowy'],
        'IsHoliday': [False, True, False],
        'Hour': [10, 14, 18],
        'DayOfWeek': ['Monday', 'Wednesday', 'Friday'],
        'Month': ['January', 'March', 'May']
    })
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_train_model():
    """Test training the model."""
    data = pd.DataFrame({
        'temp': [20, 22, 18],
        'rain_1h': [0.1, 0.2, 0.0],
        'snow_1h': [0.0, 0.0, 0.1],
        'clouds_all': [50, 70, 30],
        'weather_main': ['sunny', 'rainy', 'snowy'],
        'IsHoliday': [False, True, False],
        'Hour': [10, 14, 18],
        'DayOfWeek': ['Monday', 'Wednesday', 'Friday'],
        'Month': ['January', 'March', 'May'],
        'Congestion': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
