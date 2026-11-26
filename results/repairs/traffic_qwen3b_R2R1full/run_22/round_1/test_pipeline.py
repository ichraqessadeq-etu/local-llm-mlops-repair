from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, run_pipeline

def test_load_data():
    """Test loading data."""
    with pytest.raises(ValueError):
        load_data("nonexistent_file.csv")

def test_preprocess_data():
    """Test preprocessing data."""
    data = pd.DataFrame({
        'temp': [20, 22, 18],
        'rain_1h': [0.1, 0.0, 0.2],
        'snow_1h': [0.0, 0.1, 0.0],
        'clouds_all': [50, 40, 60],
        'weather_main': ['Sunny', 'Cloudy', 'Sunny'],
        'IsHoliday': [0, 1, 0],
        'Hour': [10, 14, 18],
        'DayOfWeek': [1, 2, 3],
        'Month': [1, 2, 3],
        'Congestion': ['Low', 'Medium', 'High']
    })
    X, y = preprocess_data(data)
    assert X.shape == (3, 10)
    assert y.shape == (3,)
    assert X.columns.tolist() == ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'IsHoliday_0', 'IsHoliday_1', 'Hour', 'DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'Month_1', 'Month_2', 'Month_3']
    assert y.tolist() == [0, 1, 2]

def test_train_model():
    """Test training the model."""
    data = pd.DataFrame({
        'temp': [20, 22, 18],
        'rain_1h': [0.1, 0.0, 0.2],
        'snow_1h': [0.0, 0.1, 0.0],
        'clouds_all': [50, 40, 60],
        'weather_main': ['Sunny', 'Cloudy', 'Sunny'],
        'IsHoliday': [0, 1, 0],
        'Hour': [10, 14, 18],
        'DayOfWeek': [1, 2, 3],
        'Month': [1, 2, 3],
        'Congestion': ['Low', 'Medium', 'High']
    })
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_run_pipeline():
    """Test the entire pipeline."""
    try:
        run_pipeline("data/traffic.csv")
    except Exception as e:
        pytest.fail(f"Pipeline error: {e}")
