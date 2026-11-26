import pytest
import pandas as pd
from pipeline import load_data, preprocess_data, train_model, evaluate_model, run_pipeline

@pytest.fixture
def test_data():
    # Create a sample DataFrame for testing
    data = {
        'temp': [20, 22, 18, 25],
        'rain_1h': [0.0, 0.1, 0.2, 0.0],
        'snow_1h': [0.0, 0.0, 0.0, 0.1],
        'clouds_all': [50, 60, 70, 80],
        'weather_main': ['Sunny', 'Rainy', 'Snowy', 'Sunny'],
        'IsHoliday': [False, True, False, True],
        'Hour': [9, 10, 11, 12],
        'DayOfWeek': ['Monday', 'Tuesday', 'Wednesday', 'Thursday'],
        'Month': [1, 2, 3, 4],
        'Congestion': [0, 1, 0, 1]
    }
    return pd.DataFrame(data)

def test_load_data(test_data):
    """Test loading data."""
    with pytest.raises(FileNotFoundError):
        load_data('non_existent_file.csv')
    with pytest.raises(pd.errors.EmptyDataError):
        load_data('empty.csv')
    with pytest.raises(pd.errors.ParserError):
        load_data('invalid.csv')
    assert load_data('data/traffic.csv').equals(test_data)

def test_preprocess_data(test_data):
    """Test preprocessing data."""
    X_train, X_test, y_train, y_test = preprocess_data(test_data)
    assert X_train.shape == (3, 8)
    assert y_train.shape == (3,)
    assert X_test.shape == (1, 8)
    assert y_test.shape == (1,)

def test_train_model(test_data):
    """Test training the model."""
    X_train, X_test, y_train, y_test = preprocess_data(test_data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model(test_data):
    """Test evaluating the model."""
    X_train, X_test, y_train, y_test = preprocess_data(test_data)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float)

def test_run_pipeline(test_data):
    """Test running the entire pipeline."""
    accuracy = run_pipeline('data/traffic.csv')
    assert isinstance(accuracy, float)
