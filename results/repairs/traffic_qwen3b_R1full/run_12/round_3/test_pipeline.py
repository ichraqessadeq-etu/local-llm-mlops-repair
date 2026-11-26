import pytest
import pandas as pd
from pipeline import load_data, preprocess_data, train_model, evaluate_model, run_pipeline

@pytest.fixture
def test_data():
    # Create a sample DataFrame for testing
    data = {
        'temp': [291.25, 295.89],
        'rain_1h': [1.02, 0.00],
        'snow_1h': [0.0, 0.0],
        'clouds_all': [50, 60],
        'Hour': [11, 18],
        'DayOfWeek': [6, 9],
        'Month': [9, 9],
        'Congestion': [4, 4]
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

def test_run_pipeline(test_data):
    """Test running the entire pipeline."""
    accuracy = run_pipeline('data/traffic.csv')
    assert isinstance(accuracy, float)
