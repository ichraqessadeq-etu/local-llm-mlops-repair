import pytest
from pipeline import load_data, preprocess_data, train_model, save_model

@pytest.fixture
def data():
    # Load a sample dataset for testing
    data = pd.DataFrame({
        'temp': [20, 22, 18, 25, 21],
        'rain_1h': [0, 1, 0, 1, 0],
        'snow_1h': [0, 0, 1, 0, 0],
        'clouds_all': [50, 70, 30, 80, 60],
        'weather_main': ['Clear', 'Clouds', 'Rain', 'Snow', 'Clear'],
        'IsHoliday': [0, 1, 0, 1, 0],
        'Hour': [8, 14, 10, 16, 12],
        'DayOfWeek': [1, 2, 3, 4, 5],
        'Month': [1, 2, 3, 4, 5],
        'Congestion': [0, 1, 0, 1, 0]
    })
    return data

def test_load_data(data):
    """Test the load_data function."""
    try:
        loaded_data = load_data(data)
        assert isinstance(loaded_data, pd.DataFrame), "Data should be a pandas DataFrame."
        assert len(loaded_data) == len(data), "Data should have the same number of rows."
    except Exception as e:
        pytest.fail(f"Test failed: {e}")

def test_preprocess_data(data):
    """Test the preprocess_data function."""
    try:
        X_train, X_test, y_train, y_test = preprocess_data(data)
        assert isinstance(X_train, pd.DataFrame), "X_train should be a pandas DataFrame."
        assert isinstance(X_test, pd.DataFrame), "X_test should be a pandas DataFrame."
        assert isinstance(y_train, pd.Series), "y_train should be a pandas Series."
        assert isinstance(y_test, pd.Series), "y_test should be a pandas Series."
        assert len(X_train) + len(X_test) == len(data), "X_train and X_test should have the same number of rows."
        assert len(y_train) + len(y_test) == len(data), "y_train and y_test should have the same number of rows."
    except Exception as e:
        pytest.fail(f"Test failed: {e}")

def test_train_model(data):
    """Test the train_model function."""
    try:
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        assert isinstance(model, RandomForestClassifier), "Model should be an instance of RandomForestClassifier."
    except Exception as e:
        pytest.fail(f"Test failed: {e}")

def test_save_model(data):
    """Test the save_model function."""
    try:
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        save_model(model, "congestion_detector")
        assert mlflow.get_experiment_by_name("congestion_detector") is not None, "Experiment should be saved."
    except Exception as e:
        pytest.fail(f"Test failed: {e}")
