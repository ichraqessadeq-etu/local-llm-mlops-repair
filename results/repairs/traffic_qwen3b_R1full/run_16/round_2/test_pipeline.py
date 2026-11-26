import pytest
from pipeline import load_data, preprocess_data, train_model

@pytest.fixture
def data():
    """Fixture to load and preprocess data."""
    file_path = 'data/traffic.csv'
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    return X_train, X_test, y_train, y_test

def test_load_data():
    """Test loading data."""
    with pytest.raises(FileNotFoundError):
        load_data('nonexistent_file.csv')

def test_preprocess_data(data):
    """Test preprocessing data."""
    X_train, X_test, y_train, y_test = data
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0

def test_train_model(data):
    """Test training the model."""
    X_train, X_test, y_train, y_test = data
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
