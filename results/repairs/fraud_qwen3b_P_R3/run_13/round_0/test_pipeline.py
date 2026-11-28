import pytest
from pipeline import load_data, preprocess_data, train_model, save_model

@pytest.fixture
def data():
    """Fixture to load and preprocess data."""
    data_path = 'data/fraud.csv'
    data = load_data(data_path)
    X, y = preprocess_data(data)
    return X, y

def test_load_data():
    """Test loading data."""
    with pytest.raises(FileNotFoundError):
        load_data('nonexistent_file.csv')

def test_preprocess_data():
    """Test preprocessing data."""
    with pytest.raises(KeyError):
        preprocess_data(pd.DataFrame())

def test_train_model():
    """Test training the model."""
    X, y = data()
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_save_model():
    """Test saving the model."""
    X, y = data()
    model = train_model(X, y)
    save_model(model, 'fraud_detection', 'test_run')
