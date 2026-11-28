from sklearn.ensemble import RandomForestClassifier
import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

@pytest.fixture
def data():
    """Fixture to load and preprocess data."""
    file_path = 'data/fraud.csv'
    data = load_data(file_path)
    X, y = preprocess_data(data)
    return X, y

def test_load_data():
    """Test loading data from a CSV file."""
    with pytest.raises(ValueError):
        load_data('nonexistent_file.csv')
    with pytest.raises(ValueError):
        load_data('data/empty.csv')

def test_preprocess_data(data):
    """Test preprocessing the data."""
    X, y = data
    assert X.shape[0] > 0
    assert y.shape[0] > 0

def test_train_model(data):
    """Test training the model."""
    X, y = data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model(data):
    """Test evaluating the model."""
    X, y = data
    model = train_model(X, y)
    accuracy = evaluate_model(model, X, y)
    assert 0 <= accuracy <= 1
