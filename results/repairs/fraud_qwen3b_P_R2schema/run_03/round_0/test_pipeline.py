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
    """Test loading data."""
    with pytest.raises(Exception):
        load_data('non_existent_file.csv')

def test_preprocess_data():
    """Test preprocessing data."""
    X, y = data()
    assert X.shape[0] > 0
    assert y.shape[0] > 0

def test_train_model():
    """Test training the model."""
    X, y = data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert model is not None

def test_evaluate_model():
    """Test evaluating the model."""
    X, y = data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert accuracy >= 0.5
