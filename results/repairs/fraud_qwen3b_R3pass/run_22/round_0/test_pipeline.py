import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

@pytest.fixture
def data():
    # Load a small subset of the dataset for testing
    data = pd.read_csv('data/fraud.csv', nrows=1000)
    return data

def test_load_data(data):
    """Test the load_data function."""
    assert isinstance(load_data('data/fraud.csv'), pd.DataFrame)

def test_preprocess_data(data):
    """Test the preprocess_data function."""
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model(data):
    """Test the train_model function."""
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model(data):
    """Test the evaluate_model function."""
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float)
