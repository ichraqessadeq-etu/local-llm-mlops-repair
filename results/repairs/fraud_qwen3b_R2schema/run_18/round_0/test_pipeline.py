import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

@pytest.fixture
def data():
    """Load the dataset for testing."""
    file_path = 'data/fraud.csv'
    return load_data(file_path)

def test_load_data(data):
    """Test if data is loaded correctly."""
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data(data):
    """Test if data is preprocessed correctly."""
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model(data):
    """Test if the model is trained correctly."""
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model(data):
    """Test if the model is evaluated correctly."""
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float)
