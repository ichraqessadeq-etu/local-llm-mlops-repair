import pytest
from pipeline import load_data, preprocess_data, train_model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

@pytest.fixture
def mock_data():
    X, y = make_classification(n_samples=1000, n_features=25, n_informative=20, n_redundant=5, random_state=42)
    df = pd.DataFrame(X, columns=[f'V{i}' for i in range(1, 26)])
    df['Class'] = y
    return df

def test_load_data(mock_data):
    """Test loading the data."""
    data = load_data(mock_data)
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data(mock_data):
    """Test preprocessing the data."""
    X_train, X_test, y_train, y_test = preprocess_data(mock_data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model(mock_data):
    """Test training the model."""
    X_train, X_test, y_train, y_test = preprocess_data(mock_data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
