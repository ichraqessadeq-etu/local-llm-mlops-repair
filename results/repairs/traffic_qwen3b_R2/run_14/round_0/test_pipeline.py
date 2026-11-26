import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model, run_pipeline

@pytest.fixture
def data():
    """Fixture to load and preprocess data."""
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    return X, y

def test_load_data():
    """Test loading data."""
    with pytest.raises(FileNotFoundError):
        load_data('nonexistent_file.csv')

def test_preprocess_data():
    """Test preprocessing data."""
    X, y = preprocess_data(pd.DataFrame())
    assert len(X.columns) == 10  # Assuming 10 features including 'Congestion'

def test_train_model():
    """Test training the model."""
    X, y = pd.DataFrame(), pd.Series([0, 1, 0, 1])
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    """Test evaluating the model."""
    X, y = pd.DataFrame(), pd.Series([0, 1, 0, 1])
    model = RandomForestClassifier()
    model.fit(X, y)
    accuracy, report = evaluate_model(model, X, y)
    assert accuracy > 0.5

def test_run_pipeline():
    """Test running the entire pipeline."""
    with pytest.raises(Exception):
        run_pipeline('nonexistent_file.csv')
