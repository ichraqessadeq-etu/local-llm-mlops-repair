import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model, run_pipeline

def test_load_data():
    """Test the load_data function."""
    data = load_data("data/fraud.csv")
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test the preprocess_data function."""
    data = pd.read_csv("data/fraud.csv")
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model():
    """Test the train_model function."""
    data = pd.read_csv("data/fraud.csv")
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    """Test the evaluate_model function."""
    data = pd.read_csv("data/fraud.csv")
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_test)
    evaluate_model(model, X_test, y_test)

def test_run_pipeline():
    """Test the run_pipeline function."""
    try:
        run_pipeline("data/fraud.csv")
    except Exception as e:
        pytest.fail(f"Pipeline failed: {e}")

if __name__ == "__main__":
    pytest.main()
