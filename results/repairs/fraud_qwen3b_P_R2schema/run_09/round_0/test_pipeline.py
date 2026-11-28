import pytest
import pipeline

def test_load_data():
    """Test the load_data function."""
    data = pipeline.load_data('data/fraud.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test the preprocess_data function."""
    data = pipeline.load_data('data/fraud.csv')
    X_train, y_train, X_test, y_test = pipeline.preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)

def test_train_model():
    """Test the train_model function."""
    data = pipeline.load_data('data/fraud.csv')
    X_train, y_train, X_test, y_test = pipeline.preprocess_data(data)
    model = pipeline.train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
