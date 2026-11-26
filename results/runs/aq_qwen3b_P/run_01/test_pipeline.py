import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

def test_load_data():
    """Test loading data from a CSV file."""
    data = load_data('data/air_quality.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test preprocessing the data."""
    data = pd.DataFrame({
        'PT08.S1(CO)': [1, 2, 3],
        'Anomaly': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_train_model():
    """Test training the model."""
    data = pd.DataFrame({
        'PT08.S1(CO)': [1, 2, 3],
        'Anomaly': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    """Test evaluating the model."""
    data = pd.DataFrame({
        'PT08.S1(CO)': [1, 2, 3],
        'Anomaly': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    model = train_model(X, y)
    X_test = X[:2]
    y_test = y[:2]
    accuracy = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float)
