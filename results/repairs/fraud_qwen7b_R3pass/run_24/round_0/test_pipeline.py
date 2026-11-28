import pandas as pd
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test data loading."""
    data = load_data('data/fraud.csv')
    assert not data.empty
    assert 'Class' in data.columns

def test_preprocess_data():
    """Test data preprocessing."""
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    assert not X.empty
    assert not y.empty
    assert len(X) == len(y)

def test_train_model():
    """Test model training."""
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert model is not None
    y_pred = model.predict(X_test)
    assert accuracy_score(y_test, y_pred) > 0.5
