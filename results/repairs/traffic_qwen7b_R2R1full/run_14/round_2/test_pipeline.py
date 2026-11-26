from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from pipeline import load_data, preprocess_data, train_model, predict

def test_load_data():
    """Test loading data."""
    data = load_data('data/traffic.csv')
    assert not data.empty

def test_preprocess_data():
    """Test data preprocessing."""
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    assert X.shape[1] == len(data.columns) - 1
    assert y.shape[0] == data.shape[0]

def test_train_model():
    """Test model training."""
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    y_pred = predict(model, X_test)
    assert len(y_pred) == len(y_test)

def test_predict():
    """Test model prediction."""
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    y_pred = predict(model, X_test)
    assert len(y_pred) == len(y_test)

