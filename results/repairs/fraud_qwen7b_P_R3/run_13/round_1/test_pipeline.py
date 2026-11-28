from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test the load_data function."""
    data = load_data('data/fraud.csv')
    assert isinstance(data, pd.DataFrame)
    assert 'Class' in data.columns

def test_preprocess_data():
    """Test the preprocess_data function."""
    data = load_data('data/fraud.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert X_train.shape[1] == 29  # 29 features after scaling
    assert y_train.shape[0] == X_train.shape[0]

def test_train_model():
    """Test the train_model function."""
    data = load_data('data/fraud.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100
