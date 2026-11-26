from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test the load_data function."""
    data = load_data('data/fraud.csv')
    assert not data.empty
    assert 'Class' in data.columns

def test_preprocess_data():
    """Test the preprocess_data function."""
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    assert X.shape[1] == 29  # 29 features
    assert y.shape[0] == X.shape[0]  # Same number of samples

def test_train_model():
    """Test the train_model function."""
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert model is not None
    assert isinstance(model, RandomForestClassifier)
