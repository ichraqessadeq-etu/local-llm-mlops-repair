from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test the load_data function."""
    data = load_data('data/fraud.csv')
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

def test_preprocess_data():
    """Test the preprocess_data function."""
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert not X.empty
    assert not y.empty

def test_train_model():
    """Test the train_model function."""
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    # Check if the model is fitted by checking if the feature importance is not empty
    assert not model.feature_importances_.size == 0

