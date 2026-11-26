import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    data = load_data('data/fraud.csv')
    assert isinstance(data, pd.DataFrame)
    assert 'Class' in data.columns

def test_preprocess_data():
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)

def test_train_model():
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
