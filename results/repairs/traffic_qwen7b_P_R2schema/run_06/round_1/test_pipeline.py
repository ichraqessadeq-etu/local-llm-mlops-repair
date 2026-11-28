from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    data = load_data('data/traffic.csv')
    assert isinstance(data, pd.DataFrame)
    assert 'Congestion' in data.columns

def test_preprocess_data():
    data = load_data('data/traffic.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

def test_train_model():
    data = load_data('data/traffic.csv')
    X_train, _, y_train, _ = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
