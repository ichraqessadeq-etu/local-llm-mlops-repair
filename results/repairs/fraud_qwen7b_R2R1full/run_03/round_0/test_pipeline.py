import pandas as pd
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    data = load_data('data/fraud.csv')
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

def test_preprocess_data():
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert not X.empty
    assert not y.empty

def test_train_model():
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100
