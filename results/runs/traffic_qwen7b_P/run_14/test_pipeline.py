import pandas as pd
from pipeline import load_data, preprocess_data, train_model, predict

def test_load_data():
    data = load_data('data/traffic.csv')
    assert not data.empty
    assert 'Congestion' in data.columns

def test_preprocess_data():
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    assert not X.empty
    assert not y.empty
    assert len(X) == len(y)

def test_train_model():
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert model.is_fitted_

def test_predict():
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    model = train_model(X, y)
    X_test, _ = preprocess_data(data.sample(frac=0.2, random_state=42))
    predictions = predict(model, X_test)
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(X_test)
