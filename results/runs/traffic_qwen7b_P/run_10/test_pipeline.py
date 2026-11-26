import pandas as pd
from pipeline import load_data, preprocess_data, train_model, evaluate_model

def test_load_data():
    data = load_data('data/traffic.csv')
    assert isinstance(data, pd.DataFrame)
    assert 'Congestion' in data.columns

def test_preprocess_data():
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == data.shape[0]
    assert y.shape[0] == data.shape[0]

def test_train_model():
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100
    assert model.random_state == 42

def test_evaluate_model():
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1
