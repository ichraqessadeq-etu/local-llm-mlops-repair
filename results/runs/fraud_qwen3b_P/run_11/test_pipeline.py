import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model, run_pipeline

def test_load_data():
    data = load_data('data/fraud.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_train_model():
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float)
    assert isinstance(report, str)

def test_run_pipeline():
    accuracy, report = run_pipeline('data/fraud.csv')
    assert isinstance(accuracy, float)
    assert isinstance(report, str)
