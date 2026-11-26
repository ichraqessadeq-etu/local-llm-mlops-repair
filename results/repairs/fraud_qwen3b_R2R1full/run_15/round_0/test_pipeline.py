import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

def test_load_data():
    data = load_data('data/fraud.csv')
    assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame."

def test_preprocess_data():
    data = load_data('data/fraud.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame), "X_train should be a pandas DataFrame."
    assert isinstance(X_test, pd.DataFrame), "X_test should be a pandas DataFrame."
    assert isinstance(y_train, pd.Series), "y_train should be a pandas Series."
    assert isinstance(y_test, pd.Series), "y_test should be a pandas Series."

def test_train_model():
    data = load_data('data/fraud.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model should be a RandomForestClassifier."

def test_evaluate_model():
    data = load_data('data/fraud.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float), "Accuracy should be a float."
