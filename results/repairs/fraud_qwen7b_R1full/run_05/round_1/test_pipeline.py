import pandas as pd
from pipeline import load_data, preprocess_data, train_model, evaluate_model
from sklearn.model_selection import train_test_split

def test_load_data():
    data = load_data('data/fraud.csv')
    assert not data.empty
    assert 'Class' in data.columns

def test_preprocess_data():
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    assert not X.empty
    assert not y.empty
    assert len(X) == len(y)

def test_train_model():
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert model is not None
    accuracy = evaluate_model(model, X_test, y_test)
    assert accuracy > 0.5

if __name__ == "__main__":
    test_load_data()
    test_preprocess_data()
    test_train_model()
