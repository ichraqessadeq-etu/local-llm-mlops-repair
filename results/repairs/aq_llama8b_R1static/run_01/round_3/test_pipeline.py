import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

def test_load_data():
    data = load_data('data/air_quality.csv')
    assert data is not None

def test_preprocess_data():
    data = load_data('data/air_quality.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None

def test_train_model():
    data = load_data('data/air_quality.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert model is not None

def test_evaluate_model():
    data = load_data('data/air_quality.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    assert y_pred is not None

def test_main():
    with pytest.raises(SystemExit):
        pipeline.main()
