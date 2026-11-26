import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

def test_load_data():
    data = load_data('data/air_quality.csv')
    assert data is not None

def test_preprocess_data():
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    assert X is not None
    assert y is not None

def test_train_model():
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert model is not None

def test_evaluate_model():
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

def test_main():
    try:
        main()
    except Exception as e:
        pytest.fail(f"main() raised an exception: {e}")
