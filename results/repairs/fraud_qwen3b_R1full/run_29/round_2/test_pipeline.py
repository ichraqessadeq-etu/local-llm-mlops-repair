import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

@pytest.fixture
def data():
    return load_data('data/fraud.csv')

def test_load_data(data):
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data(data):
    X_train, y_train, X_test, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)

def test_train_model(data):
    model = train_model(*preprocess_data(data))
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model(data):
    model = train_model(*preprocess_data(data))
    evaluate_model(model, *preprocess_data(data))
