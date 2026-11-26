import pytest
from pipeline import load_data, preprocess_data, train_model

@pytest.fixture(scope='module')
def data():
    return load_data('data/fraud.csv')

def test_preprocess_data(data):
    X, y = preprocess_data(data)
    assert len(X) > 0 and len(y) > 0

def test_train_model(data):
    X, y, Amount = preprocess_data(data)
    model = train_model(X_train=X, y_train=y, Amount=Amount)
    # Add assertions to check model's performance on a validation set
