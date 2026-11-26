import pytest
from pipeline import load_data, preprocess_data, train_model

@pytest.fixture(scope='module')
def data():
    return load_data('data/fraud.csv')

def test_preprocess_data(data):
    X, y = preprocess_data(data)
    assert len(X) > 0 and len(y) > 0

def test_train_model(data):
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    # Add assertions to check model's performance on a validation set
