from sklearn.ensemble import RandomForestClassifier
import pytest
import pandas as pd
from pipeline import load_data, preprocess_data, train_model

@pytest.fixture
def data():
    # Create a sample DataFrame
    data = pd.DataFrame({
        'Time': [1, 2, 3],
        'V1': [4, 5, 6],
        'V2': [7, 8, 9],
        'Class': [0, 1, 0]
    })
    return data

def test_load_data(data):
    file_path = 'data/fraud.csv'
    with pytest.raises(MlflowException):
        load_data(file_path)

def test_preprocess_data(data):
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model(data):
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

