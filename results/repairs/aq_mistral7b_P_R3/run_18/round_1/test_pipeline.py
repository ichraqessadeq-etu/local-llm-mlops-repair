from pipeline import load_data, preprocess_data
import pytest
import pandas as pd

@pytest.fixture(scope="module")
def data():
    return load_data('data/air_quality.csv')

def test_load_data(data):
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data(data):
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
