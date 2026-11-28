from sklearn.model_selection import train_test_split
from pipeline import load_data, preprocess_data, train_model
import pytest
import pandas as pd

def test_load_data():
    data = load_data("data/fraud.csv")
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    data = load_data("data/fraud.csv")
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_train_model():
    data = load_data("data/fraud.csv")
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    # Add assertions for model evaluation here

pytest.main(["-v", __file__])
