import pytest
import pipeline
import pandas as pd

def test_load_data():
    data = pipeline.load_data('data/fraud.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    data = pd.DataFrame({
        'Time': [1, 2, 3],
        'V1': [4, 5, 6],
        'V2': [7, 8, 9],
        'Class': [0, 1, 0]
    })
    X, y = pipeline.preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_train_model():
    data = pd.DataFrame({
        'Time': [1, 2, 3],
        'V1': [4, 5, 6],
        'V2': [7, 8, 9],
        'Class': [0, 1, 0]
    })
    X, y = pipeline.preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = pipeline.train_model(X_train, y_train)
    assert isinstance(model, pipeline.RandomForestClassifier)
