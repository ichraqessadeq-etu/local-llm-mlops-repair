from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, predict, evaluate_model, run_pipeline

def test_load_data():
    with pytest.raises(FileNotFoundError):
        load_data('nonexistent_file.csv')
    with pytest.raises(pd.errors.EmptyDataError):
        load_data('data/empty.csv')
    with pytest.raises(pd.errors.ParserError):
        load_data('data/invalid_format.csv')

def test_preprocess_data():
    data = pd.DataFrame({'PT08.S1(CO)': [1, 2, 3], 'Anomaly': [0, 1, 0]})
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (3, 1)
    assert y.shape == (3,)

def test_train_model():
    X = pd.DataFrame({'PT08.S1(CO)': [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_predict():
    model = RandomForestClassifier(random_state=42)
    X = pd.DataFrame({'PT08.S1(CO)': [1, 2, 3]})
    y_pred = predict(model, X)
    assert isinstance(y_pred, np.ndarray)

def test_evaluate_model():
    y_true = pd.Series([0, 1, 0])
    y_pred = pd.Series([0, 1, 0])
    accuracy = evaluate_model(y_true, y_pred)
    assert accuracy == 1.0

def test_run_pipeline():
    run_pipeline('data/air_quality.csv')