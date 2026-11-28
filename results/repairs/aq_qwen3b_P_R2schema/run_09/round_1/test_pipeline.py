from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, predict_anomaly

def test_load_data():
    data = load_data('data/air_quality.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_train_model():
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_predict_anomaly():
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    model = train_model(X, y)
    predictions = predict_anomaly(model, X)
    assert isinstance(predictions, np.ndarray)
