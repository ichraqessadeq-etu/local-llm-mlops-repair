import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from pipeline import load_data, preprocess_data  # Import both load_data and preprocess_data from pipeline.py

def test_load_data():
    data = load_data('data/fraud.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    data = load_data('data/fraud.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model():
    data = load_data('data/fraud.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    data = load_data('data/fraud.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float)
