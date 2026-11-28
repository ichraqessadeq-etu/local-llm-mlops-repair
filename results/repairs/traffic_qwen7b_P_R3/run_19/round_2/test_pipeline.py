from sklearn.ensemble import RandomForestClassifier
import mlflow
from sklearn.model_selection import train_test_split
# test_pipeline.py

import pandas as pd
from pipeline import load_data, preprocess_data, train_model, evaluate_model

def test_load_data():
    df = load_data('data/traffic.csv')
    assert isinstance(df, pd.DataFrame)
    assert 'Congestion' in df.columns

def test_preprocess_data():
    df = load_data('data/traffic.csv')
    X, y = preprocess_data(df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert 'Congestion' not in X.columns

def test_train_model():
    df = load_data('data/traffic.csv')
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert 'model' in mlflow.list_artifacts()

def test_evaluate_model():
    df = load_data('data/traffic.csv')
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1
