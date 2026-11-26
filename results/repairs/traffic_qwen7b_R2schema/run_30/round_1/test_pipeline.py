from sklearn.ensemble import RandomForestClassifier
import pipeline
import pandas as pd
from sklearn.model_selection import train_test_split

def test_load_data():
    data = pipeline.load_data('data/traffic.csv')
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

def test_preprocess_data():
    data = pipeline.load_data('data/traffic.csv')
    X_train, X_test, y_train, y_test = pipeline.preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert len(X_train) + len(X_test) == len(data)
    assert len(y_train) + len(y_test) == len(data)

def test_train_model():
    data = pipeline.load_data('data/traffic.csv')
    X_train, X_test, y_train, y_test = pipeline.preprocess_data(data)
    model = pipeline.train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100
    assert model.max_depth == 10
    assert model.min_samples_split == 2
