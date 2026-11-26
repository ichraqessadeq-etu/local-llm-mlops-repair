import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def test_load_data():
    with pytest.raises(ValueError):
        load_data('nonexistent_file.csv')
    with pytest.raises(ValueError):
        load_data('data/fraud.csv')

def test_preprocess_data():
    data = pd.DataFrame({
        'Class': [0, 1, 0, 1],
        'V1': [1, 2, 3, 4],
        'V2': [5, 6, 7, 8]
    })
    X, y = preprocess_data(data)
    assert X.shape == (4, 2)
    assert y.shape == (4,)

def test_train_model():
    data = pd.DataFrame({
        'Class': [0, 1, 0, 1],
        'V1': [1, 2, 3, 4],
        'V2': [5, 6, 7, 8]
    })
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
