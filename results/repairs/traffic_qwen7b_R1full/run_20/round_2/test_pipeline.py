import pandas as pd
from pipeline import load_data, preprocess_data, train_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_load_data():
    data = load_data("data/traffic.csv")
    assert not data.empty

def test_preprocess_data():
    data = load_data("data/traffic.csv")
    X, y = preprocess_data(data)
    assert X.shape[1] == 9  # 8 features + 1 target
    assert y.shape[0] == data.shape[0]

def test_train_model():
    data = load_data("data/traffic.csv")
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    assert accuracy_score(y_test, y_pred) > 0.5
