from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    data = load_data('data/fraud.csv')
    assert not data.empty
    assert 'Class' in data.columns

def test_preprocess_data():
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    assert X.shape[1] == 29  # 30 features minus the target
    assert y.shape[0] == X.shape[0]

def test_train_model():
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    assert isinstance(y_pred, pd.Series)
    assert accuracy_score(y_test, y_pred) >= 0.5
