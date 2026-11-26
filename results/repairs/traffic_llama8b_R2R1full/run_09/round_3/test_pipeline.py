from sklearn.ensemble import RandomForestClassifier
import pytest
from pipeline import load_data, preprocess_data, train_model
import warnings

def test_load_data():
    data = load_data('data/traffic.csv')
    assert data.shape[0] > 0

def test_preprocess_data():
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    assert X.shape[0] > 0
    assert y.shape[0] > 0

def test_train_model():
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_warnings():
    with pytest.warns(None) as record:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit([[1, 2], [3, 4]], [0, 1])
    assert len(record.list) == 0

def test_mlflow_log_param():
    import mlflow
    with pytest.raises(AttributeError):
        mlflow.log_param("error", "Test error")
