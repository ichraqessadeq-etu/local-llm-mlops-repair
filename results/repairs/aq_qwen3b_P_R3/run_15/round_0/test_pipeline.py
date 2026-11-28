import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model, save_model

@pytest.fixture
def data():
    file_path = 'data/air_quality.csv'
    return load_data(file_path)

def test_load_data(data):
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data(data):
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model():
    X_train, X_test, y_train, y_test = preprocess_data(load_data('data/air_quality.csv'))
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    X_train, X_test, y_train, y_test = preprocess_data(load_data('data/air_quality.csv'))
    model = train_model(X_train, y_test)
    report, confusion_matrix = evaluate_model(model, X_test, y_test)
    assert isinstance(report, str)
    assert isinstance(confusion_matrix, pd.DataFrame)

def test_save_model():
    X_train, X_test, y_train, y_test = preprocess_data(load_data('data/air_quality.csv'))
    model = train_model(X_train, y_test)
    save_model(model)
