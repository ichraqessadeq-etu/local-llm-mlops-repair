# test_pipeline.py

import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model, run_pipeline

@pytest.fixture
def data():
    # Load a sample dataset
    return pd.DataFrame({
        'PT08.S1(CO)': [10, 20, 30],
        'PT08.S2(NMHC)': [40, 50, 60],
        'PT08.S3(NOx)': [70, 80, 90],
        'PT08.S4(NO2)': [100, 110, 120],
        'PT08.S5(O3)': [130, 140, 150],
        'NOx(GT)': [160, 170, 180],
        'C6H6(GT)': [190, 200, 210],
        'T': [220, 230, 240],
        'RH': [250, 260, 270],
        'AH': [280, 290, 300],
        'Hour': [1, 2, 3],
        'DayOfWeek': [4, 5, 6],
        'Month': [7, 8, 9],
        'Anomaly': [0, 1, 0]
    })

def test_load_data(data):
    """Test data loading."""
    with pytest.raises(FileNotFoundError):
        load_data('nonexistent_file.csv')
    with pytest.raises(pd.errors.EmptyDataError):
        load_data('empty.csv')
    with pytest.raises(pd.errors.ParserError):
        load_data('invalid.csv')
    loaded_data = load_data('data/air_quality.csv')
    assert isinstance(loaded_data, pd.DataFrame)

def test_preprocess_data(data):
    """Test data preprocessing."""
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model(data):
    """Test model training."""
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model(data):
    """Test model evaluation."""
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float)

def test_run_pipeline(data):
    """Test full pipeline."""
    run_pipeline('data/air_quality.csv')
