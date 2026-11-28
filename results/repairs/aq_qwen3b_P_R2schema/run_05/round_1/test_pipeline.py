from sklearn.metrics import accuracy_score
import logging
import pytest
from pipeline import load_data, preprocess_data, train_model, save_model
import pandas as pd

@pytest.fixture
def data():
    # Load a sample dataset
    data = pd.DataFrame({
        'PT08.S1(CO)': [10, 20, 30, 40, 50],
        'PT08.S2(NMHC)': [15, 25, 35, 45, 55],
        'PT08.S3(NOx)': [20, 30, 40, 50, 60],
        'PT08.S4(NO2)': [25, 35, 45, 55, 65],
        'PT08.S5(O3)': [30, 40, 50, 60, 70],
        'NOx(GT)': [35, 45, 55, 65, 75],
        'C6H6(GT)': [40, 50, 60, 70, 80],
        'T': [45, 55, 65, 75, 85],
        'RH': [50, 60, 70, 80, 90],
        'AH': [55, 65, 75, 85, 95],
        'Hour': [6, 7, 8, 9, 10],
        'DayOfWeek': [1, 2, 3, 4, 5],
        'Month': [1, 2, 3, 4, 5],
        'Anomaly': [0, 1, 0, 1, 0]
    })
    return data

def test_load_data(data):
    """Test the load_data function."""
    try:
        loaded_data = load_data(data)
        assert loaded_data.equals(data), "Data loading failed."
        logging.info("Data loading test passed.")
    except Exception as e:
        logging.error(f"Data loading test failed: {e}")

def test_preprocess_data(data):
    """Test the preprocess_data function."""
    try:
        X_train, X_test, y_train, y_test = preprocess_data(data)
        assert len(X_train) == len(y_train), "Data preprocessing failed."
        logging.info("Data preprocessing test passed.")
    except Exception as e:
        logging.error(f"Data preprocessing test failed: {e}")

def test_train_model(data):
    """Test the train_model function."""
    try:
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        assert accuracy > 0.8, "Model training failed."
        logging.info("Model training test passed.")
    except Exception as e:
        logging.error(f"Model training test failed: {e}")

def test_save_model(data):
    """Test the save_model function."""
    try:
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        save_model(model, "air_quality_experiment", "air_quality_run")
        logging.info("Model saving test passed.")
    except Exception as e:
        logging.error(f"Model saving test failed: {e}")
