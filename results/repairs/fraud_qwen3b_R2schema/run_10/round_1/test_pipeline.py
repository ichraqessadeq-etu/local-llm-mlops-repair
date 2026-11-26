import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import pytest
import pipeline

def test_load_data():
    """Test the load_data function."""
    try:
        data = pipeline.load_data("data/fraud.csv")
        assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame."
        logging.info("Test load_data passed.")
    except Exception as e:
        logging.error(f"Test load_data failed: {e}")

def test_preprocess_data():
    """Test the preprocess_data function."""
    try:
        data = pipeline.load_data("data/fraud.csv")
        X, y = pipeline.preprocess_data(data)
        assert isinstance(X, pd.DataFrame), "X should be a pandas DataFrame."
        assert isinstance(y, pd.Series), "y should be a pandas Series."
        logging.info("Test preprocess_data passed.")
    except Exception as e:
        logging.error(f"Test preprocess_data failed: {e}")

def test_train_model():
    """Test the train_model function."""
    try:
        data = pipeline.load_data("data/fraud.csv")
        X, y = pipeline.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = pipeline.train_model(X_train, y_train)
        assert isinstance(model, pipeline.RandomForestClassifier), "Model should be a RandomForestClassifier."
        logging.info("Test train_model passed.")
    except Exception as e:
        logging.error(f"Test train_model failed: {e}")

def test_evaluate_model():
    """Test the evaluate_model function."""
    try:
        data = pipeline.load_data("data/fraud.csv")
        X, y = pipeline.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = pipeline.train_model(X_train, y_train)
        accuracy = pipeline.evaluate_model(model, X_test, y_test)
        assert isinstance(accuracy, float), "Accuracy should be a float."
        logging.info("Test evaluate_model passed.")
    except Exception as e:
        logging.error(f"Test evaluate_model failed: {e}")

def test_main():
    """Test the main function."""
    try:
        pipeline.main("data/fraud.csv")
        logging.info("Test main passed.")
    except Exception as e:
        logging.error(f"Test main failed: {e}")

if __name__ == "__main__":
    test_load_data()
    test_preprocess_data()
    test_train_model()
    test_evaluate_model()
    test_main()
