from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import pytest
from pipeline import load_data, preprocess_data, train_model, run_pipeline

def test_load_data():
    """Test loading data."""
    data = load_data("data/traffic.csv")
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test preprocessing data."""
    data = load_data("data/traffic.csv")
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_train_model():
    """Test training the model."""
    data = load_data("data/traffic.csv")
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_run_pipeline():
    """Test the entire pipeline."""
    with pytest.raises(Exception):
        run_pipeline("non_existent_file.csv")

if __name__ == "__main__":
    pytest.main()
