from pipeline import load_data, preprocess_data, train_model
import pytest
import numpy as np

@pytest.fixture(scope="module")
def data():
    return load_data("data/traffic.csv")

def test_train_model(data):
    X, y = preprocess_data(data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)

    # Add assertions to check the model's performance and parameters
    assert len(X_train) == len(y_train) == len(model.classes_) == len(np.unique(y_train))
