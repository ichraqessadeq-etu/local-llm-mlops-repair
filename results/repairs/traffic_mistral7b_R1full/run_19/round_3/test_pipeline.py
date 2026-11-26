from pipeline import load_data, train_model
import pytest
import os
import numpy as np

def test_train_model():
    X, y = load_data(os.path.join('data', 'traffic.csv'))
    model = train_model(X_train=X, y_train=y)
    # Add assertions to check if the model is trained correctly
    assert isinstance(model, RandomForestClassifier)
