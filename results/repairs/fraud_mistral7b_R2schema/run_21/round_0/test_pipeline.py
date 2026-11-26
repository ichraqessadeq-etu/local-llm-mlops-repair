from pipeline import load_data, preprocess_data, train_model
from random_forest_classifier import train
import pytest
import pandas as pd

def test_train_model():
    data = load_data("data/fraud.csv")
    X, y = preprocess_data(data)
    model = train(X_train=X, y_train=y)
    # Add assertions to check the model's performance
