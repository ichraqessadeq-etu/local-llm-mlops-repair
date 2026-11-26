import pandas as pd
from sklearn.model_selection import train_test_split
import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data("data/fraud.csv")
        self.assertIsInstance(data, pd.DataFrame)

    def test_preprocess_data(self):
        data, _ = preprocess_data(load_data("data/fraud.csv"))
        self.assertIsInstance(data, pd.DataFrame)

    def test_train_model(self):
        data, y = preprocess_data(load_data("data/fraud.csv"))
        X_train, X_test, y_train, y_test = train_test_split(data.drop('Class', axis=1), y, test_size=0.2, random_state=42)
        y_type = type_of_target(y)
        if y_type not in [
            "binary",
            "multiclass",
            "multiclass-multioutput",
            "multilabel-indicator",
            "multilabel-sequences",
        ]:
            y_type = "continuous"
        self.assertIn(y_type, ["continuous"])
