import unittest
import tempfile
import pandas as pd
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'feature': [20, 25, 30, 22, 28],
            'Congestion': [0, 1, 0, 1, 0]
        })

    def tearDown(self):
        pass

    def test_preprocess_data(self):
        X, y = preprocess_data(self.data)
        self.assertIsInstance(X, pd.Series)
        self.assertIsInstance(y, pd.Series)

    def test_train_model(self):
        X_train, y_train = preprocess_data(self.data)
        model = train_model(X_train, y_train)
        self.assertIsInstance(model, RandomForestClassifier)

if __name__ == "__main__":
    unittest.main()
