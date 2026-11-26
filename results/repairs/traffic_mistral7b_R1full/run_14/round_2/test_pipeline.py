import unittest
import tempfile
import os
import pandas as pd
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.file_path = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
        data = {'Features': [0, 1, 2, 3, 4], 'Congestion': [0, 1, 0, 1, 0]}
        pd.DataFrame(data).to_csv(self.file_path)

    def tearDown(self):
        os.remove(self.file_path)

    def test_load_data(self):
        data = load_data(self.file_path)
        self.assertIsInstance(data, pd.DataFrame)

    def test_preprocess_data(self):
        data = load_data(self.file_path)
        X, y = preprocess_data(data)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_train_model(self):
        data = load_data(self.file_path)
        X, y = preprocess_data(data)
        model = train_model(X, y)
        self.assertIsInstance(model, RandomForestClassifier)

if __name__ == '__main__':
    unittest.main()
