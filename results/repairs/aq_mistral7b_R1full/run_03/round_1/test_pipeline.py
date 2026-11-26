import unittest
import numpy as np
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data('data/air_quality.csv')
        self.X, self.y = preprocess_data(self.data)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

    def test_load_data(self):
        self.assertIsInstance(self.data, pd.DataFrame)

    def test_preprocess_data(self):
        X, y = preprocess_data(self.data)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_train_model(self):
        model = train_model(self.X_train, self.y_train)
        self.assertIsInstance(model, RandomForestClassifier)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
