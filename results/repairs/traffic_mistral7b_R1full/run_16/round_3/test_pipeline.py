import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data('data/traffic.csv')
        self.X, self.y = preprocess_data(self.data)

    def test_load_data(self):
        self.assertIsInstance(self.data, pd.DataFrame)

    def test_preprocess_data(self):
        self.assertIsInstance(self.X, pd.DataFrame)
        self.assertIsInstance(self.y, pd.Series)

    def test_train_model(self):
        X_train, y_train, X_test, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        train_model(X_train, y_train)
        # Add more assertions as needed
