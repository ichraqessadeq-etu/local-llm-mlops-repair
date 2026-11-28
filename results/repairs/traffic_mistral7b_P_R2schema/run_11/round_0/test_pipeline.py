import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data_path = "data/traffic.csv"
        self.X, self.y = None, None

    def test_load_data(self):
        data = load_data(self.data_path)
        self.assertTrue(len(data) > 0)

    def test_preprocess_data(self):
        data = load_data(self.data_path)
        self.X, self.y = preprocess_data(data)
        self.assertTrue(len(self.X) > 0 and len(self.y) > 0)

    def test_train_model(self):
        data = load_data(self.data_path)
        self.X, self.y = preprocess_data(data)
        X_train, y_train, X_test, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        model = train_model(X_train, y_train)
        # Add your custom assertions for the model performance here

if __name__ == "__main__":
    unittest.main()
