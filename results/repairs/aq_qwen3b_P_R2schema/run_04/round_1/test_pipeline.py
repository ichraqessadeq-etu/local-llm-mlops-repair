from sklearn.model_selection import train_test_split
import unittest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.file_path = 'data/air_quality.csv'

    def test_load_data(self):
        data = load_data(self.file_path)
        self.assertIsNotNone(data)

    def test_preprocess_data(self):
        data = load_data(self.file_path)
        X, y = preprocess_data(data)
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)

    def test_train_model(self):
        data = load_data(self.file_path)
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        self.assertIsNotNone(model)

    def test_evaluate_model(self):
        data = load_data(self.file_path)
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        self.assertGreater(accuracy, 0.5)

if __name__ == "__main__":
    unittest.main()
