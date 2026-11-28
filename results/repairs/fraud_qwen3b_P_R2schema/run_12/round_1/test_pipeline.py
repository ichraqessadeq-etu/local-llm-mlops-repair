from sklearn.model_selection import train_test_split
# test_pipeline.py

import unittest
import pipeline

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        file_path = 'data/fraud.csv'
        data = pipeline.load_data(file_path)
        self.assertIsNotNone(data)

    def test_preprocess_data(self):
        file_path = 'data/fraud.csv'
        data = pipeline.load_data(file_path)
        X, y = pipeline.preprocess_data(data)
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)

    def test_train_model(self):
        file_path = 'data/fraud.csv'
        data = pipeline.load_data(file_path)
        X, y = pipeline.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = pipeline.train_model(X_train, y_train)
        self.assertIsNotNone(model)

    def test_evaluate_model(self):
        file_path = 'data/fraud.csv'
        data = pipeline.load_data(file_path)
        X, y = pipeline.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = pipeline.train_model(X_train, y_train)
        accuracy = pipeline.evaluate_model(model, X_test, y_test)
        self.assertGreater(accuracy, 0.5)

if __name__ == '__main__':
    unittest.main()
