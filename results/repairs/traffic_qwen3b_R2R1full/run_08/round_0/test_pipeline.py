import unittest
from pipeline import load_data, preprocess_data, train_model, save_model, run_pipeline

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/traffic.csv')
        self.assertIsNotNone(data)

    def test_preprocess_data(self):
        data = load_data('data/traffic.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)

    def test_train_model(self):
        data = load_data('data/traffic.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        self.assertIsNotNone(model)

    def test_save_model(self):
        data = load_data('data/traffic.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        save_model(model, 'models/random_forest_congestion_detector.pkl')

    def test_run_pipeline(self):
        try:
            run_pipeline('data/traffic.csv')
            self.assertTrue(True)
        except Exception as e:
            self.assertFalse(True, f"Pipeline failed: {e}")

if __name__ == '__main__':
    unittest.main()
