import unittest
from pipeline import load_data, preprocess_data, train_model, main

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/traffic.csv')
        self.assertIsNotNone(data)
        self.assertIn('Congestion', data.columns)

    def test_preprocess_data(self):
        data = load_data('data/traffic.csv')
        X, y = preprocess_data(data)
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertEqual(X.shape[1], 9)

    def test_train_model(self):
        data = load_data('data/traffic.csv')
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        self.assertIsNotNone(model)

    def test_main(self):
        try:
            main()
        except Exception as e:
            self.fail(f"Main function failed with error: {str(e)}")

if __name__ == "__main__":
    unittest.main()
