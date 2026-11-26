import unittest
from pipeline import load_data, preprocess_data, train_model
import randomforest_opus as rf

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data("data/traffic.csv")
        self.assertIsInstance(data, pd.DataFrame)

    def test_preprocess_data(self):
        # Add a test for preprocess_data function

    def test_train_model(self):
        # Assuming you have X_train, y_train, X_test, y_test prepared
        X_train, y_train = preprocess_data(data)
        np.random.seed(42)  # Seed random state for reproducibility
        model = train_model(X_train, y_train)
        # Add assertions to check model properties or performance metrics

if __name__ == "__main__":
    unittest.main()
