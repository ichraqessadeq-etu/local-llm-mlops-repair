import unittest
from pipeline import load_data, preprocess_data, train_model
import randomforest_opus as rf

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data("data/traffic.csv")
        self.X, self.y = preprocess_data(self.data)

    def test_load_data(self):
        self.assertIsInstance(self.data, pd.DataFrame)

    def test_preprocess_data(self):
        self.assertIsInstance(self.X, np.ndarray)
        self.assertIsInstance(self.y, np.ndarray)

    def test_train_model(self):
        model = train_model(self.X[:-100], self.y[:-100])
        # Add assertions to check model properties or performance metrics

if __name__ == "__main__":
    unittest.main()
