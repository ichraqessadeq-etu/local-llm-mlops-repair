import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data("data/traffic.csv")
        self.X, self.y = preprocess_data(self.data)

    def test_train_model(self):
        model = train_model(self.X_train, self.y_train)
        # Add your assertions here to test the model

if __name__ == "__main__":
    unittest.main()
