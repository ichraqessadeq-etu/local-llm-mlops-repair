import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data('data/fraud.csv')
        self.X, self.y = preprocess_data(self.data)

    def test_train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        model = train_model(X_train, y_train)
        # Add your assertions here to test the trained model

if __name__ == '__main__':
    unittest.main()
