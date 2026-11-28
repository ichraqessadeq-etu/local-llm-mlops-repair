from sklearn.model_selection import train_test_split
import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data('data/fraud.csv')
        cls.X, cls.y = preprocess_data(cls.data)

    def test_train_model(self):
        X_train, y_train, _, _ = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        train_model(X_train, y_train)

if __name__ == '__main__':
    unittest.main()
