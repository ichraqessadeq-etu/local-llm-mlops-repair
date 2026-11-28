from sklearn.model_selection import train_test_split
import unittest
from pipeline import load_data, preprocess_data, train_model
from sklearn.metrics import accuracy_score

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data("data/fraud.csv")
        cls.X, cls.y = preprocess_data(cls.data)

    def test_train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)
        self.assertGreaterEqual(accuracy_score(y_test, predictions), 0.8)

if __name__ == "__main__":
    unittest.main()
