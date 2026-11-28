import unittest
from pipeline import load_data, preprocess_data, train_model
from sklearn.metrics import accuracy_score

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data("data/traffic.csv")
        self.X, self.y = preprocess_data(self.data)
        self.X_train, self.y_train, self.X_test, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

    def test_train_model(self):
        model = train_model(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        expected_accuracy = accuracy_score(self.y_test, predictions)
        self.assertAlmostEqual(expected_accuracy, 0.8, delta=0.05)
