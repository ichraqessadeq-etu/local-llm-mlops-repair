from sklearn.model_selection import train_test_split
import unittest
from pipeline import load_data, preprocess_data, train_model
import pickle

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data("data/air_quality.csv")
        self.X, self.y = preprocess_data(self.data)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

    def tearDown(self):
        if self.has_teardown_errors():
            raise self.failureException("test_train_model has teardown errors")
        try:
            os.remove("model.pkl")
        except OSError:
            pass

    def test_train_model(self):
        train_model(self.X_train, self.y_train)
        # Add more tests as needed
