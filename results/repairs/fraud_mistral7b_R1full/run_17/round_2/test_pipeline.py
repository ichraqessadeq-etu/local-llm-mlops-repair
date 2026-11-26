import unittest
from pipeline import load_data, preprocess_data, train_model, serialize_model, load_model
import pandas as pd

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data("data/fraud.csv")

    def test_load_data(self):
        data = load_data("data/fraud.csv")
        self.assertIsInstance(data, pd.DataFrame)

    def test_preprocess_data(self):
        data, _ = preprocess_data(self.data)
        self.assertIsInstance(data, tuple)
        self.assertIsInstance(data[0], pd.DataFrame)
        self.assertIsInstance(data[1], pd.Series)

    def test_train_model(self):
        X, y = preprocess_data(self.data)
        model = train_model(X_train=X, y_train=y)
        self.assertIsInstance(model, RandomForestClassifier)

        # Adding a test to serialize and load the model
        serialize_model(model)
        loaded_model = load_model()
        self.assertIsInstance(loaded_model, RandomForestClassifier)

    def tearDown(self):
        if "model.pkl" in next(os.walk("."))[1]:
            os.remove("model.pkl")

if __name__ == "__main__":
    unittest.main()
