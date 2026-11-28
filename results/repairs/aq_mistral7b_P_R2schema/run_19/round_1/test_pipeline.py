import os
import unittest
from pipeline import load_data, preprocess_data, train_model, save_model_and_data
from sklearn.metrics import accuracy_score

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data('data/air_quality.csv')
        self.X, self.y = preprocess_data(self.data)

    def test_train_model(self):
        model = train_model(self.X, self.y)
        X_test, y_test = preprocess_data(load_data('test_data/air_quality.csv'))
        y_pred = model.predict(X_test)
        self.assertGreaterEqual(accuracy_score(y_test, y_pred), 0.8)

    def test_save_model_and_data(self):
        model = train_model(self.X, self.y)
        save_model_and_data(model, 'test_artifacts')
        assert os.path.exists('test_artifacts/model')
        assert os.path.exists('test_artifacts/data')
