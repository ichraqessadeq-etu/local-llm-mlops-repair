import unittest
import pandas as pd
from pipeline import load_data, preprocess_data, train_model
from sklearn.model_selection import train_test_split

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data('data/air_quality.csv')

    def test_load_data(self):
        self.assertIsInstance(self.data, pd.DataFrame)

    def test_preprocess_data(self):
        X, y = preprocess_data(self.data)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_train_model(self):
        X_train, y_train = train_test_split(self.data.drop('Anomaly', axis=1), self.data['Anomaly'], test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        self.assertIsInstance(model, RandomForestClassifier)

if __name__ == '__main__':
    unittest.main()
