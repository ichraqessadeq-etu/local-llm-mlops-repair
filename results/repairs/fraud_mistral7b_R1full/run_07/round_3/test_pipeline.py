import unittest
from pipeline import load_data, preprocess_data, train_model
from sklearn.model_selection import train_test_split

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data("data/fraud.csv")
        self.X, self.y = preprocess_data(self.data)
        self.X_train, self.y_train, self.X_test, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def test_train_model(self):
        model = train_model(self.X_train, self.y_train)
        # Add your assertions here to test the model

if __name__ == "__main__":
    unittest.main()

