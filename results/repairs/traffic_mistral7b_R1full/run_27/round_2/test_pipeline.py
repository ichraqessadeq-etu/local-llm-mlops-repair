import unittest
import tempfile
import os
from pipeline import load_data, preprocess_data, train_model, get_random_state, set_random_state

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data('data/traffic.csv')
        self.X, self.y = preprocess_data(self.data)
        self.random_state = get_random_state()

    def tearDown(self):
        set_random_state(self.random_state)

    def test_train_model(self):
        model = train_model(self.X, self.y, random_state=self.random_state)
        # Add your assertions here

if __name__ == '__main__':
    unittest.main()
