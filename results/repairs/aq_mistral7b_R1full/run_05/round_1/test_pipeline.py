import unittest
from pipeline import load_data, preprocess_data, train_model
import tempfile
import os

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data('data/air_quality.csv')
        self.X, self.y = preprocess_data(self.data)

    def test_train_model(self):
        temp_dir = tempfile.TemporaryDirectory()
        os.chdir(temp_dir.name)
        with self.assertRaises(TypeError):
            train_model()
        train_model(random_state=42)
        # Add assertions to test the model here

    def tearDown(self):
        temp_dir.cleanup()

if __name__ == '__main__':
    unittest.main()
