import unittest
import pickle
import os
import shutil
from pipeline import train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Copy the model.pkl file from the pipeline.py directory to the test directory
        shutil.copy('../pipeline.py/model.pkl', 'model.pkl')

    def tearDown(self):
        # Remove the model.pkl file after testing
        os.remove('model.pkl')

    def test_train_model(self):
        # Load the model from the pickle file
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Check if the model is of type RandomForestClassifier
        self.assertIsInstance(model, RandomForestClassifier)

if __name__ == '__main__':
    unittest.main()
