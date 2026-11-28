import unittest
from unittest.mock import patch
import pipeline
import pandas as pd

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data("data/air_quality.csv")
        cls.X, cls.y = preprocess_data(cls.data)

    def test_train_model(self):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            model = train_model(self.X_train, self.y_train)
            self.assertIsInstance(model, RandomForestClassifier)
            self.assertTrue(any(map(lambda art: art.startswith("model"), log_artifact.get_artifact_paths())))
            self.assertGreater(float(fake_out.getvalue().split("\n")[-1]), 0)

if __name__ == "__main__":
    unittest.main()
