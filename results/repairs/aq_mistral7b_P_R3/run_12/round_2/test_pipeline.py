import mlflow
from sklearn.model_selection import train_test_split
import unittest
from pipeline import load_data, preprocess_data, train_model, save_model
from sklearn.metrics import accuracy_score

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data('data/air_quality.csv')
        cls.X, cls.y = preprocess_data(cls.data)

    def test_train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)
        assert accuracy_score(y_test, y_pred) > 0.8

    def test_save_model(self):
        with pyfunc.local_context() as mlflow_run:
            mlflow.start_run(run_name="anomaly_detector")
            model = train_model(self.X, self.y)
            save_model(model, mlflow_run.info.run_id)
            artifact_path = mlflow.get_artifact_uri("models/{}".format(mlflow_run.info.run_id))
            assert artifact_path is not None
