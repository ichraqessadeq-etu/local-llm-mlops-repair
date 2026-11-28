import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, track, pyfunc, run_local

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data.drop('Amount', axis=1), data['Amount'])

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    mlflow_model = pyfunc.PyFuncTrackingEstimator(module=__name__, init_code='\n            import pandas as pd\n            import mlflow.pyfunc\n            import numpy as np\n            def init(model):\n                return model\n            def predict(model, X):\n                return model.predict(X)\n        ', source_code='\n            def predict(X):\n                return model.predict(X)\n        ', init=init_code, source_code=source_code, model=model)
    mlflow.log_artifact(artifact=mlflow_model, artifact_path='models')
    track(mlflow_model.model_name)
    return mlflow_model