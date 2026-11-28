# pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    - file_path: str, path to the CSV file.
    
    Returns:
    - df: pandas DataFrame containing the data.
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(data):
    """
    Preprocess the data by separating features and target.
    
    Parameters:
    - data: pandas DataFrame containing the data.
    
    Returns:
    - X: pandas DataFrame containing the features.
    - y: pandas Series containing the target.
    """
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']
    return X, y

def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier model.
    
    Parameters:
    - X_train: pandas DataFrame containing the training features.
    - y_train: pandas Series containing the training target.
    
    Returns:
    - model: trained RandomForestClassifier model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    
    Parameters:
    - model: trained model.
    - X_test: pandas DataFrame containing the test features.
    - y_test: pandas Series containing the test target.
    
    Returns:
    - accuracy: float, accuracy score of the model on the test set.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# --- R2schema shim: data schema known to the operator (added by the deterministic corrector, not by the model) ---
import pandas as _r2s_pd
_R2S_CATEGORIES = {'weather_main': ['Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 'Rain', 'Smoke', 'Snow', 'Squall', 'Thunderstorm']}


def _r2s_encode(X, full=True):
    if not isinstance(X, _r2s_pd.DataFrame):
        return X
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == object and c in _R2S_CATEGORIES:
            X[c] = X[c].map({v: i for i, v in enumerate(_R2S_CATEGORIES[c])}).fillna(-1)
        elif X[c].dtype == object and full:
            X[c] = _r2s_pd.to_numeric(X[c], errors="coerce")
    if full:
        num = X.select_dtypes(include="number").columns
        X[num] = X[num].fillna(X[num].median()).fillna(0)
    return X


class _R2SEncoded(object):
    def __init__(self, obj, name):
        self.obj, self.name = obj, name

    def __call__(self, X, *a, **k):
        return getattr(type(self.obj), self.name)(self.obj, _r2s_encode(X), *a, **k)


if "preprocess_data" in globals():
    _r2s_preprocess = preprocess_data

    def preprocess_data(data, *a, **k):
        return _r2s_preprocess(_r2s_encode(data, full=False), *a, **k)

if "train_model" in globals():
    _r2s_train = train_model

    def train_model(X_train, y_train, *a, **k):
        m = _r2s_train(_r2s_encode(X_train), y_train, *a, **k)
        for o in (m if isinstance(m, (tuple, list)) else [m]):
            if hasattr(o, "fit"):
                for n in ("predict", "predict_proba", "decision_function", "score"):
                    if hasattr(type(o), n):
                        try:
                            setattr(o, n, _R2SEncoded(o, n))
                        except Exception:
                            pass
        return m
