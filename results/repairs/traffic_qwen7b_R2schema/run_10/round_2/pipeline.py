import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data."""
    # Drop any rows with missing values
    data = data.dropna()
    
    # Separate features and target
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']
    
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier."""
    with mlflow.start_run():
        # Set experiment name
        mlflow.set_experiment("RandomForestCongestionDetector")
        
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log metrics
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        
        # Infer signature
        signature = infer_signature(X_train, y_pred)
        
        # Save model with signature
        mlflow.sklearn.save_model(model, "model", signature=signature)
        
        return model

def load_model():
    """Load the trained model."""
    client = MlflowClient()
    model_uri = "models:/RandomForestCongestionDetector/1"
    model = mlflow.sklearn.load_model(model_uri)
    return model

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
