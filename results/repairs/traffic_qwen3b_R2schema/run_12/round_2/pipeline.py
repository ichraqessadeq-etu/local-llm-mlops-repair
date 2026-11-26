import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logger.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logger.error("File not found.")
        raise
    except pd.errors.EmptyDataError:
        logger.error("Empty file.")
        raise
    except pd.errors.ParserError:
        logger.error("Error parsing CSV file.")
        raise

def preprocess_data(data):
    """Preprocess the data by splitting features and target."""
    try:
        features = data.drop('Congestion', axis=1)
        target = data['Congestion']
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        logger.info("Data preprocessed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logger.info("Model trained successfully.")
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy:.2f}")
        return accuracy
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

def run_pipeline(file_path):
    """Run the entire pipeline."""
    try:
        data = load_data(file_path)
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        return accuracy
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        raise

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
