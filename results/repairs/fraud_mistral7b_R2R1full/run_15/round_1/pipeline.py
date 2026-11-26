import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from sklearn.utils.multiclass import type_of_target

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data, data['Amount'].astype(int))

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, 'random_forest')
    return model

def type_of_target(y):
    if y.dtype == np.int8 or y.dtype == np.int16 or y.dtype == np.int32 or y.dtype == np.int64:
        return "multiclass"
    elif y.dtype == np.uint8 or y.dtype == np.uint16 or y.dtype == np.uint32 or y.dtype == np.uint64:
        return "multiclass-multioutput"
    elif y.dtype == np.bool:
        return "binary"
    elif y.dtype == np.float16 or y.dtype == np.float32 or y.dtype == np.float64:
        return "continuous"
    else:
        raise ValueError("Unknown label type: %r" % y.dtype)
