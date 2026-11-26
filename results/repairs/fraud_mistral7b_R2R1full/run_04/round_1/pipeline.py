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
        mlflow.sklearn.log_model(model, 'random-forest')
    return model

def type_of_target(y):
    if y.dtype == 'int8':
        return "multiclass"
    elif y.dtype == 'int16':
        return "multiclass"
    elif y.dtype == 'int32':
        return "multiclass"
    elif y.dtype == 'int64':
        return "multiclass"
    elif y.dtype == 'uint8':
        return "multiclass"
    elif y.dtype == 'uint16':
        return "multiclass"
    elif y.dtype == 'uint32':
        return "multiclass"
    elif y.dtype == 'uint64':
        return "multiclass"
    elif y.dtype == 'float16':
        return "continuous"
    elif y.dtype == 'float32':
        return "continuous"
    elif y.dtype == 'float64':
        return "continuous"
    else:
        raise ValueError("Unknown label type: %r" % y.dtype)
