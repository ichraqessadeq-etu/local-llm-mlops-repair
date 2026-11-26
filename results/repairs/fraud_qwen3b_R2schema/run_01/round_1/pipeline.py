import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise MlflowException(f"File {file_path} not found.")
    except pd.errors.EmptyDataError:
        raise MlflowException("The file is empty.")
    except pd.errors.ParserError:
        raise MlflowException("Error parsing the CSV file.")

def preprocess_data(data):
    try:
        # Define preprocessing steps
        num_features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
        cat_features = [col for col in data.columns if col not in num_features]

        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_features),
                ('cat', cat_transformer, cat_features)
            ])

        X = data[num_features]
        y = data['Class']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise MlflowException(f"Error preprocessing data: {e}")

def train_model(X_train, y_train):
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise MlflowException(f"Error training model: {e}")

def run_pipeline(file_path):
    try:
        data = load_data(file_path)
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        mlflow.sklearn.log_model(model, "random_forest_model")
        print("Model trained and logged successfully.")
    except MlflowException as e:
        print(f"An error occurred: {e}")
