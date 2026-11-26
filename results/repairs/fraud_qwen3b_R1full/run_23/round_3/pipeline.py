import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import mlflow
import mlflow.sklearn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error("File not found.")
        raise
    except pd.errors.EmptyDataError:
        logging.error("Empty data file.")
        raise

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Separate features and target
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        # Define preprocessing steps
        numerical_features = X.select_dtypes(include=['number']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Apply preprocessing
        X_preprocessed = preprocessor.fit_transform(X)
        
        # Convert NumPy array to DataFrame
        X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=numerical_features + list(categorical_transformer.named_steps['onehot'].get_feature_names_out()))
        
        logging.info("Data preprocessed successfully.")
        return X_preprocessed_df, y
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        logging.info("Model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def main():
    file_path = 'data/fraud.csv'
    data = load_data(file_path)
    X_preprocessed, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    
    # Log model to MLflow
    mlflow.set_experiment("fraud_detection")
    with mlflow.start_run():
        mlflow.log_params({
            "n_estimators": 100,
            "random_state": 42
        })
        mlflow.sklearn.log_model(model, "random_forest_model")
        logging.info("Model logged to MLflow successfully.")

if __name__ == "__main__":
    main()
