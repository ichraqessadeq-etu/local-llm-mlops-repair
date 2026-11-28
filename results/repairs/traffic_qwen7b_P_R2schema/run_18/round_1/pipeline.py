import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data for model training."""
    # Separate features and target
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']
    
    # Categorical features
    categorical_features = ['weather_main', 'IsHoliday']
    
    # Numerical features
    numerical_features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'Hour', 'DayOfWeek', 'Month']
    
    # Create transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Apply preprocessing
    X_preprocessed = preprocessor.fit_transform(X)
    
    return X_preprocessed, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier and log it using MLflow."""
    with mlflow.start_run():
        # Define the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        # Log the parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        
        # Log the metrics
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        
        # Infer the signature
        signature = infer_signature(X_train, y_pred)
        
        # Save the signature
        mlflow.sklearn.save_model(model, "model", signature=signature)
        
        return model
