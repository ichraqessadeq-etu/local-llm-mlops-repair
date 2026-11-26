import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data."""
    # Drop rows with missing values
    data = data.dropna()
    
    # Convert categorical variables to numerical
    data['weather_main'] = data['weather_main'].astype('category').cat.codes
    
    # Split data into features and target
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']
    
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    with mlflow.start_run():
        # Set MLflow parameters
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("features", X_train.columns.tolist())
        
        # Train the model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        # Log metrics
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        
        return model
