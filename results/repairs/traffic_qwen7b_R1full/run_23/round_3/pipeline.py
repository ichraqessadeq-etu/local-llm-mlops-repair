import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise ValueError(f"Failed to load data: {e}")

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Separate features and target
        X = data.drop(columns=['Congestion'])
        y = data['Congestion']
        
        # Identify numeric and categorical columns
        numeric_features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'Hour', 'DayOfWeek', 'Month']
        categorical_features = ['weather_main', 'IsHoliday']
        
        # Create transformers for numeric and categorical features
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Apply preprocessing
        X_processed = preprocessor.fit_transform(X)
        
        return X_processed, y
    except Exception as e:
        raise ValueError(f"Failed to preprocess data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(random_state=42)  # Add random_state for reproducibility
            model.fit(X_train, y_train)
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            mlflow.sklearn.log_model(model, "model")
            y_pred = model.predict(X_train)
            accuracy = accuracy_score(y_train, y_pred)
            mlflow.log_metric("train_accuracy", accuracy)
            return model
    except Exception as e:
        raise ValueError(f"Failed to train model: {e}")
