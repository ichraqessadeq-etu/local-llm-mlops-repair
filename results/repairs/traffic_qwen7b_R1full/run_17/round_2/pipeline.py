import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess data for model training."""
    # Separate features and target
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']
    
    # Identify numeric and categorical columns
    numeric_features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'Hour', 'DayOfWeek', 'Month']
    categorical_features = ['weather_main', 'IsHoliday']
    
    # Define transformers for numeric and categorical columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Apply preprocessing
    X = preprocessor.fit_transform(X)
    
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    # Set up MLflow experiment
    mlflow.set_experiment("traffic-congestion-pipeline")
    
    # Define the pipeline
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='mean'), ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'Hour', 'DayOfWeek', 'Month']),
                ('cat', SimpleImputer(strategy='most_frequent'), ['weather_main', 'IsHoliday']),
                ('onehot', OneHotEncoder(handle_unknown='ignore'), ['weather_main', 'IsHoliday'])
            ]
        )),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        mlflow.sklearn.log_model(pipeline, "model")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("mse", mean_squared_error(y_train, pipeline.predict(X_train)))
    
    return pipeline
