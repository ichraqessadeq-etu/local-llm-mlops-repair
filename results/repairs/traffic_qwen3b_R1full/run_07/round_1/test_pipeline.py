import pytest
import pandas as pd
from pipeline import load_data, preprocess_data, train_model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def test_load_data():
    # Create a mock DataFrame
    data = make_classification(n_samples=100, n_features=8, n_informative=4, n_redundant=4, random_state=42)
    df = pd.DataFrame(data, columns=['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month'])
    
    # Load data
    loaded_data = load_data(df)
    
    # Check if the loaded data is the same as the original DataFrame
    assert loaded_data.equals(df)

def test_preprocess_data():
    # Create a mock DataFrame
    data = make_classification(n_samples=100, n_features=8, n_informative=4, n_redundant=4, random_state=42)
    df = pd.DataFrame(data, columns=['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month'])
    
    # Preprocess data
    X_train, y_train = preprocess_data(df)
    
    # Check if the target column is binary
    assert y_train.dtype == 'int64'
    
    # Check if irrelevant columns are dropped
    assert 'weather_main' not in X_train.columns
    assert 'IsHoliday' not in X_train.columns
    
    # Check if the data is split correctly
    assert X_train.shape[0] + X_test.shape[0] == df.shape[0]

def test_train_model():
    # Create a mock DataFrame
    data = make_classification(n_samples=100, n_features=8, n_informative=4, n_redundant=4, random_state=42)
    df = pd.DataFrame(data, columns=['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month'])
    
    # Preprocess data
    X_train, y_train = preprocess_data(df)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Check if the model is not None
    assert model is not None
