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
    # Convert categorical variables to numerical
    data['weather_main'] = data['weather_main'].map({'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Snow': 3})
    data['IsHoliday'] = data['IsHoliday'].map({True: 1, False: 0})
    
    # Split features and target
    X = data.drop('Congestion', axis=1)
    y = data['Congestion']
    
    return X, y

def train_model(X, y):
    """Train the RandomForest model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run():
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        
        # Log the model
        mlflow.sklearn.log_model(rf, 'rf_model')
        
        # Evaluate the model
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)
        
        return rf
