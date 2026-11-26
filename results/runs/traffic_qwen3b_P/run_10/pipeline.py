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
    # Convert 'IsHoliday' to binary
    data['IsHoliday'] = data['IsHoliday'].astype(int)
    
    # Encode categorical variables
    data = pd.get_dummies(data, columns=['weather_main', 'DayOfWeek', 'Month'])
    
    # Split features and target
    X = data.drop('Congestion', axis=1)
    y = data['Congestion']
    
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def run_pipeline(file_path):
    """Run the entire pipeline."""
    try:
        # Load data
        data = load_data(file_path)
        
        # Preprocess data
        X, y = preprocess_data(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")
        
        print(f"Model trained with accuracy: {accuracy:.2f}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_pipeline("data/traffic.csv")
