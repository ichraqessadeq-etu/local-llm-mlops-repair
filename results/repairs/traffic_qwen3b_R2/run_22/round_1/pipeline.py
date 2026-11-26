import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Convert target column to binary
        data['Congestion'] = data['Congestion'].map({'Low': 0, 'Medium': 1, 'High': 2})
        
        # Drop unnecessary columns
        data = data.drop(['weather_main'], axis=1)
        
        # Encode categorical variables
        data = pd.get_dummies(data, columns=['IsHoliday', 'DayOfWeek', 'Month'])
        
        # Split features and target
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        return X, y
    except Exception as e:
        raise ValueError(f"Preprocessing error: {e}")

def train_model(X_train, y_train):
    """Train the RandomForest model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise ValueError(f"Model training error: {e}")

def run_pipeline(file_path):
    """Run the entire pipeline."""
    try:
        data = load_data(file_path)
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = train_model(X_train, y_train)
        
        # Log the model to MLflow
        mlflow.set_experiment("traffic-congestion")
        with mlflow.start_run():
            mlflow.log_params({
                "n_estimators": 100,
                "random_state": 42
            })
            mlflow.sklearn.log_model(model, "rf-congestion-model")
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            
            print(f"Model accuracy: {accuracy:.2f}")
    except Exception as e:
        print(f"Pipeline error: {e}")

if __name__ == "__main__":
    run_pipeline("data/traffic.csv")
