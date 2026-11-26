import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
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
    """Preprocess the data and split into features and target."""
    try:
        # Separate features and target
        X = data.drop(columns=['Congestion'])
        y = data['Congestion']
        
        # Encode categorical features
        categorical_features = ['weather_main', 'IsHoliday', 'DayOfWeek', 'Month']
        encoder = OneHotEncoder(sparse=False)
        encoded_features = encoder.fit_transform(X[categorical_features])
        encoded_feature_names = encoder.get_feature_names_out(categorical_features)
        
        # Drop categorical features and concatenate encoded features
        X = X.drop(columns=categorical_features)
        X_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names)
        X = pd.concat([X, X_encoded], axis=1)
        
        return X, y
    except Exception as e:
        raise ValueError(f"Failed to preprocess data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForest model and log it using MLflow."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(random_state=42)  # Add random_state for reproducibility
            model.fit(X_train, y_train)
            mlflow.log_params(model.get_params())
            mlflow.sklearn.log_model(model, "model")
            return model
    except Exception as e:
        raise ValueError(f"Failed to train model: {e}")

if __name__ == "__main__":
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
