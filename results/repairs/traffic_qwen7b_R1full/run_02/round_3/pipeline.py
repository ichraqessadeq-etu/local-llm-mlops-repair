import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from mlflow import log_param, log_metric

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess data to split into features and target."""
    # Separate features and target
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']
    
    # One-hot encode categorical features
    categorical_features = ['weather_main', 'IsHoliday']
    encoder = OneHotEncoder(sparse=False)
    encoded_features = encoder.fit_transform(X[categorical_features])
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    
    # Drop categorical features and concatenate encoded features
    X = X.drop(columns=categorical_features)
    X_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names)
    X = pd.concat([X, X_encoded], axis=1)
    
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Log model parameters and metrics
    log_param("n_estimators", model.n_estimators)
    log_param("random_state", model.random_state)
    
    y_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    log_metric("train_accuracy", accuracy)
    
    # Save the model to a file
    model_path = "random_forest_model.pkl"
    model.save(model_path)
    
    return model
