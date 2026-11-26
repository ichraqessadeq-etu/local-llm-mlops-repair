import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Preprocess data to prepare it for modeling."""
    # Assuming 'Congestion' is the target column
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']
    
    # One-hot encode categorical columns
    encoder = OneHotEncoder(sparse=False)
    encoded_features = encoder.fit_transform(X[['weather_main']])
    encoded_feature_names = encoder.get_feature_names_out(['weather_main'])
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
    
    # Drop the original categorical column
    X = X.drop(columns=['weather_main'])
    
    # Concatenate the encoded features with the rest of the data
    X = pd.concat([X, encoded_df], axis=1)
    
    # Convert IsHoliday to numeric
    X['IsHoliday'] = X['IsHoliday'].astype(int)
    
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    with mlflow.start_run():
        mlflow.log_param("model", "RandomForestClassifier")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("accuracy", accuracy_score(y_train, model.predict(X_train)))
        return model
