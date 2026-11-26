import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from the given file path."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Preprocess the data to split it into features (X) and target (y)."""
    # Separate features and target
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']
    
    # One-hot encode categorical variables
    categorical_features = ['weather_main', 'IsHoliday']
    encoder = OneHotEncoder(sparse=False)
    encoded_features = encoder.fit_transform(X[categorical_features])
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    
    # Drop the original categorical columns
    X = X.drop(columns=categorical_features)
    
    # Concatenate the encoded features with the remaining features
    X = pd.concat([X, pd.DataFrame(encoded_features, columns=encoded_feature_names)], axis=1)
    
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    with mlflow.start_run():
        mlflow.log_param("model", "RandomForestClassifier")
        model = RandomForestClassifier(random_state=42)  # Add random_state for reproducibility
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("accuracy", accuracy_score(y_train, model.predict(X_train)))
        return model

def main():
    """Main function to run the pipeline."""
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    print(f"Model trained with accuracy: {accuracy_score(y_train, model.predict(X_train))}")

if __name__ == "__main__":
    main()
