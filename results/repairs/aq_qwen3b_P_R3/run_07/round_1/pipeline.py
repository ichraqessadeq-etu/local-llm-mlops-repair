import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError("File not found.")
    except pd.errors.EmptyDataError:
        raise ValueError("Empty data in the file.")

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Select features and target
        X = data[['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'NOx(GT)', 'C6H6(GT)', 'T', 'RH', 'AH', 'Hour', 'DayOfWeek', 'Month']]
        y = data['Anomaly']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except KeyError:
        raise ValueError("Missing column in the data.")

def train_model(X_train, y_train):
    """Train a RandomForest classifier."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise Exception(f"An error occurred during model training: {e}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    except Exception as e:
        raise Exception(f"An error occurred during model evaluation: {e}")

def main(file_path):
    """Main function to load, preprocess, train, and evaluate the model."""
    try:
        data = load_data(file_path)
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        print(f"Model accuracy: {accuracy}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main("data/air_quality.csv")
