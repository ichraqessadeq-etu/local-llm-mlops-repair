import pandas as pd
from sklearn.model_selection import train_test_split  # Added import for train_test_split
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test the load_data function."""
    data = load_data('data/fraud.csv')
    assert isinstance(data, pd.DataFrame)
    assert 'Class' in data.columns

def test_preprocess_data():
    """Test the preprocess_data function."""
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert 'Class' not in X.columns

def test_train_model():
    """Test the train_model function."""
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    y_pred = model.predict(X_test)
    assert isinstance(y_pred, np.ndarray)
