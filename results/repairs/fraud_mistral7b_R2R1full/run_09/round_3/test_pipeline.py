from sklearn.model_selection import train_test_split
import pytest
from pipeline import load_data, preprocess_data, train_model, type_of_target

def test_load_data():
    data = load_data("data/fraud.csv")
    assert len(data) > 0

def test_preprocess_data():
    data = load_data("data/fraud.csv")
    X, y = preprocess_data(data)
    assert X.shape[0] == y.shape[0]

def test_type_of_target():
    y = pd.Series([0, 1, 2])
    assert type_of_target(y) == "continuous"

    y = pd.Series([0, 1, 1])
    assert type_of_target(y) == "multiclass"

    y = pd.Series([0, 1, 1, 0])
    assert type_of_target(y) == "multiclass"

    y = pd.Series([0, 1, 1, 0, 1])
    assert type_of_target(y) == "multiclass"

    y = pd.Series([0, 1, 1, 0, 1, 0])
    assert type_of_target(y) == "multiclass"

    y = pd.Series([1, 0, 1, 0, 1, 0])
    assert type_of_target(y) == "multiclass"

    y = pd.Series([1, 1, 0, 1, 1, 0])
    assert type_of_target(y) == "multiclass"

    y = pd.Series([1, 1, 1, 0, 1, 0])
    assert type_of_target(y) == "multiclass"

    y = pd.Series([1, 1, 1, 1, 1, 0])
    assert type_of_target(y) == "multiclass"

    y = pd.Series([1, 1, 1, 1, 1, 1])
    assert type_of_target(y) == "multiclass"

    y = pd.Series([0, 0, 0, 1, 1, 1])
    assert type_of_target(y) == "multiclass"

    y = pd.Series([0, 0, 1, 1, 1, 1])
    assert type_of_target(y) == "multiclass"

    y = pd.Series([0, 1, 1, 1, 1, 1])
    assert type_of_target(y) == "multiclass"

    y = pd.Series([1, 0, 1, 0, 1, 0])
    assert type_of_target(y) == "multiclass"

    y = pd.Series([1, 1, 0, 0, 1, 0])
    assert type_of_target(y) == "multiclass"

    y = pd.Series([1, 1, 1, 0, 0, 0])
    assert type_of_target(y) == "multiclass"

    y = pd.Series([1, 1, 1, 1, 0, 0])
    assert type_of_target(y) == "multiclass"

    y = pd.Series([1, 1, 1, 1, 1, 1])
    assert type_of_target(y) == "multiclass"

    y = pd.Series([0, 0, 0, 0, 0, 0])
    assert type_of_target(y) == "multiclass"

def test_train_model():
    data = load_data("data/fraud.csv")
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    # Add assertions to check the model's performance on the test set
