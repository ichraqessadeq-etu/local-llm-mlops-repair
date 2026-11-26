from numpy.testing import assert_array_equal

def test_train_model():
    data = load_data("data/fraud.csv")
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    train_model(X_train, y_train)
    predictions = train_model.get_predictions(X_test)
    assert_array_equal(y_test, predictions)
