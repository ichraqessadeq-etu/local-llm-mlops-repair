def test_preprocess_data():
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    assert 'DayOfWeek' not in X.columns
    assert 'Month' not in X.columns
    assert X is not None
    assert y is not None
