def test_preprocess_data():
    data = load_data('data/air_quality.csv')
    print(data.columns)  # Debug print to check the actual columns
    X, y = preprocess_data(data)
    assert X.shape[1] == 15  # 14 features + 1 target
    assert y.shape[0] == data.shape[0]
