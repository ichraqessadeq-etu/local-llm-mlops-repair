def test_preprocess_data():
    data = load_data('data/air_quality.csv')
    print(f"Data shape: {data.shape}")
    X, y = preprocess_data(data)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    assert X.shape[1] == 14  # 14 features + 1 target
    assert y.shape[0] == data.shape[0]
