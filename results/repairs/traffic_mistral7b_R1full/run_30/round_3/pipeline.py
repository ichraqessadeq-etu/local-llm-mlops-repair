def train_model(X_train, y_train):
    if 'Clouds_all' not in X_train.columns:
        raise KeyError('Clouds_all column not found in the data')

    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)  # Seed the random number generator
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random-forest")

    return model
