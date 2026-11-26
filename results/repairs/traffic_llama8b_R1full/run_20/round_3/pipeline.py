def train_model(data):
    try:
        # Import RandomForestClassifier here
        from sklearn.ensemble import RandomForestClassifier
        X = data.drop(['Congestion', 'weather_main', 'DayOfWeek', 'Month'], axis=1)
        y = data['Congestion']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Log model with MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        mlflow.end_run()

        return model
    except Exception as e:
        print(f"Error training model: {e}")
        raise
