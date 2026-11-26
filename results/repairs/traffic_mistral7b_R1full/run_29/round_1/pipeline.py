import pandas as pd
import mlflow
import random_forest_classifier as rfc

# ... (previously defined functions)

class RandomForestClassifier:
    @staticmethod
    def train_model(X_train, y_train, random_state=42):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=random_state)
        clf.fit(X_train, y_train)
        return clf
