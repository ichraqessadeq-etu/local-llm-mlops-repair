# test_pipeline.py
from pipeline import load_data, preprocess_data, train_model
import pytest
import pandas

features = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'NOx(GT)', 'C6H6(GT)', 'T', 'RH', 'AH', 'Hour', 'DayOfWeek', 'Month']

def test_train_model():
    data = pandas.read_csv("data/air_quality.csv")
    X, y = preprocess_data(data)
    model = train_model(X, y)
    # Add assertions to check if the model is trained correctly

if __name__ == "__main__":
    test_train_model()
