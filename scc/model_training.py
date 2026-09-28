"""
model_training.py
-------------------
Antreneaza un model de regresie de BAZA (Linear Regression) pentru
predictia pretului unei masini, folosind pipeline-ul de preprocesare
definit in data_preprocessing.py.

Acesta e un punct de plecare simplu - modelul castigator final este
ales si salvat de model_comparison.py, dupa ce se compara mai multi
algoritmi. Acest script serveste ca prim test rapid: "merge tot fluxul
de la un capat la altul?".

Ruleaza din radacina proiectului:
    python3 src/model_training.py
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import build_preprocessor, prepare_features_and_target

FEATURES_DATA_PATH = "data/cars_features.csv"
MODEL_OUTPUT_PATH = "models/car_price_model.joblib"
RANDOM_STATE = 42


def main():
    print(f"Incarc datele din {FEATURES_DATA_PATH} ...")
    df = pd.read_csv(FEATURES_DATA_PATH)

    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Masini de antrenament: {len(X_train)}")
    print(f"Masini de testare: {len(X_test)}")

    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("regressor", LinearRegression()),
    ])

    print("\nAntrenez modelul de baza (Linear Regression) ...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nRezultate model de baza (Linear Regression):")
    print(f"MAE  = {mae:.2f} USD")
    print(f"MSE  = {mse:.2f}")
    print(f"RMSE = {rmse:.2f} USD")
    print(f"R2   = {r2:.4f}")
    print(f"\nInterpretare: in medie, modelul greseste cu aproximativ "
          f"{mae:.0f} USD fata de pretul real al masinii.")

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"\nModel de baza salvat in {MODEL_OUTPUT_PATH}")
    print("(Acest model va fi inlocuit de model_comparison.py daca un alt "
          "algoritm are performanta mai buna.)")


if __name__ == "__main__":
    main()
