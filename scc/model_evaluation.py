"""
model_evaluation.py
---------------------
Incarca modelul deja antrenat si salvat (models/car_price_model.joblib)
si evalueaza performanta lui pe setul de testare, folosind metrici de
regresie: MAE, MSE, RMSE, R2.

Afiseaza si un tabel cu exemple concrete (pret real vs. pret estimat vs.
eroare), util pentru a intelege intuitiv cat de bine functioneaza modelul.

IMPORTANT: refolosim exact acelasi random_state (42) la train_test_split
ca in model_training.py / model_comparison.py, ca sa obtinem exact
acelasi set de testare - altfel am evalua modelul pe date pe care le-a
vazut deja la antrenare, ceea ce ar da rezultate fals optimiste.

Ruleaza din radacina proiectului:
    python3 src/model_evaluation.py
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data_preprocessing import prepare_features_and_target

FEATURES_DATA_PATH = "data/cars_features.csv"
MODEL_PATH = "models/car_price_model.joblib"
RANDOM_STATE = 42


def main():
    print(f"Incarc modelul din {MODEL_PATH} ...")
    pipeline = joblib.load(MODEL_PATH)

    df = pd.read_csv(FEATURES_DATA_PATH)
    X, y = prepare_features_and_target(df)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Metrici de evaluare ---")
    print(f"MAE  = {mae:.2f} USD")
    print(f"MSE  = {mse:.2f}")
    print(f"RMSE = {rmse:.2f} USD")
    print(f"R2   = {r2:.4f}")

    print("\nInterpretare:")
    print(f"- MAE {mae:.0f} USD inseamna ca, in medie, modelul greseste cu "
          f"aproximativ {mae:.0f} USD fata de pretul real al masinii.")
    print(f"- R2 {r2:.4f} inseamna ca modelul explica aproximativ "
          f"{r2 * 100:.1f}% din variatia preturilor din setul de testare.")

    print("\n--- Exemple: pret real vs. pret estimat ---")
    sample = pd.DataFrame({
        "pret real": y_test.values,
        "pret estimat": np.round(y_pred).astype(int),
    })
    sample["eroare"] = (sample["pret real"] - sample["pret estimat"]).abs()
    print(sample.sample(10, random_state=RANDOM_STATE).to_string(index=False))


if __name__ == "__main__":
    main()
