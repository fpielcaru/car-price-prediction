"""
model_comparison.py
---------------------
Antreneaza si compara mai multi algoritmi de regresie, folosind exact
acelasi set de antrenament/testare pentru toti (comparatie corecta).

Modele comparate:
- Linear Regression       (simplu, rapid, usor de interpretat)
- Decision Tree Regressor (captureaza relatii ne-liniare)
- Random Forest Regressor (ansamblu de arbori - de obicei robust)
- Gradient Boosting Regressor (ansamblu secvential - adesea foarte precis)

La final, alege modelul cu cel mai mic RMSE si il salveaza in
models/car_price_model.joblib, inlocuind modelul de baza din
model_training.py daca acesta e mai performant.

Ruleaza din radacina proiectului:
    python3 src/model_comparison.py
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

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

    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=60, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    results = []
    trained_pipelines = {}

    for name, regressor in candidate_models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ])

        print(f"\nAntrenez: {name} ...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2,
        })
        trained_pipelines[name] = pipeline

        print(f"  MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}")

    results_df = pd.DataFrame(results).sort_values(by="RMSE")
    print("\n" + "=" * 70)
    print("Tabel comparativ (sortat dupa RMSE, cel mai mic = cel mai bun):")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]

    print(f"\nCel mai bun model: {best_model_name}")
    print(f"RMSE = {results_df.iloc[0]['RMSE']:.2f} USD, "
          f"MAE = {results_df.iloc[0]['MAE']:.2f} USD, "
          f"R2 = {results_df.iloc[0]['R2']:.4f}")

    joblib.dump(best_pipeline, MODEL_OUTPUT_PATH)
    print(f"\nModel final ({best_model_name}) salvat in {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
