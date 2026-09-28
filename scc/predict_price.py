"""
predict_price.py
------------------
Script optional de testare interactiva: introdu caracteristicile unei
masini din consola si primesti pretul estimat de model.

Nu este o cerinta explicita a task-ului, dar e util pentru a verifica
rapid modelul cu exemple concrete (ca cel din enuntul temei: Volkswagen
Golf 2014, diesel, 180000 km).

IMPORTANT: trebuie sa rulezi mai intai model_comparison.py (sau macar
model_training.py) o data, ca sa existe fisierul models/car_price_model.joblib.

Ruleaza din radacina proiectului:
    python3 src/predict_price.py
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

import joblib
import pandas as pd

from feature_engineering import add_features
from data_preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    group_rare_categories,
)

MODEL_PATH = "models/car_price_model.joblib"

# Valorile posibile pentru fiecare camp categoric, asa cum apar in setul
# de date original - le afisam utilizatorului ca ghid, ca sa introduca
# valori pe care modelul le recunoaste.
VALID_VALUES = {
    "condition": ["with mileage", "with damage", "for parts"],
    "fuel_type": ["petrol", "diesel", "electrocar"],
    "transmission": ["mechanics", "auto"],
    "drive_unit": [
        "front-wheel drive", "rear drive", "all-wheel drive",
        "part-time four-wheel drive", "unknown",
    ],
    "segment": ["A", "B", "C", "D", "E", "F", "J", "M", "S", "unknown"],
}


def ask(prompt, cast=str):
    value = input(prompt).strip()
    return cast(value)


def build_example_row() -> pd.DataFrame:
    print("Introdu caracteristicile masinii:\n")

    make = ask("Marca (ex: volkswagen): ").lower()
    model_name = ask("Modelul (ex: golf): ").lower()
    year = ask("Anul fabricatiei (ex: 2014): ", int)

    print(f"Stare posibila: {VALID_VALUES['condition']}")
    condition = ask("Starea masinii: ").lower()

    mileage_km = ask("Kilometrajul (in km, ex: 180000): ", float)

    print(f"Combustibil posibil: {VALID_VALUES['fuel_type']}")
    fuel_type = ask("Tipul de combustibil: ").lower()

    volume_cm3 = ask("Capacitatea motorului (in cm3, ex: 1600; scrie 0 pentru masini electrice): ", float)

    color = ask("Culoarea (ex: black): ").lower()

    print(f"Transmisie posibila: {VALID_VALUES['transmission']}")
    transmission = ask("Tipul transmisiei: ").lower()

    print(f"Tractiune posibila: {VALID_VALUES['drive_unit']}")
    drive_unit = ask("Tipul tractiunii: ").lower()

    print(f"Segment posibil: {VALID_VALUES['segment']}")
    segment = ask("Segmentul masinii: ").upper()

    row = pd.DataFrame([{
        "make": make,
        "model": model_name,
        "year": year,
        "condition": condition,
        "mileage_km": mileage_km,
        "fuel_type": fuel_type,
        "volume_cm3": volume_cm3,
        "color": color,
        "transmission": transmission,
        "drive_unit": drive_unit,
        "segment": segment,
    }])

    return row


def predict(pipeline, raw_row: pd.DataFrame) -> float:
    row = add_features(raw_row)
    row["brand_model_grouped"] = group_rare_categories(row["brand_model"], top_n=40)
    X = row[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return pipeline.predict(X)[0]


if __name__ == "__main__":
    print(f"Incarc modelul din {MODEL_PATH} ...")
    model_pipeline = joblib.load(MODEL_PATH)
    print("Model incarcat cu succes.\n")

    while True:
        example_row = build_example_row()
        estimated_price = predict(model_pipeline, example_row)
        print(f"\n>> Pret estimat: {estimated_price:.0f} USD\n")

        again = input("Vrei sa testezi alta masina? (da/nu): ").strip().lower()
        if again != "da":
            print("La revedere!")
            break
