"""
feature_engineering.py
------------------------
Adauga caracteristici suplimentare (feature engineering) la setul de date
deja curatat, pentru a ajuta modelul de regresie sa prezica mai precis
pretul unei masini.

Poate fi folosit fie ca modul (import din alt script), fie rulat direct:
    python3 feature_engineering.py
va citi data/cars_clean.csv, va adauga caracteristicile si va salva
rezultatul in data/cars_features.csv.
"""

import pandas as pd

CLEAN_DATA_PATH = "data/cars_clean.csv"
FEATURES_DATA_PATH = "data/cars_features.csv"

# Anul curent, folosit ca referinta pentru calculul varstei masinii.
# Il fixam ca o constanta (nu datetime.now()) ca rezultatele sa fie
# reproductibile - oricine ruleaza scriptul obtine aceleasi valori.
CURRENT_YEAR = 2024


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # car_age - varsta masinii. Are sens intuitiv: masinile mai vechi
    # valoreaza, in general, mai putin. E o transformare mai directa a
    # coloanei "year", mai usor de interpretat de unele modele (ex:
    # Linear Regression) decat anul absolut.
    df["car_age"] = CURRENT_YEAR - df["year"]
    # O masina cu an de fabricatie in viitor (eroare de date) ar da o
    # varsta negativa - o corectam la 0.
    df["car_age"] = df["car_age"].clip(lower=0)

    # mileage_per_year - kilometrajul mediu pe an de utilizare. Distinge
    # o masina de 10 ani cu kilometraj mic (folosita rar) de una cu
    # kilometraj foarte mare (folosita intens) - ambele scenarii
    # afecteaza diferit pretul de piata.
    # Adaugam +1 la varsta ca sa evitam impartirea la zero pentru
    # masinile din anul curent (car_age = 0).
    df["mileage_per_year"] = df["mileage_km"] / (df["car_age"] + 1)

    # engine_volume_liters - capacitatea motorului in litri, in loc de
    # cm3. E aceeasi informatie, dar litri este formatul in care oamenii
    # descriu de obicei motoarele (ex: "motor de 1.6" in loc de "1600 cm3").
    df["engine_volume_liters"] = df["volume_cm3"] / 1000

    # is_newer_car - indicator simplu (0/1) daca masina are cel mult 5 ani.
    # Ofera modelului un "prag" clar in plus fata de varsta continua -
    # utile mai ales pentru modele bazate pe arbori de decizie.
    df["is_newer_car"] = (df["car_age"] <= 5).astype(int)

    # is_high_mileage - indicator daca masina are un kilometraj neobisnuit
    # de mare (peste 200.000 km, un prag comun folosit informal pe piata
    # auto second-hand ca semnal de uzura ridicata).
    df["is_high_mileage"] = (df["mileage_km"] > 200_000).astype(int)

    # brand_model - combinatie marca+model. Unele combinatii specifice
    # (ex: "bmw 3 series" vs "bmw x5") au valori de piata foarte diferite,
    # chiar daca "make" (bmw) e acelasi. Aceasta caracteristica permite
    # modelului sa invete tipare specifice per combinatie, nu doar per marca.
    df["brand_model"] = df["make"].astype(str) + "_" + df["model"].astype(str)

    return df


if __name__ == "__main__":
    print(f"Incarc datele curatate din {CLEAN_DATA_PATH} ...")
    df = pd.read_csv(CLEAN_DATA_PATH)

    df = add_features(df)
    print("Caracteristici noi adaugate:")
    print([c for c in df.columns if c in [
        "car_age", "mileage_per_year", "engine_volume_liters",
        "is_newer_car", "is_high_mileage", "brand_model",
    ]])

    df.to_csv(FEATURES_DATA_PATH, index=False)
    print(f"Date cu caracteristici noi salvate in {FEATURES_DATA_PATH}")
