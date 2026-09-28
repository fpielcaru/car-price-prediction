"""
data_preprocessing.py
-----------------------
Defineste variabila tinta, coloanele numerice/categorice si pipeline-ul
de preprocesare (transformarea datelor brute in numere pe care un model
de regresie le poate folosi).

Acest fisier NU antreneaza niciun model - doar pregateste "reteta" de
transformare a datelor, refolosita identic in model_training.py,
model_evaluation.py si model_comparison.py.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "priceUSD"

# Coloane numerice folosite ca intrare pentru model.
# Nota: NU includem si "year" si "volume_cm3" separat, pentru ca sunt
# aproape complet redundante cu "car_age" si "engine_volume_liters"
# (aceeasi informatie, doar transformata) - a le include pe amandoua
# nu ar aduce informatie noua, doar ar creste inutil dimensiunea datelor.
NUMERIC_FEATURES = [
    "car_age",
    "mileage_km",
    "mileage_per_year",
    "engine_volume_liters",
    "is_newer_car",
    "is_high_mileage",
]

# Coloane categorice folosite ca intrare pentru model.
CATEGORICAL_FEATURES = [
    "make",
    "condition",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit",
    "segment",
    "brand_model_grouped",
]

# Cate variante distincte de "brand_model" pastram ca fiind categorii
# proprii - restul sunt grupate sub eticheta "other". Coloana "model"
# are 1034 valori distincte in setul original; fara aceasta grupare,
# codificarea one-hot ar crea 1000+ coloane noi, majoritatea foarte rare
# (aparand la doar 1-2 masini), ceea ce ar face modelul mai lent si mai
# predispus la overfitting pe combinatii rare marca+model.
TOP_N_BRAND_MODEL = 40


def group_rare_categories(series: pd.Series, top_n: int, other_label: str = "other") -> pd.Series:
    """Pastreaza doar cele mai frecvente `top_n` valori dintr-o coloana
    categorica; restul valorilor sunt inlocuite cu `other_label`."""
    top_categories = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_categories), other_label)


def prepare_features_and_target(df: pd.DataFrame):
    """Primeste DataFrame-ul complet (dupa curatare + inginerie
    caracteristici) si returneaza X (caracteristicile de intrare) si
    y (variabila tinta, priceUSD)."""
    df = df.copy()

    df["brand_model_grouped"] = group_rare_categories(
        df["brand_model"], top_n=TOP_N_BRAND_MODEL
    )

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Construieste pipeline-ul de preprocesare:
    - coloanele numerice: completare valori lipsa cu mediana, apoi scalare
      (StandardScaler) - scalarea ajuta modele precum Linear Regression
      sa trateze corect coloane cu scari foarte diferite (ex: mileage_km
      in sute de mii, vs is_newer_car care e 0 sau 1).
    - coloanele categorice: completare valori lipsa cu cea mai frecventa
      valoare, apoi codificare one-hot (transforma fiecare categorie
      intr-o coloana binara 0/1). handle_unknown="ignore" asigura ca
      modelul nu crapa daca intalneste, la predictie, o categorie
      nevazuta la antrenare (ex: o marca noua)."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return preprocessor


if __name__ == "__main__":
    df = pd.read_csv("data/cars_features.csv")
    X, y = prepare_features_and_target(df)

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    print(f"Forma X (inainte de preprocesare): {X.shape}")
    print(f"Forma X (dupa preprocesare - matrice numerica): {X_transformed.shape}")
    print(f"Numar de categorii distincte pastrate pentru brand_model_grouped: "
          f"{X['brand_model_grouped'].nunique()}")
