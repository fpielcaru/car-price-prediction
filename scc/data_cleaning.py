"""
data_cleaning.py
-----------------
Curata setul de date brut (cars.csv) si il pregateste pentru pasii urmatori
(inginerie caracteristici, preprocesare, antrenare).

Ce face acest script, pe scurt:
1. Redenumeste coloanele cu nume mai usor de folosit in cod
   (ex: "mileage(kilometers)" -> "mileage_km").
2. Trateaza valorile lipsa in mod specific pentru fiecare coloana.
3. Elimina randurile clar invalide (ani de fabricatie imposibili,
   kilometraje absurde, preturi extreme care distorsioneaza modelul).

Poate fi folosit fie ca modul (import din alt script), fie rulat direct:
    python3 data_cleaning.py
va citi data/cars.csv, va curata datele si va salva rezultatul in
data/cars_clean.csv.
"""

import pandas as pd

RAW_DATA_PATH = "data/cars.csv"
CLEAN_DATA_PATH = "data/cars_clean.csv"


def load_and_clean_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ------------------------------------------------------------------
    # 1. Redenumirea coloanelor
    # ------------------------------------------------------------------
    # Numele originale ("mileage(kilometers)", "volume(cm3)") contin
    # paranteze, ceea ce ingreuneaza scrierea codului (df.mileage(kilometers)
    # nu ar functiona ca atribut). Le redenumim in forme simple, fara
    # caractere speciale.
    df = df.rename(columns={
        "mileage(kilometers)": "mileage_km",
        "volume(cm3)": "volume_cm3",
    })

    # ------------------------------------------------------------------
    # 2. Tratarea valorilor lipsa
    # ------------------------------------------------------------------
    # volume_cm3 lipseste EXCLUSIV la masinile electrice (verificat in
    # explorarea datelor) - are sens, masinile electrice nu au un motor
    # cu ardere interna, deci nu au o "capacitate a motorului" in cm3.
    # Completam cu 0 in loc sa stergem randurile sau sa punem media
    # (media ar fi inselatoare pentru o masina electrica).
    df["volume_cm3"] = df["volume_cm3"].fillna(0)

    # drive_unit si segment au multe valori lipsa (introduse optional de
    # vanzatori). In loc sa stergem aceste randuri (am pierde ~10% din
    # date), le tratam ca o categorie proprie "unknown" - modelul poate
    # invata ca "nu stim tractiunea/segmentul" e si el un semnal valid.
    df["drive_unit"] = df["drive_unit"].fillna("unknown")
    df["segment"] = df["segment"].fillna("unknown")

    # ------------------------------------------------------------------
    # 3. Eliminarea randurilor invalide / valori extreme
    # ------------------------------------------------------------------
    # Ani de fabricatie nerealisti: sub 1950 sunt foarte rari (124 din
    # 56000+ randuri) si cresc riscul de date gresit introduse de
    # vanzatori (typo-uri). Pastram doar masini din 1950 incoace.
    df = df[df["year"] >= 1950]

    # Kilometraj absurd de mare (peste 1.000.000 km) - aproape sigur
    # erori de introducere (ex: cifre in plus adaugate din greseala).
    df = df[df["mileage_km"] <= 1_000_000]

    # Preturi extrem de mici (sub 100$) - probabil anunturi test/eronate,
    # nu preturi reale de vanzare pentru o masina.
    df = df[df["priceUSD"] >= 100]

    # Preturi extrem de mari (peste percentila 99.9) - cateva masini de
    # lux/colectie foarte scumpe pot distorsiona puternic modelul de
    # regresie (MSE e foarte sensibil la valori extreme). Le eliminam
    # ca outlieri, mentionand explicit acest lucru in documentatie.
    price_upper_limit = df["priceUSD"].quantile(0.999)
    df = df[df["priceUSD"] <= price_upper_limit]

    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    print(f"Incarc datele brute din {RAW_DATA_PATH} ...")
    raw_df = pd.read_csv(RAW_DATA_PATH)
    print(f"Randuri brute: {len(raw_df)}")

    clean_df = load_and_clean_data(RAW_DATA_PATH)
    print(f"Randuri dupa curatare: {len(clean_df)}")
    print(f"Randuri eliminate: {len(raw_df) - len(clean_df)}")

    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Date curatate salvate in {CLEAN_DATA_PATH}")
