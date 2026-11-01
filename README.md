Predicția prețului mașinilor second-hand
Proiect de regresie ML care estimează prețul de piață ( priceUSD) al unei mașini second-hand pe baza caracteristicilor ei (marcă, model, an, stare, kilometraj, combustibil, transmisie, tracțiune, segment), folosit un set de date real cu peste 56.00 de execuție.

Structura proiectului
car-price-prediction/
├── data/
│   ├── cars.csv                  # setul de date brut
│   ├── cars_clean.csv            # generat de data_cleaning.py
│   └── cars_features.csv         # generat de feature_engineering.py
├── notebooks/
│   └── 01_eda.ipynb              # analiza completă, pas cu pas
├── src/
│   ├── data_cleaning.py          # curățarea datelor
│   ├── feature_engineering.py    # caracteristici suplimentare
│   ├── data_preprocessing.py     # pipeline de preprocesare (partajat)
│   ├── model_training.py         # antrenare model de bază (Linear Regression)
│   ├── model_comparison.py       # compară 4 algoritmi, salvează cel mai bun
│   ├── model_evaluation.py       # evaluează modelul salvat, exemple predicții
│   └── predict_price.py          # testare interactivă opțională
├── models/
│   └── car_price_model.joblib    # modelul final antrenat (generat)
├── requirements.txt
└── README.md
Cerințe
Python 3.10+
Biblioteci: vezirequirements.txt
Instalare:

pip install -r requirements.txt
Cum rulezi proiectul (în ordine)
Toate comenzile se rulează din rădăcina proiectului (folderul car-price-prediction/, nu din src/).

1. Curățarea datelor
python3 src/data_cleaning.py
Citește data/cars.csv, redenumește coloanele, tratează valorile lipsă și elimină rândurile invalide (ani de fabricație imposibili, kilometraje absurde, prețuri extreme). Salvează rezultatul în data/cars_clean.csv.

2. Ingineria caracteristicilor
python3 src/feature_engineering.py
Adaugă caracteristici noi ( car_age, mileage_per_year, engine_volume_liters, is_newer_car, is_high_mileage, brand_model). Salvează rezultatul în data/cars_features.csv.

3. Antrenarea unui model de bază
python3 src/model_training.py
Antrenează un model Linear Regression simplu și îl salvează în models/car_price_model.joblib— un prim test rapid, "merge tot fluxul?".

4. Compararea mai multor modele
python3 src/model_comparison.py
Antrenează și compară 4 algoritmi (Linear Regression, Decision Tree, Random Forest, Gradient Boosting) pe același set de antrenament/testare, afișează un tabel comparativ și salvează cel mai bun model , înlocuind modelul de bază din pasul anterior.

5. Evaluarea modelului final
python3 src/model_evaluation.py
Încarcă modelul salvat, calculează metricile de regresie (MAE, MSE, RMSE, R²) și afișează exemple concrete (preț real vs. preț estimat).

6. Testare interactivă (opțional)
python3 src/predict_price.py
Introdu caracteristicile unei mașini din consolă și primești prețul estimat.

7. Notebook-ul complet
Deschide notebooks/01_eda.ipynbpentru analiza completă, cu explorarea vizuală a datelor, explicații pas cu pas și concluzia finală.

Rezultate obținute
Model	MAE (USD)	RMSE (USD)	R²
Regresor aleatoriu de pădure	1173	2293	0,912
Regresor de amplificare a gradientului	1451	2630	0,885
Regresorul arborelui de decizie	1405	2781	0,871
Regresie liniară	2446	4425	0,673
Modelul ales: Random Forest Regressor — cel mai bun echilibru între acuratețe (cea mai mică eroare) și robustețe (nu produce predicții absurde, precum prețuri negative, spre deosebire de Linear Regression).

MAE de ~1173 USD înseamnă că, în medie, modelul greșește cu aproximativ 1173 USD față de prețul real al mașinii — iar R² de 0.91 arată că modelul explică peste 91% din variația prețurilor din setul de testare.

Despre data stabilită
cars.csvconține 56.244 de mașini second-hand, cu coloanele: make, model( priceUSDvariabila țintă), year, condition, mileage(kilometers), fuel_type, volume(cm3), color, transmission, drive_unit, segment.

Decizii cheie de curățare și inginerie a caracteristicilor
Coloanele mileage(kilometers)și volume(cm3)au fost redenumite ( mileage_km, volume_cm3) pentru a evita caracterele speciale în cod.
volume_cm3lipsește exclusiv la mașinile electrice — completat cu 0.
drive_unitși segmentlipsă au fost tratate ca o categorie proprie "unknown", nu elimina (ar fi însemnat pierderea a ~10% din date).
Au fost eliminate: mașini cu an de fabricație sub 1950, kilometraj peste 1.000.000 km, și prețuri sub 100 USD sau peste percentila 99.9 (outlieri).
Coloana modelare peste 1000 de valori distincte — pentru a evita o codificare one-hot excesiv de mare, caracteristica derivată brand_model (marcă+model) este grupată la preprocesare, păstrând doar cele mai frecvente 40 de combinații; restul devin "other".
Observații și posibilități posibile
Ar putea fi testată optimizarea hiperparametrilor (GridSearchCV / RandomizedSearchCV) pentru Random Forest și Gradient Boosting.
O transformare logaritmică a variabilei țintă ( log(priceUSD)) ar putea putea stabilitatea modelelor liniare pe distribuția asimetrică a prețurilor.
