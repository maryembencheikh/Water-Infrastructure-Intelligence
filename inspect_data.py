
from pathlib import Path
import pandas as pd

data_dir = Path("data/raw")

files = {
    "Training values": "Training set values.csv",
    "Training labels": "Training set labels.csv",
    "Test values": "Test set values.csv",
    "Submission format": "SubmissionFormat.csv",
}

for name, filename in files.items():
    path = data_dir / filename

    if not path.exists():
        print(f"\nFichier introuvable : {path}")
        continue
    df = pd.read_csv(path)

    print(f"\n{'=' * 50}")
    print(name)
    print(f"Dimensions : {df.shape}")
    print(f"Colonnes : {df.columns.tolist()}")
    print("\nPremières lignes :")
    print(df.head())
    print("\nTypes des colonnes :")
    print(df.dtypes)



# Audit initial de la qualité des données

train = pd.read_csv(data_dir / files["Training values"])
labels = pd.read_csv(data_dir / files["Training labels"])

print("\n--- VALEURS MANQUANTES ---")
missing = train.isna().sum()
print(missing[missing > 0].sort_values(ascending=False))

print("\n--- LIGNES DUPLIQUÉES ---")
print(f"Training values : {train.duplicated().sum()}")
print(f"Training labels : {labels.duplicated().sum()}")

print("\n--- IDENTIFIANTS ---")
print(f"IDs uniques dans train : {train['id'].nunique()}")
print(f"IDs uniques dans labels : {labels['id'].nunique()}")
print(f"IDs communs aux deux fichiers : "
      f"{train['id'].isin(labels['id']).sum()}")

print("\n--- RÉPARTITION DES CLASSES ---")
print(labels["status_group"].value_counts())
print("\nProportions (%) :")
print((labels["status_group"].value_counts(normalize=True) * 100).round(2))    



print("\n--- STATISTIQUES NUMÉRIQUES ---")
print(train.select_dtypes(include="number").describe().T)

print("\n--- ANNÉES DE CONSTRUCTION ---")
print(train["construction_year"].value_counts().sort_index().head(15))
print("\nAnnées les plus récentes :")
print(train["construction_year"].value_counts().sort_index().tail(10))

print("\n--- COORDONNÉES ---")
print("Longitude égale à 0 :", (train["longitude"] == 0).sum())
print("Latitude égale à 0 :", (train["latitude"] == 0).sum())
print("Latitude hors plage :", (
    ~train["latitude"].between(-90, 90)
).sum())
print("Longitude hors plage :", (
    ~train["longitude"].between(-180, 180)
).sum())

print("\n--- CATÉGORIES LES PLUS FRÉQUENTES ---")
for col in ["quantity", "water_quality", "source", "extraction_type"]:
    print(f"\n{col}")
    print(train[col].value_counts(dropna=False).head(10))



print("\n--- ANALYSE DES ZÉROS ---")

for col in [
    "amount_tsh",
    "gps_height",
    "longitude",
    "latitude",
    "num_private",
    "population",
    "construction_year",
]:
    print(f"{col}: {(train[col] == 0).sum()} valeurs égales à zéro")

print("\n--- PÉRIODE D'ENREGISTREMENT ---")
dates = pd.to_datetime(train["date_recorded"], errors="coerce")
print("Date la plus ancienne :", dates.min())
print("Date la plus récente :", dates.max())
print("Dates non convertibles :", dates.isna().sum())


missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("\n--- VALEURS MANQUANTES ---")
print((missing / len(train) * 100).round(2))


print("\n--- VALEURS UNKNOWN ---")

for col in train.select_dtypes(include="object").columns:
    count = (
        train[col]
        .astype("string")
        .str.lower()
        .str.strip()
        .eq("unknown")
        .sum()
    )

    if count > 0:
        print(f"{col}: {count} ({count / len(train) * 100:.2f}%)")


print("\n--- VALEURS SUSPECTES ---")

print("Construction year = 0:",
      (train["construction_year"] == 0).sum())

print("Longitude = 0:",
      (train["longitude"] == 0).sum())

print("Population = 0:",
      (train["population"] == 0).sum())

print("GPS height = 0:",
      (train["gps_height"] == 0).sum())

print("Amount TSH = 0:",
      (train["amount_tsh"] == 0).sum())        


print("\n--- ANNÉES DE CONSTRUCTION ---")

years = train.loc[train["construction_year"] != 0, "construction_year"]

print("Année minimale :", years.min())
print("Année maximale :", years.max())
print("Nombre d'années distinctes :", years.nunique())

print("\nAnnées les plus fréquentes :")
print(years.value_counts().head(10))


print("\n--- COORDONNÉES GPS ---")

zero_longitude = train[train["longitude"] == 0]

print("Lignes avec longitude = 0 :", len(zero_longitude))

print("\nExemples :")
print(zero_longitude[["latitude", "longitude", "region", "district_code"]].head(10))


print("\n--- COORDONNÉES ANORMALES ---")

mask = (train["longitude"] == 0) & (train["latitude"].abs() < 0.000001)

print("Longitude = 0 et latitude proche de 0 :", mask.sum())
print("Pourcentage :", round(mask.mean() * 100, 2), "%")



print("\n--- VARIABLE CIBLE ---")

target = labels["status_group"]

print(target.value_counts())

print("\nPourcentages :")
print((target.value_counts(normalize=True) * 100).round(2))