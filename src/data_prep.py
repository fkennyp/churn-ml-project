from pathlib import Path

import pandas as pd


# =====================================================
# Path configuration
# =====================================================
ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = ROOT / "data" / "raw" / "churn_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DIR / "churn_clean.csv"


# =====================================================
# Feature configuration
# =====================================================
SELECTED_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "Churn",
]


def load_raw_data(path: Path) -> pd.DataFrame:
    """
    Load raw churn dataset from local CSV.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\n"
            "Pastikan kamu sudah download dataset Kaggle dan rename jadi churn_data.csv"
        )

    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data and prepare churn target.
    """
    df = df.copy()

    # Pastikan kolom numerik benar-benar numerik
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

    # Hapus baris dengan data numerik kosong
    df = df.dropna(subset=["tenure", "MonthlyCharges"])

    # Bersihkan kolom kategorikal dari spasi berlebih
    categorical_columns = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "Churn",
    ]

    for column in categorical_columns:
        df[column] = df[column].astype(str).str.strip()

    # Ubah target Churn:
    # Yes = 1
    # No  = 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Hapus target yang tidak bisa di-map
    df = df.dropna(subset=["Churn"])
    df["Churn"] = df["Churn"].astype(int)

    return df


def main():
    print("Loading raw data...")
    raw_df = load_raw_data(RAW_DATA_PATH)

    print("Selecting columns...")
    raw_df = raw_df[SELECTED_COLUMNS]

    print("Cleaning data...")
    clean_df = clean_data(raw_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("=" * 50)
    print(f"Data berhasil disimpan di: {PROCESSED_DATA_PATH}")
    print("=" * 50)
    print(clean_df.head())
    print("\nDistribusi Churn:")
    print(clean_df["Churn"].value_counts(normalize=True))


if __name__ == "__main__":
    main()