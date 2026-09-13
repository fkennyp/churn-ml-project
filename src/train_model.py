import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# =====================================================
# Path configuration
# =====================================================
ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_PATH = ROOT / "data" / "processed" / "churn_clean.csv"

API_DIR = ROOT / "api"
MODEL_PATH = API_DIR / "model.pkl"
MODEL_META_PATH = API_DIR / "model_meta.json"

# =====================================================
# Feature configuration
# =====================================================
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
]

CATEGORICAL_FEATURES = [
    "Contract",
    "InternetService",
    "PaymentMethod",
]

TARGET_COLUMN = "Churn"


def load_processed_data() -> pd.DataFrame:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed data file not found.\n"
            "Run this first: python src/data_prep.py"
        )

    return pd.read_csv(PROCESSED_DATA_PATH)


def build_model(scale_pos_weight: float) -> Pipeline:
    """
    Build ML pipeline:
    - Numeric features: used as-is (passthrough)
    - Categorical features: one-hot encoding
    - Classifier: XGBoost
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                ),
            ),
        ]
    )

    return model


def main():
    print("Loading processed data...")
    df = load_processed_data()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)

    # Churn datasets are usually imbalanced.
    # scale_pos_weight makes the model pay more attention to the churn class.
    scale_pos_weight = negative_count / max(positive_count, 1)

    print("Building model...")
    model = build_model(scale_pos_weight=scale_pos_weight)

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Confusion matrix counts (the "4 boxes")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"Caught (TP): {tp} | Missed (FN): {fn} | False alarm (FP): {fp} | Safe (TN): {tn}")

    print("=" * 50)
    print("Classification Report")
    print("=" * 50)
    print(classification_report(y_test, y_pred))

    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    API_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    model_metadata = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "positive_count_training": positive_count,
        "negative_count_training": negative_count,
        "scale_pos_weight": scale_pos_weight,
    }

    MODEL_META_PATH.write_text(json.dumps(model_metadata, indent=2))

    print("=" * 50)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {MODEL_META_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()