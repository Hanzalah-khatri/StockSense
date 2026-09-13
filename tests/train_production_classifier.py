"""
StockSense - Production Classifier Training

Milestone 5 - Step 1

Trains the confirmed Extra Trees classifier on the
full available training dataset.

This script creates a NEW production artifact.
It does NOT overwrite the existing Random Forest model.
"""

from pathlib import Path
import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier

from services.market_data import fetch_stock_data
from services.prediction_service import PredictionService
from services.validation_service import ValidationService


# =========================================================
# CONFIGURATION
# =========================================================

TICKER = "AAPL"

FEATURES = [
    "Return_Lag_1",
    "Return_Lag_3",
    "Return_Lag_5",
    "Volatility",
    "Volume_Change",
]

MODEL_CONFIG = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_split": 10,
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced",
}

MODEL_PATH = Path(
    "models/stocksense_extra_trees_classifier.pkl"
)

METADATA_PATH = Path(
    "models/extra_trees_model_metadata.json"
)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 75)
    print("STOCKSENSE PRODUCTION CLASSIFIER TRAINING")
    print("=" * 75)

    # -----------------------------------------------------
    # Create directories
    # -----------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Fetch market data
    # -----------------------------------------------------

    print("\nFetching market data...")

    market_data = fetch_stock_data(
        TICKER,
        period="2y"
    )

    print(
        f"Raw market rows: {len(market_data)}"
    )

    # -----------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------

    prediction_service = PredictionService()

    features = prediction_service.prepare_features(
        market_data
    )

    print(
        f"Engineered rows: {len(features)}"
    )

    # -----------------------------------------------------
    # Prepare validation data
    # -----------------------------------------------------

    validation_service = ValidationService()

    X, y = validation_service.prepare_data(
        features
    )

    # Keep ONLY confirmed production features.
    X = X[FEATURES].copy()

    print(
        f"Training observations: {len(X)}"
    )

    print(
        f"Training features: {len(FEATURES)}"
    )

    print("\nFeatures:")

    for feature in FEATURES:
        print(f"  - {feature}")

    # -----------------------------------------------------
    # Target distribution
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("TARGET DISTRIBUTION")
    print("=" * 75)

    target_counts = y.value_counts()

    down_count = int(
        target_counts.get(0, 0)
    )

    up_count = int(
        target_counts.get(1, 0)
    )

    total = len(y)

    print(
        f"DOWN: {down_count} "
        f"({down_count / total * 100:.2f}%)"
    )

    print(
        f"UP:   {up_count} "
        f"({up_count / total * 100:.2f}%)"
    )

    # -----------------------------------------------------
    # Final NaN check
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("DATA QUALITY CHECK")
    print("=" * 75)

    print(
        f"Missing feature values: "
        f"{X.isna().sum().sum()}"
    )

    print(
        f"Missing target values: "
        f"{y.isna().sum()}"
    )

    if X.isna().sum().sum() > 0:
        raise ValueError(
            "Training data contains missing feature values."
        )

    if y.isna().sum() > 0:
        raise ValueError(
            "Training target contains missing values."
        )

    # -----------------------------------------------------
    # Create final model
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("TRAINING FINAL EXTRA TREES MODEL")
    print("=" * 75)

    print(
        f"n_estimators: "
        f"{MODEL_CONFIG['n_estimators']}"
    )

    print(
        f"max_depth: "
        f"{MODEL_CONFIG['max_depth']}"
    )

    print(
        f"min_samples_split: "
        f"{MODEL_CONFIG['min_samples_split']}"
    )

    print(
        f"class_weight: "
        f"{MODEL_CONFIG['class_weight']}"
    )

    model = ExtraTreesClassifier(
        **MODEL_CONFIG
    )

    model.fit(
        X,
        y
    )

    print("\nModel training complete.")

    # -----------------------------------------------------
    # Training sanity check
    # -----------------------------------------------------

    training_predictions = model.predict(
        X
    )

    training_accuracy = (
        training_predictions == y
    ).mean()

    print(
        f"Training accuracy: "
        f"{training_accuracy * 100:.2f}%"
    )

    print(
        "\nNote: training accuracy is NOT "
        "the validation accuracy."
    )

    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("FINAL FEATURE IMPORTANCE")
    print("=" * 75)

    importance_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    for _, row in importance_df.iterrows():

        print(
            f"{row['feature']:<20}"
            f"{row['importance']:.4f}"
        )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("SAVING PRODUCTION MODEL")
    print("=" * 75)

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"Model saved to:\n"
        f"{MODEL_PATH.resolve()}"
    )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    metadata = {
        "model_name": "Extra Trees Classifier",
        "model_type": "ExtraTreesClassifier",
        "ticker": TICKER,
        "target": "next_day_direction",
        "target_mapping": {
            "0": "DOWN",
            "1": "UP",
        },
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "hyperparameters": MODEL_CONFIG,
        "training_rows": len(X),
        "training_period": "2y",
        "validation": {
            "method": "Expanding walk-forward validation",
            "confirmation_splits": 7,
            "confirmation_min_train_size": 150,
            "accuracy": 0.5828,
            "precision": 0.5865,
            "recall": 0.7531,
            "f1": 0.6595,
            "roc_auc": 0.5649,
            "majority_baseline": 0.5364,
            "improvement_over_baseline": 0.0464,
        },
        "feature_selection": {
            "method": "Feature importance + walk-forward comparison",
            "selected_features": FEATURES,
        },
        "trained_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        f"Metadata saved to:\n"
        f"{METADATA_PATH.resolve()}"
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("PRODUCTION MODEL READY")
    print("=" * 75)

    print(
        "Model: Extra Trees Classifier"
    )

    print(
        "Features: Top 5"
    )

    print(
        "Confirmed validation accuracy: 58.28%"
    )

    print(
        "Confirmed ROC-AUC: 0.5649"
    )

    print(
        "Improvement over baseline: +4.64 pp"
    )

    print(
        "\nExisting Random Forest model was NOT overwritten."
    )

    print("\n" + "=" * 75)
    print("Production classifier training complete.")
    print("=" * 75)


if __name__ == "__main__":
    main()