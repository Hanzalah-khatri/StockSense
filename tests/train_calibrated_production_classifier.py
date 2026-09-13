"""
StockSense - Calibrated Production Classifier Training

Milestone 5 - Step 5A

Creates the production calibrated classifier:

    ExtraTreesClassifier
            +
    Sigmoid Probability Calibration
            ↓
    CalibratedClassifierCV

The existing Extra Trees production model is NOT overwritten.

Important:
The calibration approach was previously evaluated using
7-fold expanding walk-forward validation.

Confirmed calibration results:

Raw Extra Trees:
    Accuracy: 58.28%
    Brier Score: 0.2498
    Log Loss: 0.6949
    ROC-AUC: 0.5649

Calibrated Extra Trees:
    Accuracy: 57.62%
    Brier Score: 0.2454
    Log Loss: 0.6840
    ROC-AUC: 0.5723
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import joblib

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV

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

CALIBRATION_METHOD = "sigmoid"
CALIBRATION_CV = 3

MODEL_PATH = Path(
    "models/stocksense_calibrated_classifier.pkl"
)

METADATA_PATH = Path(
    "models/calibrated_model_metadata.json"
)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 75)
    print("STOCKSENSE CALIBRATED PRODUCTION CLASSIFIER")
    print("=" * 75)

    # -----------------------------------------------------
    # Create model directory
    # -----------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Fetch market data
    # -----------------------------------------------------

    print("\nFetching market data...")

    market_data = fetch_stock_data(
        TICKER,
        period="2y",
    )

    print(
        f"Raw market rows: {len(market_data)}"
    )

    # -----------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------

    prediction_service = PredictionService()

    features = (
        prediction_service.prepare_features(
            market_data
        )
    )

    print(
        f"Engineered rows: {len(features)}"
    )

    # -----------------------------------------------------
    # Prepare classification data
    # -----------------------------------------------------

    validation_service = ValidationService()

    X, y = validation_service.prepare_data(
        features
    )

    # Use ONLY the confirmed production features.
    X = X[
        FEATURES
    ].copy()

    print(
        f"Usable training rows: {len(X)}"
    )

    print(
        f"Production features: {len(FEATURES)}"
    )

    # -----------------------------------------------------
    # Data validation
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("DATA QUALITY CHECK")
    print("=" * 75)

    missing_features = (
        X.isna().sum().sum()
    )

    missing_target = (
        y.isna().sum()
    )

    print(
        f"Missing feature values: "
        f"{missing_features}"
    )

    print(
        f"Missing target values: "
        f"{missing_target}"
    )

    if missing_features > 0:

        raise ValueError(
            "Training features contain missing values."
        )

    if missing_target > 0:

        raise ValueError(
            "Training target contains missing values."
        )

    # -----------------------------------------------------
    # Class distribution
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("TARGET DISTRIBUTION")
    print("=" * 75)

    down_count = int(
        (y == 0).sum()
    )

    up_count = int(
        (y == 1).sum()
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
    # Base model
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("BASE EXTRA TREES MODEL")
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

    base_model = ExtraTreesClassifier(
        **MODEL_CONFIG
    )

    # -----------------------------------------------------
    # Calibrated model
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("CALIBRATED MODEL")
    print("=" * 75)

    print(
        f"Calibration method: "
        f"{CALIBRATION_METHOD}"
    )

    print(
        f"Calibration CV: "
        f"{CALIBRATION_CV}"
    )

    print(
        "\nTraining calibrated classifier..."
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method=CALIBRATION_METHOD,
        cv=CALIBRATION_CV,
    )

    calibrated_model.fit(
        X,
        y,
    )

    print(
        "Calibration training complete."
    )

    # -----------------------------------------------------
    # Sanity check
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("MODEL SANITY CHECK")
    print("=" * 75)

    predictions = calibrated_model.predict(
        X
    )

    probabilities = calibrated_model.predict_proba(
        X
    )

    training_accuracy = (
        predictions == y
    ).mean()

    print(
        f"Training accuracy: "
        f"{training_accuracy * 100:.2f}%"
    )

    print(
        "Training accuracy is NOT "
        "the validation accuracy."
    )

    # -----------------------------------------------------
    # Probability sanity check
    # -----------------------------------------------------

    probability_up = probabilities[
        :,
        1,
    ]

    print(
        f"\nMinimum UP probability: "
        f"{probability_up.min() * 100:.2f}%"
    )

    print(
        f"Maximum UP probability: "
        f"{probability_up.max() * 100:.2f}%"
    )

    print(
        f"Average UP probability: "
        f"{probability_up.mean() * 100:.2f}%"
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("SAVING CALIBRATED PRODUCTION MODEL")
    print("=" * 75)

    joblib.dump(
        calibrated_model,
        MODEL_PATH,
    )

    print(
        f"Model saved to:\n"
        f"{MODEL_PATH.resolve()}"
    )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    metadata = {
        "model_name": (
            "Calibrated Extra Trees Classifier"
        ),

        "model_type": (
            "CalibratedClassifierCV"
        ),

        "base_model_type": (
            "ExtraTreesClassifier"
        ),

        "ticker": TICKER,

        "target": (
            "next_day_direction"
        ),

        "target_mapping": {
            "0": "DOWN",
            "1": "UP",
        },

        "features": FEATURES,

        "feature_count": len(
            FEATURES
        ),

        "base_model_hyperparameters": (
            MODEL_CONFIG
        ),

        "calibration": {
            "method": CALIBRATION_METHOD,
            "cv": CALIBRATION_CV,
        },

        "training_rows": len(X),

        "training_period": "2y",

        "validation": {
            "method": (
                "Expanding walk-forward validation"
            ),

            "confirmation_splits": 7,

            "confirmation_min_train_size": 150,

            "raw_extra_trees": {
                "accuracy": 0.5828,
                "brier_score": 0.2498,
                "log_loss": 0.6949,
                "roc_auc": 0.5649,
            },

            "calibrated_extra_trees": {
                "accuracy": 0.5762,
                "brier_score": 0.2454,
                "log_loss": 0.6840,
                "roc_auc": 0.5723,
            },

            "brier_improvement": 0.0044,

            "log_loss_improvement": 0.0110,

            "accuracy_difference": -0.0066,
        },

        "selection_reason": (
            "Calibration improved Brier score "
            "and log loss while retaining similar "
            "classification performance."
        ),

        "trained_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(
        f"Metadata saved to:\n"
        f"{METADATA_PATH.resolve()}"
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("CALIBRATED PRODUCTION MODEL READY")
    print("=" * 75)

    print(
        "Model: Calibrated Extra Trees"
    )

    print(
        "Calibration: Sigmoid"
    )

    print(
        "Features: 5"
    )

    print(
        "Training rows: "
        f"{len(X)}"
    )

    print(
        "Confirmed accuracy: 57.62%"
    )

    print(
        "Confirmed Brier score: 0.2454"
    )

    print(
        "Confirmed Log Loss: 0.6840"
    )

    print(
        "Confirmed ROC-AUC: 0.5723"
    )

    print(
        "\nExisting Extra Trees model was NOT overwritten."
    )

    print(
        "\nNext step: test the calibrated artifact "
        "before integrating it into ClassifierService."
    )

    print("\n" + "=" * 75)
    print(
        "Calibrated production training complete."
    )
    print("=" * 75)


if __name__ == "__main__":
    main()
