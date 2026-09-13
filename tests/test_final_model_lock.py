"""
StockSense — Milestone 3.17
Final Model Lock & Save

Purpose:
Train the final production Random Forest classifier and save it
as a reusable model artifact.

This is NOT another model comparison.

The model has already been selected through:
- baseline comparison
- hyperparameter tuning
- holdout validation
- probability analysis
- calibration testing
- explainability
- feature subset validation
- stability testing

Final model:
Tuned Random Forest + all 30 features
"""

import json
import os
import pickle
import warnings
from datetime import datetime, timezone

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from services.market_data import fetch_stock_data
from services.prediction_service import PredictionService


warnings.filterwarnings("ignore")


# =========================================================
# CONFIGURATION
# =========================================================

TICKER = "AAPL"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "stocksense_rf_classifier.pkl",
)

METADATA_PATH = os.path.join(
    MODEL_DIR,
    "model_metadata.json",
)


# =========================================================
# FINAL MODEL CONFIGURATION
# =========================================================

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1,
}


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("STOCKSENSE — MILESTONE 3.17")
    print("=" * 70)

    print()
    print("FINAL MODEL LOCK & SAVE")

    print()
    print(f"Ticker: {TICKER}")
    print("Model: Tuned Random Forest")
    print("Features: All 30 features")

    # =====================================================
    # 1. CREATE MODEL DIRECTORY
    # =====================================================

    print()
    print("=" * 70)
    print("1. PREPARING MODEL DIRECTORY")
    print("=" * 70)

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    print()
    print(
        f"Model directory ready: {MODEL_DIR}"
    )


    # =====================================================
    # 2. LOAD MARKET DATA
    # =====================================================

    print()
    print("=" * 70)
    print("2. LOADING MARKET DATA")
    print("=" * 70)

    market_data = fetch_stock_data(
        TICKER,
        period="2y",
    )

    print()
    print(
        f"Loaded {len(market_data)} market rows"
    )


    # =====================================================
    # 3. PREPARE FEATURES
    # =====================================================

    print()
    print("=" * 70)
    print("3. PREPARING FINAL FEATURES")
    print("=" * 70)

    prediction_service = PredictionService()

    features = (
        prediction_service.prepare_features(
            market_data
        )
    )

    X, y = (
        prediction_service.prepare_ml_data(
            features
        )
    )

    # -----------------------------------------------------
    # Convert return target into direction
    #
    # UP   = 1
    # DOWN = 0
    # -----------------------------------------------------

    y = (y > 0).astype(int)

    feature_columns = (
        prediction_service.get_feature_columns()
    )

    X = X[feature_columns]

    print()
    print(
        f"Usable rows: {len(X)}"
    )

    print(
        f"Feature count: {len(feature_columns)}"
    )

    print()
    print("Features used by final model:")

    for index, feature in enumerate(
        feature_columns,
        start=1,
    ):

        print(
            f"{index:02d}. {feature}"
        )


    # =====================================================
    # 4. TARGET DISTRIBUTION
    # =====================================================

    print()
    print("=" * 70)
    print("4. FINAL TRAINING TARGET")
    print("=" * 70)

    up_count = int(
        np.sum(y == 1)
    )

    down_count = int(
        np.sum(y == 0)
    )

    total = len(y)

    print()

    print(
        f"UP   : {up_count} "
        f"({up_count / total:.2%})"
    )

    print(
        f"DOWN : {down_count} "
        f"({down_count / total:.2%})"
    )


    # =====================================================
    # 5. CREATE FINAL MODEL
    # =====================================================

    print()
    print("=" * 70)
    print("5. CREATING FINAL MODEL")
    print("=" * 70)

    model = RandomForestClassifier(
        **RF_PARAMS
    )

    print()

    print(
        "Final hyperparameters:"
    )

    for key, value in RF_PARAMS.items():

        print(
            f"  {key}: {value}"
        )


    # =====================================================
    # 6. TRAIN FINAL MODEL
    # =====================================================

    print()
    print("=" * 70)
    print("6. TRAINING FINAL MODEL")
    print("=" * 70)

    print()
    print(
        "Training on all available historical "
        "training observations..."
    )

    model.fit(
        X,
        y,
    )

    print()
    print(
        "✓ Final model trained successfully."
    )


    # =====================================================
    # 7. VERIFY MODEL
    # =====================================================

    print()
    print("=" * 70)
    print("7. VERIFYING MODEL")
    print("=" * 70)

    # -----------------------------------------------------
    # Training prediction
    #
    # This is NOT a real-world performance metric.
    # It only verifies that the saved model is functional.
    # -----------------------------------------------------

    training_predictions = model.predict(
        X
    )

    training_accuracy = np.mean(
        training_predictions == y
    )

    probabilities = model.predict_proba(
        X
    )

    print()

    print(
        "Model verification:"
    )

    print(
        f"  Classes: "
        f"{model.classes_.tolist()}"
    )

    print(
        f"  Number of trees: "
        f"{len(model.estimators_)}"
    )

    print(
        f"  Training rows: "
        f"{len(X)}"
    )

    print(
        f"  Training accuracy: "
        f"{training_accuracy:.2%}"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "Training accuracy is NOT used as the model's"
    )

    print(
        "real-world performance estimate."
    )

    print(
        "The previously measured holdout and stability"
    )

    print(
        "results remain the valid evaluation."
    )


    # =====================================================
    # 8. SAVE MODEL
    # =====================================================

    print()
    print("=" * 70)
    print("8. SAVING FINAL MODEL")
    print("=" * 70)

    with open(
        MODEL_PATH,
        "wb",
    ) as file:

        pickle.dump(
            model,
            file,
        )

    print()

    print(
        f"✓ Model saved:"
    )

    print(
        f"  {MODEL_PATH}"
    )


    # =====================================================
    # 9. SAVE METADATA
    # =====================================================

    print()
    print("=" * 70)
    print("9. SAVING MODEL METADATA")
    print("=" * 70)

    metadata = {
        "model_name": (
            "StockSense Tuned Random Forest"
        ),

        "model_type": (
            "RandomForestClassifier"
        ),

        "ticker_used_for_training": TICKER,

        "target": (
            "Next-day direction"
        ),

        "target_encoding": {
            "DOWN": 0,
            "UP": 1,
        },

        "feature_count": len(
            feature_columns
        ),

        "features": feature_columns,

        "hyperparameters": RF_PARAMS,

        "training_rows": len(X),

        "training_up_rows": up_count,

        "training_down_rows": down_count,

        "evaluation_summary": {
            "final_holdout_accuracy": 0.5574,
            "final_holdout_roc_auc": 0.5887,
            "stability_average_accuracy": 0.5227,
            "stability_average_roc_auc": 0.5759,
            "stability_verdict": (
                "Moderately Stable"
            ),
        },

        "model_selection_note": (
            "Selected after chronological validation, "
            "hyperparameter tuning, holdout testing, "
            "probability analysis, calibration testing, "
            "feature explainability, feature subset "
            "validation, and prediction stability testing."
        ),

        "limitations": [
            (
                "Directional prediction performance "
                "is modest."
            ),
            (
                "Market behavior is non-stationary."
            ),
            (
                "Prediction probabilities should not "
                "be interpreted as certainty."
            ),
            (
                "Model is for educational and analytical "
                "purposes, not financial advice."
            ),
        ],

        "created_at_utc": (
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

    print()

    print(
        "✓ Metadata saved:"
    )

    print(
        f"  {METADATA_PATH}"
    )


    # =====================================================
    # 10. RELOAD TEST
    # =====================================================

    print()
    print("=" * 70)
    print("10. RELOAD TEST")
    print("=" * 70)

    with open(
        MODEL_PATH,
        "rb",
    ) as file:

        loaded_model = pickle.load(
            file
        )

    reloaded_predictions = (
        loaded_model.predict(X)
    )

    reloaded_probabilities = (
        loaded_model.predict_proba(X)
    )

    predictions_match = np.array_equal(
        training_predictions,
        reloaded_predictions,
    )

    probabilities_match = np.allclose(
        probabilities,
        reloaded_probabilities,
    )

    print()

    print(
        f"Predictions match: "
        f"{'PASS' if predictions_match else 'FAIL'}"
    )

    print(
        f"Probabilities match: "
        f"{'PASS' if probabilities_match else 'FAIL'}"
    )


    # =====================================================
    # 11. FINAL STATUS
    # =====================================================

    print()
    print("=" * 70)
    print("11. FINAL MODEL STATUS")
    print("=" * 70)

    print()

    if (
        predictions_match
        and probabilities_match
    ):

        print(
            "🏆 FINAL MODEL LOCK: SUCCESS"
        )

        print()

        print(
            "The StockSense classifier is now "
            "saved and ready for application integration."
        )

    else:

        print(
            "❌ FINAL MODEL LOCK: FAILED"
        )

        print()

        print(
            "Reload verification failed."
        )

        return


    # =====================================================
    # 12. FILE SUMMARY
    # =====================================================

    print()
    print("=" * 70)
    print("12. CREATED FILES")
    print("=" * 70)

    print()

    print(
        f"✓ {MODEL_PATH}"
    )

    print(
        f"✓ {METADATA_PATH}"
    )

    print()

    print(
        "These files will be used by the Flask application."
    )


    # =====================================================
    # 13. FINAL CONCLUSION
    # =====================================================

    print()
    print("=" * 70)
    print("MILESTONE 3.17 COMPLETE")
    print("=" * 70)

    print()

    print(
        "ML DEVELOPMENT PHASE COMPLETE."
    )

    print()

    print(
        "Final classifier:"
    )

    print(
        "Tuned Random Forest"
    )

    print()

    print(
        "Features:"
    )

    print(
        "30 technical / market features"
    )

    print()

    print(
        "Final holdout accuracy:"
    )

    print(
        "55.74%"
    )

    print()

    print(
        "Final holdout ROC-AUC:"
    )

    print(
        "0.5887"
    )

    print()

    print(
        "Stability:"
    )

    print(
        "Moderately Stable"
    )

    print()

    print(
        "NEXT PHASE:"
    )

    print(
        "StockSense Application Integration"
    )

    print()

    print(
        "Next milestone:"
    )

    print(
        "Milestone 4.1 — Production Prediction Service"
    )

    print()

    print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()