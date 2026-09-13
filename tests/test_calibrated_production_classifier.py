"""
StockSense - Calibrated Production Classifier Test

Milestone 5 - Step 5B

Purpose:
    Independently verify the saved calibrated production model
    before integrating it into ClassifierService.

Checks:
    1. Model file exists
    2. Metadata file exists
    3. Model loads with joblib
    4. Model type is CalibratedClassifierCV
    5. Feature list matches production features
    6. Model exposes predict_proba()
    7. AAPL features can be prepared
    8. Prediction returns UP/DOWN
    9. Probabilities are valid
    10. Probabilities sum to 100%
    11. Metadata matches the artifact
"""

from pathlib import Path
import json

import joblib
import pandas as pd

from services.market_data import fetch_stock_data
from services.prediction_service import PredictionService


# =========================================================
# CONFIGURATION
# =========================================================

TICKER = "AAPL"

MODEL_PATH = Path(
    "models/stocksense_calibrated_classifier.pkl"
)

METADATA_PATH = Path(
    "models/calibrated_model_metadata.json"
)

EXPECTED_FEATURES = [
    "Return_Lag_1",
    "Return_Lag_3",
    "Return_Lag_5",
    "Volatility",
    "Volume_Change",
]


# =========================================================
# HELPER
# =========================================================

def check(condition, message):
    """
    Simple test assertion.
    """
    if not condition:
        raise AssertionError(
            f"FAILED: {message}"
        )

    print(
        f"PASS: {message}"
    )


# =========================================================
# MAIN TEST
# =========================================================

def main():

    print("=" * 75)
    print("STOCKSENSE CALIBRATED PRODUCTION MODEL TEST")
    print("=" * 75)

    # -----------------------------------------------------
    # 1. Check model file
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("1. MODEL FILE CHECK")
    print("=" * 75)

    check(
        MODEL_PATH.exists(),
        f"Model file exists: {MODEL_PATH}"
    )

    print(
        f"Model size: "
        f"{MODEL_PATH.stat().st_size / 1024:.2f} KB"
    )

    # -----------------------------------------------------
    # 2. Check metadata file
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("2. METADATA FILE CHECK")
    print("=" * 75)

    check(
        METADATA_PATH.exists(),
        f"Metadata file exists: {METADATA_PATH}"
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    print(
        f"Metadata model: "
        f"{metadata.get('model_name')}"
    )

    # -----------------------------------------------------
    # 3. Load model
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("3. MODEL LOADING")
    print("=" * 75)

    model = joblib.load(
        MODEL_PATH
    )

    print(
        f"Loaded model type: "
        f"{type(model).__name__}"
    )

    check(
        type(model).__name__
        == "CalibratedClassifierCV",
        "Model is CalibratedClassifierCV"
    )

    # -----------------------------------------------------
    # 4. Check probability support
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("4. PROBABILITY SUPPORT")
    print("=" * 75)

    check(
        hasattr(
            model,
            "predict_proba",
        ),
        "Model supports predict_proba()"
    )

    # -----------------------------------------------------
    # 5. Check metadata
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("5. METADATA VALIDATION")
    print("=" * 75)

    metadata_features = metadata.get(
        "features"
    )

    check(
        metadata_features
        == EXPECTED_FEATURES,
        "Metadata feature order matches production features"
    )

    check(
        metadata.get("feature_count")
        == 5,
        "Metadata feature count is 5"
    )

    check(
        metadata.get("base_model_type")
        == "ExtraTreesClassifier",
        "Base model is ExtraTreesClassifier"
    )

    calibration = metadata.get(
        "calibration",
        {}
    )

    check(
        calibration.get("method")
        == "sigmoid",
        "Calibration method is sigmoid"
    )

    check(
        calibration.get("cv")
        == 3,
        "Calibration CV is 3"
    )

    # -----------------------------------------------------
    # 6. Fetch current AAPL data
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("6. AAPL MARKET DATA")
    print("=" * 75)

    market_data = fetch_stock_data(
        TICKER,
        period="2y",
    )

    check(
        market_data is not None,
        "Market data returned successfully"
    )

    check(
        len(market_data) > 100,
        "Sufficient market history available"
    )

    print(
        f"Market rows: "
        f"{len(market_data)}"
    )

    # -----------------------------------------------------
    # 7. Prepare features
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("7. FEATURE PREPARATION")
    print("=" * 75)

    prediction_service = PredictionService()

    features = (
        prediction_service.prepare_features(
            market_data
        )
    )

    check(
        isinstance(
            features,
            pd.DataFrame,
        ),
        "Feature engineering returned DataFrame"
    )

    print(
        f"Feature rows: "
        f"{len(features)}"
    )

    # -----------------------------------------------------
    # Select latest valid row
    # -----------------------------------------------------

    missing_features = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in features.columns
    ]

    check(
        len(missing_features) == 0,
        "All production features are available"
    )

    X_live = (
        features[
            EXPECTED_FEATURES
        ]
        .dropna()
        .tail(1)
    )

    check(
        len(X_live) == 1,
        "One valid latest feature row prepared"
    )

    print("\nLatest production features:")

    for feature in EXPECTED_FEATURES:

        value = X_live.iloc[0][feature]

        print(
            f"  {feature:<18}: "
            f"{value:.6f}"
        )

    # -----------------------------------------------------
    # 8. Run prediction
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("8. MODEL PREDICTION")
    print("=" * 75)

    prediction = model.predict(
        X_live
    )

    check(
        len(prediction) == 1,
        "Model returned one prediction"
    )

    predicted_class = int(
        prediction[0]
    )

    check(
        predicted_class in [0, 1],
        "Prediction is a valid class"
    )

    direction = (
        "UP"
        if predicted_class == 1
        else "DOWN"
    )

    print(
        f"Predicted direction: "
        f"{direction}"
    )

    # -----------------------------------------------------
    # 9. Probability prediction
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("9. CALIBRATED PROBABILITY")
    print("=" * 75)

    probabilities = (
        model.predict_proba(
            X_live
        )
    )

    check(
        probabilities.shape
        == (1, 2),
        "Probability output shape is (1, 2)"
    )

    probability_down = float(
        probabilities[0][0]
    )

    probability_up = float(
        probabilities[0][1]
    )

    print(
        f"Probability DOWN: "
        f"{probability_down * 100:.2f}%"
    )

    print(
        f"Probability UP:   "
        f"{probability_up * 100:.2f}%"
    )

    # -----------------------------------------------------
    # 10. Probability validation
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("10. PROBABILITY VALIDATION")
    print("=" * 75)

    check(
        0.0 <= probability_down <= 1.0,
        "DOWN probability is between 0 and 1"
    )

    check(
        0.0 <= probability_up <= 1.0,
        "UP probability is between 0 and 1"
    )

    probability_sum = (
        probability_down
        + probability_up
    )

    print(
        f"Probability sum: "
        f"{probability_sum:.6f}"
    )

    check(
        abs(
            probability_sum - 1.0
        ) < 1e-6,
        "Probabilities sum to 1.0"
    )

    # -----------------------------------------------------
    # 11. Prediction consistency
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("11. PREDICTION CONSISTENCY")
    print("=" * 75)

    probability_direction = (
        "UP"
        if probability_up >= probability_down
        else "DOWN"
    )

    print(
        f"Class prediction: "
        f"{direction}"
    )

    print(
        f"Probability direction: "
        f"{probability_direction}"
    )

    check(
        direction
        == probability_direction,
        "Class prediction agrees with highest probability"
    )

    # -----------------------------------------------------
    # 12. Display model metadata
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("12. PRODUCTION MODEL INFORMATION")
    print("=" * 75)

    print(
        f"Model name: "
        f"{metadata.get('model_name')}"
    )

    print(
        f"Model type: "
        f"{metadata.get('model_type')}"
    )

    print(
        f"Base model: "
        f"{metadata.get('base_model_type')}"
    )

    print(
        f"Ticker: "
        f"{metadata.get('ticker')}"
    )

    print(
        f"Training rows: "
        f"{metadata.get('training_rows')}"
    )

    print(
        f"Features: "
        f"{metadata.get('feature_count')}"
    )

    print(
        f"Calibration: "
        f"{calibration.get('method')}"
    )

    # -----------------------------------------------------
    # 13. Validation metrics from metadata
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("13. CONFIRMED VALIDATION PERFORMANCE")
    print("=" * 75)

    validation = metadata.get(
        "validation",
        {}
    )

    calibrated_results = validation.get(
        "calibrated_extra_trees",
        {}
    )

    print(
        f"Walk-forward accuracy: "
        f"{calibrated_results.get('accuracy', 0) * 100:.2f}%"
    )

    print(
        f"Brier score: "
        f"{calibrated_results.get('brier_score', 0):.4f}"
    )

    print(
        f"Log loss: "
        f"{calibrated_results.get('log_loss', 0):.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{calibrated_results.get('roc_auc', 0):.4f}"
    )

    # -----------------------------------------------------
    # FINAL RESULT
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("FINAL TEST RESULT")
    print("=" * 75)

    print(
        "PASS: Calibrated production classifier "
        "loaded and predicted successfully."
    )

    print(
        "\nThe existing ClassifierService has NOT "
        "been modified."
    )

    print(
        "The raw Extra Trees production model "
        "remains available as a backup."
    )

    print("\n" + "=" * 75)
    print("STEP 5B COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
