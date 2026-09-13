"""
StockSense - Probability Calibration Test

Milestone 5 - Step 4

Compares:
    1. Raw Extra Trees probabilities
    2. Calibrated Extra Trees probabilities

Validation:
    Expanding walk-forward validation

Important:
    Calibration is performed INSIDE each validation fold.
    The test fold is never used to fit the calibrator.

This prevents data leakage.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

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

N_SPLITS = 7
MIN_TRAIN_SIZE = 150

# Number of internal folds used by the calibrator.
#
# This must be smaller than the minimum training fold size
# and must not create an invalid calibration split.
CALIBRATION_CV = 3

REPORT_PATH = Path(
    "reports/probability_calibration.csv"
)


# =========================================================
# WALK-FORWARD SPLITS
# =========================================================

def create_expanding_splits(
    n_samples,
    n_splits=7,
    min_train_size=150,
):
    """
    Create expanding walk-forward validation splits.

    Example:

        Train       Test
        [------]    [---]
        [---------] [---]
        [------------][---]
    """

    if n_samples <= min_train_size:
        raise ValueError(
            "Not enough observations for walk-forward validation."
        )

    remaining = n_samples - min_train_size

    test_size = remaining // n_splits

    if test_size < 1:
        raise ValueError(
            "Test fold size is too small."
        )

    splits = []

    for fold in range(n_splits):

        train_end = (
            min_train_size
            + fold * test_size
        )

        if fold == n_splits - 1:

            test_end = n_samples

        else:

            test_end = (
                train_end
                + test_size
            )

        train_indices = np.arange(
            0,
            train_end
        )

        test_indices = np.arange(
            train_end,
            test_end
        )

        if len(test_indices) == 0:
            continue

        splits.append(
            (
                train_indices,
                test_indices,
            )
        )

    return splits


# =========================================================
# MODEL FACTORY
# =========================================================

def create_base_model():
    """
    Create the exact confirmed production
    Extra Trees configuration.
    """

    return ExtraTreesClassifier(
        **MODEL_CONFIG
    )


# =========================================================
# CALIBRATION FACTORY
# =========================================================

def create_calibrated_model():
    """
    Create an Extra Trees model with probability
    calibration.

    sigmoid calibration is used because the dataset
    is relatively small.
    """

    base_model = create_base_model()

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=CALIBRATION_CV,
    )

    return calibrated_model


# =========================================================
# SAFE METRIC CALCULATION
# =========================================================

def calculate_metrics(
    y_true,
    predictions,
    probabilities,
):
    """
    Calculate classification and probability metrics.
    """

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    brier = brier_score_loss(
        y_true,
        probabilities,
    )

    logloss = log_loss(
        y_true,
        probabilities,
        labels=[0, 1],
    )

    # ROC-AUC requires both classes in y_true.
    if len(np.unique(y_true)) == 2:

        auc = roc_auc_score(
            y_true,
            probabilities,
        )

    else:

        auc = np.nan

    return {
        "accuracy": accuracy,
        "brier_score": brier,
        "log_loss": logloss,
        "roc_auc": auc,
    }


# =========================================================
# MAIN TEST
# =========================================================

def main():

    print("=" * 75)
    print("STOCKSENSE - PROBABILITY CALIBRATION TEST")
    print("=" * 75)

    # -----------------------------------------------------
    # Fetch data
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
    # Prepare classification dataset
    # -----------------------------------------------------

    validation_service = ValidationService()

    X, y = validation_service.prepare_data(
        features
    )

    X = X[
        FEATURES
    ].copy()

    y = y.reset_index(
        drop=True
    )

    X = X.reset_index(
        drop=True
    )

    print(
        f"Usable observations: {len(X)}"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    # -----------------------------------------------------
    # Data quality
    # -----------------------------------------------------

    if X.isna().any().any():

        raise ValueError(
            "Feature data contains missing values."
        )

    if y.isna().any():

        raise ValueError(
            "Target contains missing values."
        )

    # -----------------------------------------------------
    # Class distribution
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("CLASS DISTRIBUTION")
    print("=" * 75)

    down = int(
        (y == 0).sum()
    )

    up = int(
        (y == 1).sum()
    )

    total = len(y)

    print(
        f"DOWN: {down} "
        f"({down / total * 100:.2f}%)"
    )

    print(
        f"UP:   {up} "
        f"({up / total * 100:.2f}%)"
    )

    # -----------------------------------------------------
    # Create validation folds
    # -----------------------------------------------------

    splits = create_expanding_splits(
        n_samples=len(X),
        n_splits=N_SPLITS,
        min_train_size=MIN_TRAIN_SIZE,
    )

    print("\n" + "=" * 75)
    print("VALIDATION CONFIGURATION")
    print("=" * 75)

    print(
        f"Ticker: {TICKER}"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    print(
        f"Splits: {len(splits)}"
    )

    print(
        f"Minimum train size: "
        f"{MIN_TRAIN_SIZE}"
    )

    print(
        "Validation: Expanding walk-forward"
    )

    print(
        "Calibration: Sigmoid"
    )

    print(
        f"Calibration CV: "
        f"{CALIBRATION_CV}"
    )

    # -----------------------------------------------------
    # Storage
    # -----------------------------------------------------

    fold_results = []

    raw_all_y = []
    raw_all_probabilities = []
    raw_all_predictions = []

    calibrated_all_y = []
    calibrated_all_probabilities = []
    calibrated_all_predictions = []

    # -----------------------------------------------------
    # Walk-forward validation
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("WALK-FORWARD CALIBRATION RESULTS")
    print("=" * 75)

    for fold_number, (
        train_indices,
        test_indices,
    ) in enumerate(
        splits,
        start=1,
    ):

        X_train = X.iloc[
            train_indices
        ]

        y_train = y.iloc[
            train_indices
        ]

        X_test = X.iloc[
            test_indices
        ]

        y_test = y.iloc[
            test_indices
        ]

        print(
            f"\nFold {fold_number}"
        )

        print(
            f"Train: {len(X_train)} | "
            f"Test: {len(X_test)}"
        )

        # -------------------------------------------------
        # Raw Extra Trees
        # -------------------------------------------------

        raw_model = create_base_model()

        raw_model.fit(
            X_train,
            y_train,
        )

        raw_predictions = (
            raw_model.predict(
                X_test
            )
        )

        raw_probabilities = (
            raw_model.predict_proba(
                X_test
            )[:, 1]
        )

        raw_metrics = calculate_metrics(
            y_test,
            raw_predictions,
            raw_probabilities,
        )

        # -------------------------------------------------
        # Calibrated Extra Trees
        # -------------------------------------------------

        calibrated_model = (
            create_calibrated_model()
        )

        calibrated_model.fit(
            X_train,
            y_train,
        )

        calibrated_predictions = (
            calibrated_model.predict(
                X_test
            )
        )

        calibrated_probabilities = (
            calibrated_model.predict_proba(
                X_test
            )[:, 1]
        )

        calibrated_metrics = (
            calculate_metrics(
                y_test,
                calibrated_predictions,
                calibrated_probabilities,
            )
        )

        # -------------------------------------------------
        # Store global predictions
        # -------------------------------------------------

        raw_all_y.extend(
            y_test.tolist()
        )

        raw_all_probabilities.extend(
            raw_probabilities.tolist()
        )

        raw_all_predictions.extend(
            raw_predictions.tolist()
        )

        calibrated_all_y.extend(
            y_test.tolist()
        )

        calibrated_all_probabilities.extend(
            calibrated_probabilities.tolist()
        )

        calibrated_all_predictions.extend(
            calibrated_predictions.tolist()
        )

        # -------------------------------------------------
        # Fold result
        # -------------------------------------------------

        result = {
            "fold": fold_number,

            "train_size": len(
                X_train
            ),

            "test_size": len(
                X_test
            ),

            "raw_accuracy": raw_metrics[
                "accuracy"
            ],

            "calibrated_accuracy": (
                calibrated_metrics[
                    "accuracy"
                ]
            ),

            "raw_brier": raw_metrics[
                "brier_score"
            ],

            "calibrated_brier": (
                calibrated_metrics[
                    "brier_score"
                ]
            ),

            "raw_log_loss": raw_metrics[
                "log_loss"
            ],

            "calibrated_log_loss": (
                calibrated_metrics[
                    "log_loss"
                ]
            ),

            "raw_auc": raw_metrics[
                "roc_auc"
            ],

            "calibrated_auc": (
                calibrated_metrics[
                    "roc_auc"
                ]
            ),
        }

        fold_results.append(
            result
        )

        print(
            f"  Raw Accuracy: "
            f"{raw_metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"  Calibrated Accuracy: "
            f"{calibrated_metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"  Raw Brier: "
            f"{raw_metrics['brier_score']:.4f}"
        )

        print(
            f"  Calibrated Brier: "
            f"{calibrated_metrics['brier_score']:.4f}"
        )

        print(
            f"  Raw Log Loss: "
            f"{raw_metrics['log_loss']:.4f}"
        )

        print(
            f"  Calibrated Log Loss: "
            f"{calibrated_metrics['log_loss']:.4f}"
        )

    # -----------------------------------------------------
    # Global metrics
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("OVERALL RESULTS")
    print("=" * 75)

    raw_global = calculate_metrics(
        np.array(raw_all_y),
        np.array(raw_all_predictions),
        np.array(raw_all_probabilities),
    )

    calibrated_global = calculate_metrics(
        np.array(calibrated_all_y),
        np.array(calibrated_all_predictions),
        np.array(
            calibrated_all_probabilities
        ),
    )

    print("\nRAW EXTRA TREES")
    print("-" * 40)

    print(
        f"Accuracy: "
        f"{raw_global['accuracy'] * 100:.2f}%"
    )

    print(
        f"Brier Score: "
        f"{raw_global['brier_score']:.4f}"
    )

    print(
        f"Log Loss: "
        f"{raw_global['log_loss']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{raw_global['roc_auc']:.4f}"
    )

    print("\nCALIBRATED EXTRA TREES")
    print("-" * 40)

    print(
        f"Accuracy: "
        f"{calibrated_global['accuracy'] * 100:.2f}%"
    )

    print(
        f"Brier Score: "
        f"{calibrated_global['brier_score']:.4f}"
    )

    print(
        f"Log Loss: "
        f"{calibrated_global['log_loss']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{calibrated_global['roc_auc']:.4f}"
    )

    # -----------------------------------------------------
    # Improvements
    # -----------------------------------------------------

    brier_improvement = (
        raw_global["brier_score"]
        - calibrated_global["brier_score"]
    )

    logloss_improvement = (
        raw_global["log_loss"]
        - calibrated_global["log_loss"]
    )

    accuracy_difference = (
        calibrated_global["accuracy"]
        - raw_global["accuracy"]
    )

    print("\n" + "=" * 75)
    print("CALIBRATION COMPARISON")
    print("=" * 75)

    print(
        f"Brier improvement: "
        f"{brier_improvement:+.4f}"
    )

    print(
        f"Log-loss improvement: "
        f"{logloss_improvement:+.4f}"
    )

    print(
        f"Accuracy difference: "
        f"{accuracy_difference * 100:+.2f} pp"
    )

    # -----------------------------------------------------
    # Decision
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("CALIBRATION DECISION")
    print("=" * 75)

    if (
        calibrated_global["brier_score"]
        < raw_global["brier_score"]
        and
        calibrated_global["log_loss"]
        < raw_global["log_loss"]
    ):

        decision = (
            "CALIBRATION IMPROVES PROBABILITY QUALITY"
        )

        print(
            "PASS: Calibrated probabilities have "
            "better Brier score and log loss."
        )

    else:

        decision = (
            "RAW PROBABILITIES PREFERRED"
        )

        print(
            "KEEP: Calibration did not improve "
            "both probability metrics."
        )

    # -----------------------------------------------------
    # Save report
    # -----------------------------------------------------

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_df = pd.DataFrame(
        fold_results
    )

    report_df.to_csv(
        REPORT_PATH,
        index=False,
    )

    print("\n" + "=" * 75)
    print("REPORT SAVED")
    print("=" * 75)

    print(
        REPORT_PATH.resolve()
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("FINAL SUMMARY")
    print("=" * 75)

    print(
        f"Raw Brier: "
        f"{raw_global['brier_score']:.4f}"
    )

    print(
        f"Calibrated Brier: "
        f"{calibrated_global['brier_score']:.4f}"
    )

    print(
        f"Raw Log Loss: "
        f"{raw_global['log_loss']:.4f}"
    )

    print(
        f"Calibrated Log Loss: "
        f"{calibrated_global['log_loss']:.4f}"
    )

    print(
        f"Decision: {decision}"
    )

    print("\n" + "=" * 75)
    print(
        "Probability calibration test complete."
    )
    print("=" * 75)


if __name__ == "__main__":
    main()
