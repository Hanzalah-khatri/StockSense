"""
StockSense - Milestone 4, Step 5
Final Classifier Confirmation

Purpose:
Confirm the selected Extra Trees configuration using
a stricter expanding walk-forward validation.

This script DOES NOT modify the production model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from services.market_data import fetch_stock_data
from services.prediction_service import PredictionService
from services.validation_service import ValidationService


# =========================================================
# CONFIGURATION
# =========================================================

TICKER = "AAPL"

# Stricter validation
N_SPLITS = 7
MIN_TRAIN_SIZE = 150

# Winning Top-5 feature set
FEATURES = [
    "Return_Lag_1",
    "Return_Lag_3",
    "Return_Lag_5",
    "Volatility",
    "Volume_Change",
]

# Winning Extra Trees configuration
MODEL_CONFIG = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_split": 10,
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced",
}


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    print("=" * 75)
    print("LOADING MARKET DATA")
    print("=" * 75)

    market_data = fetch_stock_data(
        TICKER,
        period="2y"
    )

    print(
        f"Loaded {len(market_data)} raw rows."
    )

    prediction_service = PredictionService()

    features = prediction_service.prepare_features(
        market_data
    )

    validation_service = ValidationService()

    X, y = validation_service.prepare_data(
        features
    )

    X = X[FEATURES].copy()

    print(
        f"Usable observations: {len(X)}"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    print("\nSelected features:")
    for feature in FEATURES:
        print(f"  - {feature}")

    return X, y


# =========================================================
# CREATE STRICTER WALK-FORWARD FOLDS
# =========================================================

def create_folds(
    total_rows,
    n_splits,
    min_train_size
):

    remaining_rows = (
        total_rows - min_train_size
    )

    test_size = (
        remaining_rows // n_splits
    )

    folds = []

    for fold in range(n_splits):

        train_end = (
            min_train_size
            + fold * test_size
        )

        if fold == n_splits - 1:

            test_end = total_rows

        else:

            test_end = (
                train_end + test_size
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

        folds.append(
            (
                fold + 1,
                train_indices,
                test_indices,
            )
        )

    return folds


# =========================================================
# BASELINE
# =========================================================

def calculate_baseline(
    y,
    folds
):

    test_indices = np.concatenate(
        [
            test_indices
            for _, _, test_indices in folds
        ]
    )

    actual = y.iloc[test_indices]

    majority_class = (
        actual.value_counts()
        .idxmax()
    )

    predictions = np.full(
        len(actual),
        majority_class
    )

    accuracy = accuracy_score(
        actual,
        predictions
    )

    return {
        "accuracy": accuracy,
        "majority_class": int(
            majority_class
        ),
        "observations": len(actual),
    }


# =========================================================
# FINAL VALIDATION
# =========================================================

def validate_model(
    X,
    y,
    folds
):

    actual_all = []
    predicted_all = []
    probability_all = []

    fold_results = []

    for (
        fold_number,
        train_indices,
        test_indices,
    ) in folds:

        X_train = X.iloc[
            train_indices
        ]

        X_test = X.iloc[
            test_indices
        ]

        y_train = y.iloc[
            train_indices
        ]

        y_test = y.iloc[
            test_indices
        ]

        model = ExtraTreesClassifier(
            **MODEL_CONFIG
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        actual_all.extend(
            y_test.tolist()
        )

        predicted_all.extend(
            predictions.tolist()
        )

        probability_all.extend(
            probabilities.tolist()
        )

        fold_accuracy = accuracy_score(
            y_test,
            predictions
        )

        fold_precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        fold_recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        fold_f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        try:

            fold_auc = roc_auc_score(
                y_test,
                probabilities
            )

        except ValueError:

            fold_auc = np.nan

        fold_results.append(
            {
                "fold": fold_number,
                "train_size": len(train_indices),
                "test_size": len(test_indices),
                "accuracy": fold_accuracy,
                "precision": fold_precision,
                "recall": fold_recall,
                "f1": fold_f1,
                "roc_auc": fold_auc,
            }
        )

        print(
            f"Fold {fold_number}: "
            f"Train={len(train_indices)} "
            f"Test={len(test_indices)} "
            f"Accuracy={fold_accuracy * 100:.2f}% "
            f"F1={fold_f1 * 100:.2f}% "
            f"ROC-AUC={fold_auc:.4f}"
        )

    # =====================================================
    # AGGREGATED METRICS
    # =====================================================

    accuracy = accuracy_score(
        actual_all,
        predicted_all
    )

    precision = precision_score(
        actual_all,
        predicted_all,
        zero_division=0
    )

    recall = recall_score(
        actual_all,
        predicted_all,
        zero_division=0
    )

    f1 = f1_score(
        actual_all,
        predicted_all,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        actual_all,
        probability_all
    )

    fold_df = pd.DataFrame(
        fold_results
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "fold_mean": fold_df[
            "accuracy"
        ].mean(),
        "fold_std": fold_df[
            "accuracy"
        ].std(ddof=0),
        "fold_min": fold_df[
            "accuracy"
        ].min(),
        "fold_max": fold_df[
            "accuracy"
        ].max(),
        "folds": fold_results,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    X, y = load_data()

    folds = create_folds(
        total_rows=len(X),
        n_splits=N_SPLITS,
        min_train_size=MIN_TRAIN_SIZE,
    )

    print("\n" + "=" * 75)
    print("STRICT WALK-FORWARD VALIDATION")
    print("=" * 75)

    for (
        fold_number,
        train_indices,
        test_indices,
    ) in folds:

        print(
            f"Fold {fold_number}: "
            f"Train={len(train_indices)} "
            f"Test={len(test_indices)}"
        )

    baseline = calculate_baseline(
        y,
        folds
    )

    print("\n" + "=" * 75)
    print("MAJORITY BASELINE")
    print("=" * 75)

    print(
        f"Accuracy: "
        f"{baseline['accuracy'] * 100:.2f}%"
    )

    print(
        f"Majority class: "
        f"{'UP' if baseline['majority_class'] == 1 else 'DOWN'}"
    )

    print(
        f"Observations: "
        f"{baseline['observations']}"
    )

    # =====================================================
    # MODEL
    # =====================================================

    print("\n" + "=" * 75)
    print("FINAL MODEL")
    print("=" * 75)

    print("Extra Trees Classifier")

    for key, value in MODEL_CONFIG.items():
        print(
            f"{key}: {value}"
        )

    print("\n" + "=" * 75)
    print("RUNNING FINAL CONFIRMATION")
    print("=" * 75)

    result = validate_model(
        X,
        y,
        folds
    )

    improvement = (
        result["accuracy"]
        - baseline["accuracy"]
    )

    # =====================================================
    # FINAL RESULT
    # =====================================================

    print("\n" + "=" * 75)
    print("FINAL CONFIRMATION RESULT")
    print("=" * 75)

    print(
        f"Accuracy:  "
        f"{result['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{result['precision'] * 100:.2f}%"
    )

    print(
        f"Recall:    "
        f"{result['recall'] * 100:.2f}%"
    )

    print(
        f"F1:        "
        f"{result['f1'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC:   "
        f"{result['roc_auc']:.4f}"
    )

    print(
        f"Baseline:  "
        f"{baseline['accuracy'] * 100:.2f}%"
    )

    print(
        f"vs Baseline: "
        f"{improvement * 100:+.2f} pp"
    )

    print(
        f"Fold mean: "
        f"{result['fold_mean'] * 100:.2f}%"
    )

    print(
        f"Fold std: "
        f"{result['fold_std'] * 100:.2f} pp"
    )

    print(
        f"Fold range: "
        f"{result['fold_min'] * 100:.2f}% - "
        f"{result['fold_max'] * 100:.2f}%"
    )

    # =====================================================
    # DECISION
    # =====================================================

    print("\n" + "=" * 75)
    print("MODEL DECISION")
    print("=" * 75)

    if (
        result["accuracy"]
        > baseline["accuracy"]
        and result["roc_auc"]
        > 0.50
    ):

        print(
            "PASS: Model beats the majority "
            "baseline and ROC-AUC is above 0.50."
        )

        print(
            "Extra Trees is a valid "
            "production candidate."
        )

    else:

        print(
            "CAUTION: Model does not provide "
            "strong enough confirmation."
        )

        print(
            "Keep the current production model "
            "until further testing."
        )

    # =====================================================
    # SAVE
    # =====================================================

    reports_dir = Path("reports")

    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        reports_dir
        / "final_classifier_confirmation.csv"
    )

    summary = pd.DataFrame(
        [
            {
                "model": "Extra Trees",
                "features": len(FEATURES),
                "n_estimators": MODEL_CONFIG[
                    "n_estimators"
                ],
                "max_depth": MODEL_CONFIG[
                    "max_depth"
                ],
                "min_samples_split": MODEL_CONFIG[
                    "min_samples_split"
                ],
                "accuracy": result[
                    "accuracy"
                ],
                "precision": result[
                    "precision"
                ],
                "recall": result[
                    "recall"
                ],
                "f1": result[
                    "f1"
                ],
                "roc_auc": result[
                    "roc_auc"
                ],
                "baseline_accuracy": baseline[
                    "accuracy"
                ],
                "vs_baseline": improvement,
                "fold_mean_accuracy": result[
                    "fold_mean"
                ],
                "fold_std_accuracy": result[
                    "fold_std"
                ],
                "fold_min_accuracy": result[
                    "fold_min"
                ],
                "fold_max_accuracy": result[
                    "fold_max"
                ],
            }
        ]
    )

    summary.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 75)
    print("RESULTS SAVED")
    print("=" * 75)

    print(
        output_path.resolve()
    )

    print("\n" + "=" * 75)
    print("Final classifier confirmation complete.")
    print("=" * 75)


if __name__ == "__main__":
    main()