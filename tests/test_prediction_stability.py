"""
StockSense — Milestone 3.16
Prediction Stability Test

Purpose:
Test whether the final tuned Random Forest remains reasonably
stable across different chronological market periods.

We divide the available historical data into multiple sequential
train/test periods.

The model is NEVER allowed to train on future observations.

Model:
- Tuned Random Forest
- All 30 features

Metrics:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- UP prediction percentage
- DOWN prediction percentage

The goal is NOT to find another model.

The goal is to determine whether the current model is stable
enough to be used in StockSense.
"""

import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from services.market_data import fetch_stock_data
from services.prediction_service import PredictionService


# =========================================================
# CONFIGURATION
# =========================================================

TICKER = "AAPL"

# Same tuned Random Forest used throughout Milestones 3.10+
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1,
}


# =========================================================
# STABILITY PERIODS
# =========================================================
#
# Each period is evaluated chronologically.
#
# Example:
#
# Period 1:
# Train  -> rows 0:100
# Test   -> rows 100:140
#
# Period 2:
# Train  -> rows 0:140
# Test   -> rows 140:180
#
# etc.
#
# This simulates the model being retrained as new market
# data becomes available.
# =========================================================

PERIODS = [
    {
        "name": "Period 1",
        "train_end": 100,
        "test_end": 140,
    },
    {
        "name": "Period 2",
        "train_end": 140,
        "test_end": 180,
    },
    {
        "name": "Period 3",
        "train_end": 180,
        "test_end": 220,
    },
    {
        "name": "Period 4",
        "train_end": 220,
        "test_end": 260,
    },
    {
        "name": "Period 5",
        "train_end": 260,
        "test_end": 301,
    },
]


warnings.filterwarnings("ignore")


# =========================================================
# HELPERS
# =========================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_subheader(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


def create_model():
    """
    Create the tuned Random Forest.
    """

    return RandomForestClassifier(
        **RF_PARAMS
    )


def calculate_metrics(
    model,
    X_test,
    y_test,
):
    """
    Calculate all stability metrics.
    """

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    if len(np.unique(y_test)) >= 2:

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
        )

    else:

        roc_auc = np.nan

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    up_predictions = int(
        np.sum(predictions == 1)
    )

    down_predictions = int(
        np.sum(predictions == 0)
    )

    up_prediction_pct = (
        up_predictions
        / len(predictions)
    )

    down_prediction_pct = (
        down_predictions
        / len(predictions)
    )

    actual_up_pct = (
        np.sum(y_test == 1)
        / len(y_test)
    )

    actual_down_pct = (
        np.sum(y_test == 0)
        / len(y_test)
    )

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": matrix,
        "up_prediction_pct": up_prediction_pct,
        "down_prediction_pct": down_prediction_pct,
        "actual_up_pct": actual_up_pct,
        "actual_down_pct": actual_down_pct,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print_header(
        "STOCKSENSE — MILESTONE 3.16"
    )

    print(
        "PREDICTION STABILITY TEST"
    )

    print()
    print(
        f"Ticker: {TICKER}"
    )

    print(
        "Model: Tuned Random Forest"
    )

    print(
        "Features: All 30 features"
    )

    print(
        "Validation: Chronological stability testing"
    )


    # =====================================================
    # 1. LOAD MARKET DATA
    # =====================================================

    print_header(
        "1. LOADING MARKET DATA"
    )

    market_data = fetch_stock_data(
        TICKER,
        period="2y",
    )

    print(
        f"Loaded {len(market_data)} market rows"
    )


    # =====================================================
    # 2. PREPARE FEATURES
    # =====================================================

    print_header(
        "2. PREPARING FEATURES"
    )

    prediction_service = PredictionService()

    features = prediction_service.prepare_features(
        market_data
    )

    X, y = prediction_service.prepare_ml_data(
        features
    )

    # -----------------------------------------------------
    # Convert next-day return into direction
    #
    # UP   = 1
    # DOWN = 0
    # -----------------------------------------------------

    y = (y > 0).astype(int)

    feature_columns = (
        prediction_service.get_feature_columns()
    )

    X = X[feature_columns]

    print(
        f"Usable rows   : {len(X)}"
    )

    print(
        f"Total features: {len(feature_columns)}"
    )

    print()
    print(
        "Target distribution:"
    )

    print(
        pd.Series(y)
        .map({
            0: "DOWN",
            1: "UP",
        })
        .value_counts()
    )


    # =====================================================
    # 3. VERIFY PERIOD CONFIGURATION
    # =====================================================

    print_header(
        "3. STABILITY PERIOD CONFIGURATION"
    )

    total_rows = len(X)

    for period in PERIODS:

        train_end = period[
            "train_end"
        ]

        test_end = period[
            "test_end"
        ]

        if test_end > total_rows:

            raise ValueError(
                f"{period['name']} requires "
                f"{test_end} rows, but only "
                f"{total_rows} are available."
            )

        if train_end >= test_end:

            raise ValueError(
                f"Invalid period configuration: "
                f"{period['name']}"
            )

        print()

        print(
            f"{period['name']}:"
        )

        print(
            f"  Train rows: 0 → {train_end - 1}"
        )

        print(
            f"  Test rows : {train_end} → {test_end - 1}"
        )

        print(
            f"  Train size: {train_end}"
        )

        print(
            f"  Test size : {test_end - train_end}"
        )


    # =====================================================
    # 4. WALK THROUGH PERIODS
    # =====================================================

    print_header(
        "4. RUNNING STABILITY TEST"
    )

    all_results = []

    for period in PERIODS:

        name = period[
            "name"
        ]

        train_end = period[
            "train_end"
        ]

        test_end = period[
            "test_end"
        ]

        print_subheader(
            name
        )

        # -------------------------------------------------
        # Chronological split
        # -------------------------------------------------

        X_train = X.iloc[
            :train_end
        ]

        y_train = y.iloc[
            :train_end
        ]

        X_test = X.iloc[
            train_end:test_end
        ]

        y_test = y.iloc[
            train_end:test_end
        ]

        print(
            f"Training rows : {len(X_train)}"
        )

        print(
            f"Testing rows  : {len(X_test)}"
        )

        print()

        # -------------------------------------------------
        # Target distribution
        # -------------------------------------------------

        print(
            f"Actual UP   : "
            f"{np.sum(y_test == 1)} "
            f"({np.mean(y_test == 1):.2%})"
        )

        print(
            f"Actual DOWN : "
            f"{np.sum(y_test == 0)} "
            f"({np.mean(y_test == 0):.2%})"
        )

        # -------------------------------------------------
        # Train model
        # -------------------------------------------------

        model = create_model()

        model.fit(
            X_train,
            y_train,
        )

        # -------------------------------------------------
        # Evaluate
        # -------------------------------------------------

        metrics = calculate_metrics(
            model,
            X_test,
            y_test,
        )

        # -------------------------------------------------
        # Display
        # -------------------------------------------------

        print()

        print(
            f"Accuracy  : "
            f"{metrics['accuracy']:.2%}"
        )

        print(
            f"Precision : "
            f"{metrics['precision']:.2%}"
        )

        print(
            f"Recall    : "
            f"{metrics['recall']:.2%}"
        )

        print(
            f"F1        : "
            f"{metrics['f1']:.4f}"
        )

        if np.isnan(
            metrics["roc_auc"]
        ):

            print(
                "ROC-AUC   : N/A"
            )

        else:

            print(
                f"ROC-AUC   : "
                f"{metrics['roc_auc']:.4f}"
            )

        print()

        print(
            f"Predicted UP   : "
            f"{metrics['up_prediction_pct']:.2%}"
        )

        print(
            f"Predicted DOWN : "
            f"{metrics['down_prediction_pct']:.2%}"
        )

        print()

        print(
            "Confusion Matrix:"
        )

        print(
            metrics["confusion_matrix"]
        )

        # -------------------------------------------------
        # Store results
        # -------------------------------------------------

        all_results.append(
            {
                "Period": name,
                "Train_Size": len(X_train),
                "Test_Size": len(X_test),
                "Accuracy": metrics[
                    "accuracy"
                ],
                "Precision": metrics[
                    "precision"
                ],
                "Recall": metrics[
                    "recall"
                ],
                "F1": metrics[
                    "f1"
                ],
                "ROC_AUC": metrics[
                    "roc_auc"
                ],
                "Actual_UP": metrics[
                    "actual_up_pct"
                ],
                "Predicted_UP": metrics[
                    "up_prediction_pct"
                ],
            }
        )


    # =====================================================
    # 5. RESULTS TABLE
    # =====================================================

    print_header(
        "5. STABILITY RESULTS"
    )

    results_df = pd.DataFrame(
        all_results
    )

    display_df = results_df.copy()

    percentage_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Actual_UP",
        "Predicted_UP",
    ]

    for column in percentage_columns:

        display_df[column] *= 100

    print()

    print(
        display_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.2f}%".format,
                "Precision": "{:.2f}%".format,
                "Recall": "{:.2f}%".format,
                "F1": "{:.2f}%".format,
                "ROC_AUC": "{:.4f}".format,
                "Actual_UP": "{:.2f}%".format,
                "Predicted_UP": "{:.2f}%".format,
            },
        )
    )


    # =====================================================
    # 6. AVERAGE PERFORMANCE
    # =====================================================

    print_header(
        "6. AVERAGE STABILITY PERFORMANCE"
    )

    avg_accuracy = results_df[
        "Accuracy"
    ].mean()

    std_accuracy = results_df[
        "Accuracy"
    ].std()

    avg_precision = results_df[
        "Precision"
    ].mean()

    avg_recall = results_df[
        "Recall"
    ].mean()

    avg_f1 = results_df[
        "F1"
    ].mean()

    std_f1 = results_df[
        "F1"
    ].std()

    avg_auc = results_df[
        "ROC_AUC"
    ].mean()

    std_auc = results_df[
        "ROC_AUC"
    ].std()

    avg_predicted_up = results_df[
        "Predicted_UP"
    ].mean()

    print(
        f"Average Accuracy : "
        f"{avg_accuracy:.2%}"
    )

    print(
        f"Accuracy Std     : "
        f"{std_accuracy:.2%}"
    )

    print(
        f"Average Precision: "
        f"{avg_precision:.2%}"
    )

    print(
        f"Average Recall   : "
        f"{avg_recall:.2%}"
    )

    print(
        f"Average F1       : "
        f"{avg_f1:.4f}"
    )

    print(
        f"F1 Std           : "
        f"{std_f1:.4f}"
    )

    print(
        f"Average ROC-AUC  : "
        f"{avg_auc:.4f}"
    )

    print(
        f"ROC-AUC Std      : "
        f"{std_auc:.4f}"
    )

    print(
        f"Average UP prediction: "
        f"{avg_predicted_up:.2%}"
    )


    # =====================================================
    # 7. BEST / WORST PERIOD
    # =====================================================

    print_header(
        "7. BEST AND WORST PERIOD"
    )

    best_accuracy_row = results_df.loc[
        results_df["Accuracy"].idxmax()
    ]

    worst_accuracy_row = results_df.loc[
        results_df["Accuracy"].idxmin()
    ]

    best_auc_row = results_df.loc[
        results_df["ROC_AUC"].idxmax()
    ]

    worst_auc_row = results_df.loc[
        results_df["ROC_AUC"].idxmin()
    ]

    print()

    print(
        f"Best Accuracy:"
    )

    print(
        f"  {best_accuracy_row['Period']} "
        f"→ {best_accuracy_row['Accuracy']:.2%}"
    )

    print()

    print(
        f"Worst Accuracy:"
    )

    print(
        f"  {worst_accuracy_row['Period']} "
        f"→ {worst_accuracy_row['Accuracy']:.2%}"
    )

    print()

    print(
        f"Best ROC-AUC:"
    )

    print(
        f"  {best_auc_row['Period']} "
        f"→ {best_auc_row['ROC_AUC']:.4f}"
    )

    print()

    print(
        f"Worst ROC-AUC:"
    )

    print(
        f"  {worst_auc_row['Period']} "
        f"→ {worst_auc_row['ROC_AUC']:.4f}"
    )


    # =====================================================
    # 8. CONSISTENCY CHECK
    # =====================================================

    print_header(
        "8. CONSISTENCY CHECK"
    )

    accuracy_above_50 = int(
        np.sum(
            results_df["Accuracy"] > 0.50
        )
    )

    auc_above_050 = int(
        np.sum(
            results_df["ROC_AUC"] > 0.50
        )
    )

    total_periods = len(
        results_df
    )

    print()

    print(
        f"Periods with Accuracy > 50%: "
        f"{accuracy_above_50}/{total_periods}"
    )

    print(
        f"Periods with ROC-AUC > 0.50: "
        f"{auc_above_050}/{total_periods}"
    )

    print()

    # -----------------------------------------------------
    # Prediction bias
    # -----------------------------------------------------

    prediction_bias = (
        abs(
            results_df["Predicted_UP"]
            - results_df["Actual_UP"]
        )
    )

    average_prediction_bias = (
        prediction_bias.mean()
    )

    print(
        f"Average UP prediction bias: "
        f"{average_prediction_bias:.2%}"
    )


    # =====================================================
    # 9. STABILITY VERDICT
    # =====================================================

    print_header(
        "9. STABILITY VERDICT"
    )

    print()

    # -----------------------------------------------------
    # Conservative interpretation
    #
    # We do NOT demand high accuracy.
    #
    # We want:
    # - average accuracy above baseline territory
    # - average AUC above 0.50
    # - reasonable consistency
    # - no extreme prediction bias
    # -----------------------------------------------------

    stable_accuracy = (
        avg_accuracy >= 0.53
    )

    stable_auc = (
        avg_auc >= 0.55
    )

    reasonably_consistent = (
        accuracy_above_50
        >=
        max(
            3,
            int(total_periods * 0.60)
        )
    )

    reasonable_prediction_bias = (
        average_prediction_bias
        <= 0.20
    )

    stability_score = sum(
        [
            stable_accuracy,
            stable_auc,
            reasonably_consistent,
            reasonable_prediction_bias,
        ]
    )

    print(
        f"Accuracy criterion : "
        f"{'PASS' if stable_accuracy else 'FAIL'}"
    )

    print(
        f"ROC-AUC criterion  : "
        f"{'PASS' if stable_auc else 'FAIL'}"
    )

    print(
        f"Consistency        : "
        f"{'PASS' if reasonably_consistent else 'FAIL'}"
    )

    print(
        f"Prediction balance : "
        f"{'PASS' if reasonable_prediction_bias else 'FAIL'}"
    )

    print()

    print(
        f"Stability score: "
        f"{stability_score}/4"
    )

    print()

    if stability_score >= 3:

        verdict = (
            "REASONABLY STABLE"
        )

    elif stability_score >= 2:

        verdict = (
            "MODERATELY STABLE"
        )

    else:

        verdict = (
            "UNSTABLE"
        )

    print(
        f"🏆 VERDICT: {verdict}"
    )


    # =====================================================
    # 10. SAVE RESULTS
    # =====================================================

    output_path = (
        "tests/prediction_stability_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()

    print(
        f"Results saved to: "
        f"{output_path}"
    )


    # =====================================================
    # 11. FINAL CONCLUSION
    # =====================================================

    print_header(
        "11. MILESTONE 3.16 CONCLUSION"
    )

    print()

    print(
        "Model tested:"
    )

    print(
        "Tuned Random Forest + 30 features"
    )

    print()

    print(
        f"Average Accuracy: "
        f"{avg_accuracy:.2%}"
    )

    print(
        f"Average ROC-AUC: "
        f"{avg_auc:.4f}"
    )

    print(
        f"Accuracy Std: "
        f"{std_accuracy:.2%}"
    )

    print(
        f"ROC-AUC Std: "
        f"{std_auc:.4f}"
    )

    print()

    print(
        f"Stability Verdict: "
        f"{verdict}"
    )

    print()

    print(
        "NEXT STEP:"
    )

    print(
        "Milestone 3.17 — Final Model Lock & Save"
    )

    print(
        "After 3.17, the ML testing phase will be "
        "considered complete."
    )

    print()

    print("=" * 70)

    print(
        "Milestone 3.16 complete."
    )

    print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()