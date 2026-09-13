"""
StockSense — Milestone 3.15
Final Holdout Validation

Purpose:
Evaluate the selected 5-feature Random Forest on the completely
untouched final holdout and compare it against:

1. 5-feature Random Forest
2. Full 30-feature Random Forest
3. Majority-class baseline

The final 61 rows are used ONLY here.

Selected 5 features from Milestone 3.14:
- Return_Lag_1
- Volume_Change
- Volatility
- MACD_Histogram
- Momentum_20
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

TRAIN_SIZE = 240

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1,
}


# =========================================================
# SELECTED FEATURES FROM MILESTONE 3.14
# =========================================================

SELECTED_FEATURES = [
    "Return_Lag_1",
    "Volume_Change",
    "Volatility",
    "MACD_Histogram",
    "Momentum_20",
]


warnings.filterwarnings("ignore")


# =========================================================
# HELPERS
# =========================================================

def create_model():
    """Create the tuned Random Forest classifier."""
    return RandomForestClassifier(**RF_PARAMS)


def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def calculate_metrics(model, X_test, y_test):
    """
    Calculate all final holdout metrics.
    """

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

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

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": matrix,
    }


def print_model_results(
    name,
    results,
):
    """
    Print metrics for one model.
    """

    print()
    print(f"MODEL: {name}")

    print(
        f"Accuracy  : {results['accuracy']:.2%}"
    )

    print(
        f"Precision : {results['precision']:.2%}"
    )

    print(
        f"Recall    : {results['recall']:.2%}"
    )

    print(
        f"F1 Score  : {results['f1']:.4f}"
    )

    if np.isnan(results["roc_auc"]):
        print(
            "ROC-AUC   : N/A"
        )
    else:
        print(
            f"ROC-AUC   : {results['roc_auc']:.4f}"
        )

    print()

    print("Confusion Matrix:")

    print(
        results["confusion_matrix"]
    )

    print()

    predictions = results["predictions"]

    up_count = int(
        np.sum(predictions == 1)
    )

    down_count = int(
        np.sum(predictions == 0)
    )

    print(
        f"Predicted UP   : "
        f"{up_count} "
        f"({up_count / len(predictions):.2%})"
    )

    print(
        f"Predicted DOWN : "
        f"{down_count} "
        f"({down_count / len(predictions):.2%})"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print_header(
        "STOCKSENSE — MILESTONE 3.15"
    )

    print(
        "FINAL HOLDOUT VALIDATION"
    )

    print()
    print(
        f"Ticker: {TICKER}"
    )

    print(
        f"Training rows: {TRAIN_SIZE}"
    )

    print(
        "Final holdout: remaining rows"
    )

    print(
        "Model: Tuned Random Forest"
    )


    # =====================================================
    # 1. LOAD DATA
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


    # =====================================================
    # 3. CHRONOLOGICAL SPLIT
    # =====================================================

    print_header(
        "3. FINAL CHRONOLOGICAL SPLIT"
    )

    X_train = X.iloc[
        :TRAIN_SIZE
    ].copy()

    y_train = y.iloc[
        :TRAIN_SIZE
    ].copy()

    X_holdout = X.iloc[
        TRAIN_SIZE:
    ].copy()

    y_holdout = y.iloc[
        TRAIN_SIZE:
    ].copy()

    print(
        f"Training rows : {len(X_train)}"
    )

    print(
        f"Holdout rows  : {len(X_holdout)}"
    )

    print()

    print(
        "⚠️ IMPORTANT:"
    )

    print(
        "These holdout rows have not been used "
        "for feature selection or model validation."
    )


    # =====================================================
    # 4. HOLDOUT TARGET DISTRIBUTION
    # =====================================================

    print_header(
        "4. HOLDOUT TARGET DISTRIBUTION"
    )

    holdout_up = int(
        np.sum(y_holdout == 1)
    )

    holdout_down = int(
        np.sum(y_holdout == 0)
    )

    print(
        f"UP   : {holdout_up}"
    )

    print(
        f"DOWN : {holdout_down}"
    )

    print(
        f"UP percentage: "
        f"{holdout_up / len(y_holdout):.2%}"
    )

    print(
        f"DOWN percentage: "
        f"{holdout_down / len(y_holdout):.2%}"
    )


    # =====================================================
    # 5. FEATURE VALIDATION
    # =====================================================

    print_header(
        "5. SELECTED FEATURE SET"
    )

    print(
        f"Using {len(SELECTED_FEATURES)} features:"
    )

    for i, feature in enumerate(
        SELECTED_FEATURES,
        start=1,
    ):
        print(
            f"{i}. {feature}"
        )

    # Make sure every selected feature exists
    missing_features = [
        feature
        for feature in SELECTED_FEATURES
        if feature not in X.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing selected features: "
            + ", ".join(missing_features)
        )


    # =====================================================
    # 6. TRAIN 5-FEATURE MODEL
    # =====================================================

    print_header(
        "6. TRAINING 5-FEATURE RANDOM FOREST"
    )

    five_feature_model = create_model()

    five_feature_model.fit(
        X_train[SELECTED_FEATURES],
        y_train,
    )

    print(
        "5-feature model trained successfully."
    )


    # =====================================================
    # 7. TEST 5-FEATURE MODEL
    # =====================================================

    print_header(
        "7. 5-FEATURE HOLDOUT RESULTS"
    )

    five_feature_results = calculate_metrics(
        five_feature_model,
        X_holdout[SELECTED_FEATURES],
        y_holdout,
    )

    print_model_results(
        "5-Feature Random Forest",
        five_feature_results,
    )


    # =====================================================
    # 8. TRAIN FULL 30-FEATURE MODEL
    # =====================================================

    print_header(
        "8. TRAINING FULL 30-FEATURE RANDOM FOREST"
    )

    full_feature_model = create_model()

    full_feature_model.fit(
        X_train[feature_columns],
        y_train,
    )

    print(
        "Full 30-feature model trained successfully."
    )


    # =====================================================
    # 9. TEST FULL 30-FEATURE MODEL
    # =====================================================

    print_header(
        "9. FULL 30-FEATURE HOLDOUT RESULTS"
    )

    full_feature_results = calculate_metrics(
        full_feature_model,
        X_holdout[feature_columns],
        y_holdout,
    )

    print_model_results(
        "Full 30-Feature Random Forest",
        full_feature_results,
    )


    # =====================================================
    # 10. MAJORITY BASELINE
    # =====================================================

    print_header(
        "10. MAJORITY BASELINE"
    )

    majority_class = (
        y_train.value_counts()
        .idxmax()
    )

    baseline_predictions = np.full(
        len(y_holdout),
        majority_class,
    )

    baseline_accuracy = accuracy_score(
        y_holdout,
        baseline_predictions,
    )

    baseline_precision = precision_score(
        y_holdout,
        baseline_predictions,
        zero_division=0,
    )

    baseline_recall = recall_score(
        y_holdout,
        baseline_predictions,
        zero_division=0,
    )

    baseline_f1 = f1_score(
        y_holdout,
        baseline_predictions,
        zero_division=0,
    )

    baseline_matrix = confusion_matrix(
        y_holdout,
        baseline_predictions,
    )

    print(
        f"Majority class: "
        f"{'UP' if majority_class == 1 else 'DOWN'}"
    )

    print(
        f"Accuracy  : {baseline_accuracy:.2%}"
    )

    print(
        f"Precision : {baseline_precision:.2%}"
    )

    print(
        f"Recall    : {baseline_recall:.2%}"
    )

    print(
        f"F1 Score  : {baseline_f1:.4f}"
    )

    print()

    print(
        "Confusion Matrix:"
    )

    print(
        baseline_matrix
    )


    # =====================================================
    # 11. HEAD-TO-HEAD COMPARISON
    # =====================================================

    print_header(
        "11. FINAL HEAD-TO-HEAD COMPARISON"
    )

    comparison = pd.DataFrame(
        [
            {
                "Model": "5-Feature RF",
                "Features": 5,
                "Accuracy": five_feature_results[
                    "accuracy"
                ],
                "Precision": five_feature_results[
                    "precision"
                ],
                "Recall": five_feature_results[
                    "recall"
                ],
                "F1": five_feature_results[
                    "f1"
                ],
                "ROC_AUC": five_feature_results[
                    "roc_auc"
                ],
            },
            {
                "Model": "Full 30-Feature RF",
                "Features": 30,
                "Accuracy": full_feature_results[
                    "accuracy"
                ],
                "Precision": full_feature_results[
                    "precision"
                ],
                "Recall": full_feature_results[
                    "recall"
                ],
                "F1": full_feature_results[
                    "f1"
                ],
                "ROC_AUC": full_feature_results[
                    "roc_auc"
                ],
            },
            {
                "Model": "Majority Baseline",
                "Features": 0,
                "Accuracy": baseline_accuracy,
                "Precision": baseline_precision,
                "Recall": baseline_recall,
                "F1": baseline_f1,
                "ROC_AUC": 0.50,
            },
        ]
    )

    print()

    print(
        comparison.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.2%}".format,
                "Precision": "{:.2%}".format,
                "Recall": "{:.2%}".format,
                "F1": "{:.4f}".format,
                "ROC_AUC": "{:.4f}".format,
            },
        )
    )


    # =====================================================
    # 12. IMPROVEMENT OVER BASELINE
    # =====================================================

    print_header(
        "12. IMPROVEMENT OVER BASELINE"
    )

    five_accuracy_gain = (
        five_feature_results["accuracy"]
        - baseline_accuracy
    )

    full_accuracy_gain = (
        full_feature_results["accuracy"]
        - baseline_accuracy
    )

    print(
        f"5-feature RF vs baseline: "
        f"{five_accuracy_gain:+.2%}"
    )

    print(
        f"Full 30-feature RF vs baseline: "
        f"{full_accuracy_gain:+.2%}"
    )


    # =====================================================
    # 13. FEATURE MODEL COMPARISON
    # =====================================================

    print_header(
        "13. 5-FEATURE VS FULL-30 COMPARISON"
    )

    accuracy_difference = (
        five_feature_results["accuracy"]
        - full_feature_results["accuracy"]
    )

    f1_difference = (
        five_feature_results["f1"]
        - full_feature_results["f1"]
    )

    auc_difference = (
        five_feature_results["roc_auc"]
        - full_feature_results["roc_auc"]
    )

    print(
        f"Accuracy difference : "
        f"{accuracy_difference:+.2%}"
    )

    print(
        f"F1 difference       : "
        f"{f1_difference:+.4f}"
    )

    print(
        f"ROC-AUC difference  : "
        f"{auc_difference:+.4f}"
    )


    # =====================================================
    # 14. FINAL MODEL DECISION
    # =====================================================

    print_header(
        "14. FINAL HOLDOUT DECISION"
    )

    # Primary criterion:
    # ROC-AUC
    #
    # Secondary:
    # Accuracy
    #
    # This prevents selecting a model solely because
    # of one metric.

    five_auc = five_feature_results[
        "roc_auc"
    ]

    full_auc = full_feature_results[
        "roc_auc"
    ]

    five_accuracy = five_feature_results[
        "accuracy"
    ]

    full_accuracy = full_feature_results[
        "accuracy"
    ]

    if (
        five_auc > full_auc
        and five_accuracy >= baseline_accuracy
    ):

        winner = "5-Feature Random Forest"

    elif (
        full_auc > five_auc
        and full_accuracy >= baseline_accuracy
    ):

        winner = "Full 30-Feature Random Forest"

    elif five_accuracy > full_accuracy:

        winner = "5-Feature Random Forest"

    else:

        winner = "Full 30-Feature Random Forest"


    print()
    print(
        f"🏆 FINAL HOLDOUT WINNER: {winner}"
    )

    print()

    if winner == "5-Feature Random Forest":

        print(
            "The compact 5-feature model "
            "generalizes better on the untouched "
            "holdout."
        )

        print()

        print(
            "Selected features:"
        )

        for feature in SELECTED_FEATURES:
            print(
                f"  • {feature}"
            )

    else:

        print(
            "The full 30-feature model "
            "generalizes better on the untouched "
            "holdout."
        )


    # =====================================================
    # 15. SAVE RESULTS
    # =====================================================

    output_path = (
        "tests/final_holdout_validation.csv"
    )

    comparison.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Results saved to: {output_path}"
    )


    # =====================================================
    # 16. FINAL STATUS
    # =====================================================

    print_header(
        "15. MILESTONE 3.15 STATUS"
    )

    print()

    print(
        "Final holdout evaluation complete."
    )

    print(
        "The 61-row holdout has now been used "
        "for final evaluation."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Milestone 3.16 — Prediction Stability Test"
    )

    print(
        "This will check whether the model remains "
        "reasonably stable across different market periods."
    )

    print()

    print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()