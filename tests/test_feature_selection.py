# =========================================================
# StockSense — Feature Selection Experiment
# Milestone 4 — Step 2
# =========================================================

import os
import sys

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# =========================================================
# STOCKSENSE IMPORTS
# =========================================================

from services.market_data import (
    fetch_stock_data
)

from services.prediction_service import (
    PredictionService
)

from services.validation_service import (
    ValidationService
)


# =========================================================
# CONFIGURATION
# =========================================================

TICKER = "AAPL"

N_SPLITS = 5

MIN_TRAIN_SIZE = 150


# =========================================================
# FEATURE IMPORTANCE ORDER
#
# Taken directly from the Milestone 4 Step 1
# walk-forward feature importance results.
# =========================================================

FEATURE_RANKING = [

    "Return_Lag_1",
    "Return_Lag_3",
    "Return_Lag_5",
    "Volatility",
    "Volume_Change",
    "High_Low_Pct",
    "SMA_10",
    "Momentum_5",
    "Daily_Return",
    "SMA_5",
    "SMA_20",
    "High_Low_Range",
    "Relative_Volume",
    "Return_Lag_10",
    "BB_Position",

]


# =========================================================
# EXPERIMENTS
# =========================================================

FEATURE_SETS = {

    "All 15 Features":
        FEATURE_RANKING,

    "Top 10 Features":
        FEATURE_RANKING[:10],

    "Top 7 Features":
        FEATURE_RANKING[:7],

    "Top 5 Features":
        FEATURE_RANKING[:5],

    "Bottom 10 Features":
        FEATURE_RANKING[5:],

}


# =========================================================
# HEADER
# =========================================================

print("=" * 75)

print(
    "StockSense — Feature Selection Experiment"
)

print("=" * 75)

print()

print(
    f"Ticker: {TICKER}"
)

print(
    "Validation: Expanding walk-forward"
)

print(
    f"Splits: {N_SPLITS}"
)

print(
    f"Minimum training rows: "
    f"{MIN_TRAIN_SIZE}"
)

print()


# =========================================================
# INITIALIZE SERVICES
# =========================================================

prediction_service = (
    PredictionService()
)

validation_service = (
    ValidationService()
)


# =========================================================
# LOAD MARKET DATA
# =========================================================

print("=" * 75)

print(
    "1. LOADING DATA"
)

print("=" * 75)

print()


market_data = fetch_stock_data(
    ticker=TICKER,
    period="2y"
)


if (
    market_data is None
    or market_data.empty
):

    raise RuntimeError(
        f"Unable to load data for "
        f"{TICKER}."
    )


print(
    f"Loaded {len(market_data)} raw rows."
)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

features = (
    prediction_service.prepare_features(
        market_data
    )
)


# =========================================================
# PREPARE VALIDATION DATA
# =========================================================

X, y = (
    validation_service.prepare_data(
        features
    )
)


print(
    f"Usable observations: {len(X)}"
)

print(
    f"Available features: {X.shape[1]}"
)

print()


# =========================================================
# VALIDATION WINDOWS
# =========================================================

total_rows = len(X)

test_size = (
    total_rows - MIN_TRAIN_SIZE
) // N_SPLITS


if test_size < 1:

    raise ValueError(
        "Not enough observations for "
        "walk-forward validation."
    )


# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_feature_set(
    X,
    y,
    selected_features,
):

    X_selected = X[
        selected_features
    ].copy()


    fold_results = []


    all_actual = []

    all_predictions = []

    all_probabilities = []


    for fold in range(
        N_SPLITS
    ):

        train_end = (
            MIN_TRAIN_SIZE
            + fold * test_size
        )


        if fold == N_SPLITS - 1:

            test_end = total_rows

        else:

            test_end = (
                train_end
                + test_size
            )


        X_train = X_selected.iloc[
            :train_end
        ]

        y_train = y.iloc[
            :train_end
        ]


        X_test = X_selected.iloc[
            train_end:test_end
        ]

        y_test = y.iloc[
            train_end:test_end
        ]


        if X_test.empty:

            continue


        # -------------------------------------------------
        # SAME RANDOM FOREST AS MILESTONE 3
        # -------------------------------------------------

        model = (
            RandomForestClassifier(

                n_estimators=200,

                max_depth=10,

                min_samples_split=5,

                random_state=42,

                n_jobs=-1,

                class_weight="balanced"

            )
        )


        # -------------------------------------------------
        # TRAIN
        # -------------------------------------------------

        model.fit(
            X_train,
            y_train
        )


        # -------------------------------------------------
        # PREDICTIONS
        # -------------------------------------------------

        predictions = (
            model.predict(
                X_test
            )
        )


        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )


        # -------------------------------------------------
        # FOLD METRICS
        # -------------------------------------------------

        accuracy = (
            accuracy_score(
                y_test,
                predictions
            )
        )


        precision = (
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        )


        recall = (
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        )


        f1 = (
            f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        )


        try:

            auc = (
                roc_auc_score(
                    y_test,
                    probabilities
                )
            )

        except ValueError:

            auc = None


        fold_results.append({

            "fold":
                fold + 1,

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "roc_auc":
                auc,

        })


        all_actual.extend(
            y_test.tolist()
        )

        all_predictions.extend(
            predictions.tolist()
        )

        all_probabilities.extend(
            probabilities.tolist()
        )


    # =====================================================
    # AGGREGATE
    # =====================================================

    actual = all_actual

    predictions = all_predictions

    probabilities = all_probabilities


    accuracy = (
        accuracy_score(
            actual,
            predictions
        )
    )


    precision = (
        precision_score(
            actual,
            predictions,
            zero_division=0
        )
    )


    recall = (
        recall_score(
            actual,
            predictions,
            zero_division=0
        )
    )


    f1 = (
        f1_score(
            actual,
            predictions,
            zero_division=0
        )
    )


    try:

        auc = (
            roc_auc_score(
                actual,
                probabilities
            )
        )

    except ValueError:

        auc = None


    return {

        "features":
            len(selected_features),

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "roc_auc":
            auc,

        "folds":
            fold_results,

    }


# =========================================================
# RUN EXPERIMENTS
# =========================================================

results = []


for name, selected_features in (
    FEATURE_SETS.items()
):

    print("=" * 75)

    print(
        f"TESTING: {name}"
    )

    print("=" * 75)

    print()


    print(
        "Features:"
    )

    print(
        ", ".join(
            selected_features
        )
    )

    print()


    result = evaluate_feature_set(

        X,

        y,

        selected_features

    )


    result["name"] = name


    results.append(
        result
    )


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


    if result["roc_auc"] is not None:

        print(
            f"ROC-AUC:   "
            f"{result['roc_auc']:.4f}"
        )

    else:

        print(
            "ROC-AUC:   N/A"
        )


    print()


# =========================================================
# MAJORITY BASELINE
# =========================================================

down_count = (
    sum(
        value == 0
        for value in y.iloc[
            MIN_TRAIN_SIZE:
        ]
    )
)


up_count = (
    sum(
        value == 1
        for value in y.iloc[
            MIN_TRAIN_SIZE:
        ]
    )
)


baseline_accuracy = (
    max(
        up_count,
        down_count
    )
    /
    (
        up_count
        + down_count
    )
)


# =========================================================
# FINAL COMPARISON
# =========================================================

print()

print("=" * 75)

print(
    "FINAL FEATURE SET COMPARISON"
)

print("=" * 75)

print()


print(
    f"{'Feature Set':<22}"
    f"{'N':>5}"
    f"{'Accuracy':>12}"
    f"{'F1':>12}"
    f"{'ROC-AUC':>12}"
    f"{'vs Base':>12}"
)

print("-" * 75)


for result in results:

    improvement = (
        result["accuracy"]
        - baseline_accuracy
    )


    auc_text = (

        f"{result['roc_auc']:.4f}"

        if result["roc_auc"]
        is not None

        else "N/A"

    )


    print(

        f"{result['name']:<22}"

        f"{result['features']:>5}"

        f"{result['accuracy'] * 100:>11.2f}%"

        f"{result['f1'] * 100:>11.2f}%"

        f"{auc_text:>12}"

        f"{improvement * 100:>10.2f} pp"

    )


print()

print(
    f"Majority baseline: "
    f"{baseline_accuracy * 100:.2f}%"
)


# =========================================================
# BEST FEATURE SET
# =========================================================

best_result = max(
    results,
    key=lambda item:
        item["accuracy"]
)


print()

print("=" * 75)

print(
    "BEST FEATURE SET"
)

print("=" * 75)

print()


print(
    f"Set: "
    f"{best_result['name']}"
)

print(
    f"Features: "
    f"{best_result['features']}"
)

print(
    f"Accuracy: "
    f"{best_result['accuracy'] * 100:.2f}%"
)

print(
    f"F1: "
    f"{best_result['f1'] * 100:.2f}%"
)


if best_result["roc_auc"] is not None:

    print(
        f"ROC-AUC: "
        f"{best_result['roc_auc']:.4f}"
    )


# =========================================================
# SAVE RESULTS
# =========================================================

reports_dir = os.path.join(
    PROJECT_ROOT,
    "reports"
)


os.makedirs(
    reports_dir,
    exist_ok=True
)


comparison_rows = []


for result in results:

    comparison_rows.append({

        "feature_set":
            result["name"],

        "feature_count":
            result["features"],

        "accuracy":
            result["accuracy"],

        "precision":
            result["precision"],

        "recall":
            result["recall"],

        "f1":
            result["f1"],

        "roc_auc":
            result["roc_auc"],

        "baseline_accuracy":
            baseline_accuracy,

        "baseline_improvement":
            (
                result["accuracy"]
                - baseline_accuracy
            ),

    })


comparison_df = pd.DataFrame(
    comparison_rows
)


output_path = os.path.join(
    reports_dir,
    "feature_selection_comparison.csv"
)


comparison_df.to_csv(
    output_path,
    index=False
)


print()

print("=" * 75)

print(
    "RESULTS SAVED"
)

print("=" * 75)

print()

print(
    output_path
)

print()

print("=" * 75)

print(
    "Feature selection experiment complete."
)

print("=" * 75)