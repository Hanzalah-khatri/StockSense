# =========================================================
# StockSense — Feature Importance Analysis
# Milestone 4 — Step 1
# =========================================================

import os
import sys

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


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
# HEADER
# =========================================================

print("=" * 70)

print(
    "StockSense — Feature Importance Analysis"
)

print("=" * 70)

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
    f"Minimum training rows: {MIN_TRAIN_SIZE}"
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

print("=" * 70)

print(
    "1. LOADING MARKET DATA"
)

print("=" * 70)

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
        f"Unable to load market data "
        f"for {TICKER}."
    )


print(
    f"Loaded {len(market_data)} raw rows."
)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

print()

print("=" * 70)

print(
    "2. FEATURE ENGINEERING"
)

print("=" * 70)

print()


features = (
    prediction_service.prepare_features(
        market_data
    )
)


if features.empty:

    raise RuntimeError(
        "Feature engineering produced "
        "no usable data."
    )


print(
    f"Engineered rows: {len(features)}"
)


print(
    f"Total available columns: "
    f"{len(features.columns)}"
)


# =========================================================
# VALIDATION DATA
# =========================================================

print()

print("=" * 70)

print(
    "3. PREPARING VALIDATION DATA"
)

print("=" * 70)

print()


X, y = (
    validation_service.prepare_data(
        features
    )
)


print(
    f"Usable observations: {len(X)}"
)


print(
    f"Validation features: {X.shape[1]}"
)


print()

print(
    "Features used:"
)

for index, column in enumerate(
    X.columns,
    start=1
):

    print(
        f"  {index:>2}. {column}"
    )


# =========================================================
# VALIDATION CONFIGURATION
# =========================================================

total_rows = len(X)

test_size = (
    total_rows - MIN_TRAIN_SIZE
) // N_SPLITS


if test_size < 1:

    raise ValueError(
        "Not enough data for "
        "walk-forward validation."
    )


print()

print("=" * 70)

print(
    "4. WALK-FORWARD FEATURE IMPORTANCE"
)

print("=" * 70)

print()


# =========================================================
# STORAGE
# =========================================================

feature_importance_records = []


# =========================================================
# WALK-FORWARD FOLDS
# =========================================================

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


    if X_test.empty:

        continue


    print(
        f"Fold {fold + 1}: "
        f"train={len(X_train)}, "
        f"test={len(X_test)}"
    )


    # -----------------------------------------------------
    # SAME MODEL AS VALIDATION SERVICE
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    importances = (
        model.feature_importances_
    )


    for feature, importance in zip(
        X.columns,
        importances
    ):

        feature_importance_records.append(
            {

                "fold":
                    fold + 1,

                "feature":
                    feature,

                "importance":
                    float(
                        importance
                    )

            }
        )


# =========================================================
# VALIDATE RESULTS
# =========================================================

if not feature_importance_records:

    raise RuntimeError(
        "No feature importance results "
        "were generated."
    )


importance_df = pd.DataFrame(
    feature_importance_records
)


# =========================================================
# AGGREGATE RESULTS
# =========================================================

summary = (
    importance_df

    .groupby("feature")

    .agg(

        mean_importance=(
            "importance",
            "mean"
        ),

        std_importance=(
            "importance",
            "std"
        ),

        min_importance=(
            "importance",
            "min"
        ),

        max_importance=(
            "importance",
            "max"
        )

    )

    .sort_values(
        "mean_importance",
        ascending=False
    )
)


# =========================================================
# FEATURE RANKING
# =========================================================

print()

print("=" * 70)

print(
    "5. FEATURE IMPORTANCE RANKING"
)

print("=" * 70)

print()


print(
    f"{'Rank':<6}"
    f"{'Feature':<22}"
    f"{'Mean':>12}"
    f"{'Std':>12}"
    f"{'Min':>12}"
    f"{'Max':>12}"
)

print("-" * 70)


for rank, (
    feature,
    row
) in enumerate(
    summary.iterrows(),
    start=1
):

    print(
        f"{rank:<6}"
        f"{feature:<22}"
        f"{row['mean_importance']:>12.4f}"
        f"{row['std_importance']:>12.4f}"
        f"{row['min_importance']:>12.4f}"
        f"{row['max_importance']:>12.4f}"
    )


# =========================================================
# TOP FEATURES
# =========================================================

print()

print("=" * 70)

print(
    "6. TOP 10 FEATURES"
)

print("=" * 70)

print()


for rank, (
    feature,
    row
) in enumerate(
    summary.head(10).iterrows(),
    start=1
):

    print(
        f"{rank:>2}. "
        f"{feature:<22} "
        f"{row['mean_importance']:.4f}"
    )


# =========================================================
# LOW-IMPORTANCE FEATURES
# =========================================================

print()

print("=" * 70)

print(
    "7. LOW-IMPORTANCE FEATURES"
)

print("=" * 70)

print()


for feature, row in (
    summary.tail(5).iterrows()
):

    print(
        f"{feature:<22} "
        f"{row['mean_importance']:.4f}"
    )


# =========================================================
# STABILITY
# =========================================================

print()

print("=" * 70)

print(
    "8. FEATURE IMPORTANCE STABILITY"
)

print("=" * 70)

print()


for feature, row in (
    summary.iterrows()
):

    mean_value = (
        row["mean_importance"]
    )

    std_value = (
        row["std_importance"]
    )


    if mean_value > 0:

        coefficient_variation = (
            std_value
            / mean_value
        )

    else:

        coefficient_variation = np.inf


    if coefficient_variation < 0.50:

        stability = "STABLE"

    elif coefficient_variation < 1.00:

        stability = "MODERATE"

    else:

        stability = "UNSTABLE"


    print(
        f"{feature:<22} "
        f"CV={coefficient_variation:.2f} "
        f"{stability}"
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


output_path = os.path.join(
    reports_dir,
    "feature_importance.csv"
)


summary.to_csv(
    output_path
)


# =========================================================
# COMPLETE
# =========================================================

print()

print("=" * 70)

print(
    "9. RESULTS SAVED"
)

print("=" * 70)

print()

print(
    output_path
)

print()

print("=" * 70)

print(
    "Feature importance analysis complete."
)

print("=" * 70)

print()