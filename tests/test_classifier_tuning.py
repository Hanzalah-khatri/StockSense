"""
Milestone 3.9
Random Forest Hyperparameter Tuning

Goal:
Improve StockSense's directional prediction model
without introducing time-series data leakage.
"""

import itertools
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import TimeSeriesSplit

from services.market_data import fetch_stock_data
from services.prediction_service import PredictionService


# =========================================================
# SETTINGS
# =========================================================

TICKER = "AAPL"

N_SPLITS = 4
RANDOM_STATE = 42

# Percentage of data reserved for final untouched testing
TEST_SIZE = 0.20


# =========================================================
# HELPER: CLASSIFICATION METRICS
# =========================================================

def calculate_metrics(y_true, y_pred, y_prob):

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    }

    try:
        metrics["ROC-AUC"] = roc_auc_score(
            y_true,
            y_prob
        )
    except ValueError:
        metrics["ROC-AUC"] = np.nan

    return metrics


# =========================================================
# HELPER: EVALUATE ONE PARAMETER SET
# =========================================================

def evaluate_parameters(X_train, y_train, params):

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        tscv.split(X_train),
        start=1
    ):

        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        model = RandomForestClassifier(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(
            X_fold_train,
            y_fold_train
        )

        predictions = model.predict(X_fold_val)

        probabilities = model.predict_proba(
            X_fold_val
        )[:, 1]

        metrics = calculate_metrics(
            y_fold_val,
            predictions,
            probabilities
        )

        fold_results.append(metrics)

    results_df = pd.DataFrame(fold_results)

    return {
        "Accuracy": results_df["Accuracy"].mean(),
        "Precision": results_df["Precision"].mean(),
        "Recall": results_df["Recall"].mean(),
        "F1": results_df["F1"].mean(),
        "ROC-AUC": results_df["ROC-AUC"].mean()
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("STOCKSENSE — RANDOM FOREST CLASSIFIER TUNING")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. LOAD DATA
    # -----------------------------------------------------

    print("\n[1/6] Loading market data...")

    df = fetch_stock_data(
        TICKER,
        period="2y"
    )

    if df is None or df.empty:
        raise ValueError("No market data received.")

    print(f"Rows received: {len(df)}")

    # -----------------------------------------------------
    # 2. PREPARE FEATURES
    # -----------------------------------------------------

    print("\n[2/6] Preparing features...")

    prediction_service = PredictionService()

    features = prediction_service.prepare_features(df)

    feature_columns = prediction_service.get_feature_columns()

    X = features[feature_columns].copy()

    # -----------------------------------------------------
    # TARGET
    #
    # 1 = next day's price goes UP
    # 0 = next day's price goes DOWN
    # -----------------------------------------------------

    next_day_return = (
        features["Close"]
        .shift(-1)
        .div(features["Close"])
        - 1
    )

    y = (next_day_return > 0).astype(int)

    # Remove final row because it has no next-day target
    valid_rows = next_day_return.notna()

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    # Safety cleanup
    valid_features = X.notna().all(axis=1)

    X = X.loc[valid_features].copy()
    y = y.loc[valid_features].copy()

    print(f"Usable rows: {len(X)}")
    print(f"Features: {len(feature_columns)}")

    print("\nTarget distribution:")

    print(
        f"DOWN: {(y == 0).sum()} "
        f"({(y == 0).mean() * 100:.2f}%)"
    )

    print(
        f"UP:   {(y == 1).sum()} "
        f"({(y == 1).mean() * 100:.2f}%)"
    )

    # -----------------------------------------------------
    # 3. FINAL HOLDOUT SPLIT
    # -----------------------------------------------------

    print("\n[3/6] Creating chronological holdout...")

    split_index = int(
        len(X) * (1 - TEST_SIZE)
    )

    X_train = X.iloc[:split_index].copy()
    y_train = y.iloc[:split_index].copy()

    X_test = X.iloc[split_index:].copy()
    y_test = y.iloc[split_index:].copy()

    print(f"Training rows: {len(X_train)}")
    print(f"Final test rows: {len(X_test)}")

    # -----------------------------------------------------
    # 4. DEFINE PARAMETER SEARCH
    # -----------------------------------------------------

    print("\n[4/6] Searching Random Forest parameters...")

    parameter_grid = {

        "n_estimators": [
            100,
            200,
            300
        ],

        "max_depth": [
            5,
            10,
            15,
            None
        ],

        "min_samples_split": [
            2,
            5,
            10
        ],

        "min_samples_leaf": [
            1,
            2,
            4
        ]
    }

    combinations = list(
        itertools.product(
            parameter_grid["n_estimators"],
            parameter_grid["max_depth"],
            parameter_grid["min_samples_split"],
            parameter_grid["min_samples_leaf"]
        )
    )

    print(f"Parameter combinations: {len(combinations)}")
    print(f"Time-series folds per combination: {N_SPLITS}")
    print(
        f"Total model fits: "
        f"{len(combinations) * N_SPLITS}"
    )

    tuning_results = []

    for number, combination in enumerate(
        combinations,
        start=1
    ):

        params = {
            "n_estimators": combination[0],
            "max_depth": combination[1],
            "min_samples_split": combination[2],
            "min_samples_leaf": combination[3]
        }

        print(
            f"\rTesting {number}/{len(combinations)}...",
            end=""
        )

        metrics = evaluate_parameters(
            X_train,
            y_train,
            params
        )

        result = {
            **params,
            **metrics
        }

        tuning_results.append(result)

    print("\n")

    results_df = pd.DataFrame(
        tuning_results
    )

    # -----------------------------------------------------
    # 5. DISPLAY BEST CONFIGURATIONS
    # -----------------------------------------------------

    print("[5/6] Best configurations")
    print("-" * 70)

    # Primary metric = Accuracy
    # Secondary metric = ROC-AUC

    results_df = results_df.sort_values(
        by=[
            "Accuracy",
            "ROC-AUC"
        ],
        ascending=False
    )

    print(
        results_df[
            [
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "Accuracy",
                "F1",
                "ROC-AUC"
            ]
        ].head(10).to_string(
            index=False
        )
    )

    best_row = results_df.iloc[0]

    best_params = {
        "n_estimators": int(
            best_row["n_estimators"]
        ),

        "max_depth": (
            None
            if pd.isna(best_row["max_depth"])
            else int(best_row["max_depth"])
        ),

        "min_samples_split": int(
            best_row["min_samples_split"]
        ),

        "min_samples_leaf": int(
            best_row["min_samples_leaf"]
        )
    }

    print("\nBEST PARAMETERS")
    print("-" * 70)

    for key, value in best_params.items():
        print(f"{key}: {value}")

    print("\nCross-validation performance:")

    print(
        f"Accuracy : "
        f"{best_row['Accuracy'] * 100:.2f}%"
    )

    print(
        f"F1       : "
        f"{best_row['F1'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC  : "
        f"{best_row['ROC-AUC']:.4f}"
    )

    # -----------------------------------------------------
    # 6. FINAL UNTOUCHED TEST
    # -----------------------------------------------------

    print("\n[6/6] Final untouched test...")
    print("-" * 70)

    # -------------------------
    # DEFAULT RANDOM FOREST
    # -------------------------

    default_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    default_model.fit(
        X_train,
        y_train
    )

    default_pred = default_model.predict(
        X_test
    )

    default_prob = default_model.predict_proba(
        X_test
    )[:, 1]

    default_metrics = calculate_metrics(
        y_test,
        default_pred,
        default_prob
    )

    # -------------------------
    # TUNED RANDOM FOREST
    # -------------------------

    tuned_model = RandomForestClassifier(
        **best_params,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    tuned_model.fit(
        X_train,
        y_train
    )

    tuned_pred = tuned_model.predict(
        X_test
    )

    tuned_prob = tuned_model.predict_proba(
        X_test
    )[:, 1]

    tuned_metrics = calculate_metrics(
        y_test,
        tuned_pred,
        tuned_prob
    )

    # -------------------------
    # MAJORITY BASELINE
    # -------------------------

    majority_class = y_train.mode()[0]

    majority_pred = np.full(
        len(y_test),
        majority_class
    )

    majority_metrics = calculate_metrics(
        y_test,
        majority_pred,
        majority_pred
    )

    # -----------------------------------------------------
    # FINAL COMPARISON
    # -----------------------------------------------------

    comparison = pd.DataFrame(
        {
            "Default RF": default_metrics,
            "Tuned RF": tuned_metrics,
            "Majority Baseline": majority_metrics
        }
    ).T

    print("\nFINAL HOLDOUT COMPARISON")
    print("=" * 70)

    print(
        comparison[
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC"
            ]
        ].to_string()
    )

    print("\n")

    # -----------------------------------------------------
    # IMPROVEMENT
    # -----------------------------------------------------

    default_accuracy = (
        default_metrics["Accuracy"]
    )

    tuned_accuracy = (
        tuned_metrics["Accuracy"]
    )

    improvement = (
        tuned_accuracy
        - default_accuracy
    ) * 100

    print("=" * 70)

    if improvement > 0:

        print(
            f"✅ Tuned RF improved accuracy by "
            f"{improvement:.2f} percentage points."
        )

    elif improvement < 0:

        print(
            f"⚠️ Tuned RF decreased accuracy by "
            f"{abs(improvement):.2f} percentage points."
        )

    else:

        print(
            "➡️ Tuning produced the same accuracy."
        )

    print("=" * 70)

    print("\nMilestone 3.9 complete.")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()