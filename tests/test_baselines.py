"""
StockSense — Baseline Comparison
---------------------------------

Compares ML models against simple forecasting baselines.

Baselines:
1. Zero Return
2. Mean Return
3. Previous Return (momentum)
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone

from services.market_data import fetch_stock_data
from services.prediction_service import PredictionService


# =========================================================
# SETTINGS
# =========================================================

TICKER = "AAPL"
PERIOD = "2y"
N_SPLITS = 5


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(y_actual, y_pred):

    mae = mean_absolute_error(
        y_actual,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_actual,
            y_pred
        )
    )

    r2 = r2_score(
        y_actual,
        y_pred
    )

    actual_direction = np.sign(
        np.asarray(y_actual)
    )

    predicted_direction = np.sign(
        np.asarray(y_pred)
    )

    direction = np.mean(
        actual_direction == predicted_direction
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional Accuracy": direction
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("StockSense — Baseline Comparison")
    print("=" * 70)

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    data = fetch_stock_data(
        TICKER,
        period=PERIOD
    )

    print(f"\nLoaded {len(data)} rows")

    # -----------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------

    prediction_service = PredictionService()

    features = prediction_service.prepare_features(
        data
    )

    # -----------------------------------------------------
    # Target = next-day return
    # -----------------------------------------------------

    y = (
        features["Close"].shift(-1)
        .div(features["Close"])
        - 1
    )

    valid_rows = y.notna()

    y = y.loc[valid_rows].copy()

    # -----------------------------------------------------
    # Features
    # -----------------------------------------------------

    selected_features = (
        prediction_service.get_selected_feature_columns()
    )

    X = features.loc[
        valid_rows,
        selected_features
    ].copy()

    # -----------------------------------------------------
    # Time-series split
    # -----------------------------------------------------

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS
    )

    # -----------------------------------------------------
    # Results storage
    # -----------------------------------------------------

    results = []

    # =====================================================
    # WALK-FORWARD BASELINES
    # =====================================================

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X),
        start=1
    ):

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # -------------------------------------------------
        # 1. Zero Return
        # -------------------------------------------------

        zero_prediction = np.zeros(
            len(y_test)
        )

        zero_metrics = calculate_metrics(
            y_test,
            zero_prediction
        )

        results.append({
            "Fold": fold,
            "Model": "Zero Return",
            **zero_metrics
        })

        # -------------------------------------------------
        # 2. Mean Return
        # -------------------------------------------------

        mean_return = y_train.mean()

        mean_prediction = np.full(
            len(y_test),
            mean_return
        )

        mean_metrics = calculate_metrics(
            y_test,
            mean_prediction
        )

        results.append({
            "Fold": fold,
            "Model": "Mean Return",
            **mean_metrics
        })

        # -------------------------------------------------
        # 3. Previous Return
        # -------------------------------------------------

        previous_predictions = np.full(
            len(y_test),
            y.iloc[train_idx[-1]]
        )

        # For subsequent test points, use the
        # previous observed return.
        for i in range(1, len(test_idx)):

            previous_predictions[i] = (
                y.iloc[test_idx[i] - 1]
            )

        previous_metrics = calculate_metrics(
            y_test,
            previous_predictions
        )

        results.append({
            "Fold": fold,
            "Model": "Previous Return",
            **previous_metrics
        })

    # =====================================================
    # ML MODELS
    # =====================================================

    models = prediction_service.model_manager.models

    for model_name, model in models.items():

        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(X),
            start=1
        ):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            current_model = clone(model)

            current_model.fit(
                X_train,
                y_train
            )

            predictions = current_model.predict(
                X_test
            )

            metrics = calculate_metrics(
                y_test,
                predictions
            )

            results.append({
                "Fold": fold,
                "Model": model_name,
                **metrics
            })

    # =====================================================
    # DATAFRAME
    # =====================================================

    results_df = pd.DataFrame(results)

    # -----------------------------------------------------
    # Fold-by-fold results
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FOLD-BY-FOLD RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # =====================================================
    # AVERAGES
    # =====================================================

    summary = (
        results_df
        .groupby("Model")
        .agg({
            "MAE": ["mean", "std"],
            "RMSE": ["mean", "std"],
            "R2": ["mean", "std"],
            "Directional Accuracy": ["mean", "std"]
        })
    )

    print("\n")
    print("=" * 70)
    print("AVERAGE PERFORMANCE")
    print("=" * 70)

    print(
        summary.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # =====================================================
    # SIMPLE RANKING
    # =====================================================

    average_results = (
        results_df
        .groupby("Model")
        .mean(numeric_only=True)
        .sort_values("MAE")
    )

    print("\n")
    print("=" * 70)
    print("RANKING BY MAE")
    print("=" * 70)

    print(
        average_results[
            [
                "MAE",
                "RMSE",
                "R2",
                "Directional Accuracy"
            ]
        ].to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # =====================================================
    # CONCLUSION
    # =====================================================

    print("\n")
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    best_mae = average_results["MAE"].idxmin()

    best_direction = (
        average_results[
            "Directional Accuracy"
        ].idxmax()
    )

    print(
        f"\nBest MAE: {best_mae}"
    )

    print(
        f"Best Directional Accuracy: "
        f"{best_direction}"
    )

    print("""
Interpretation:

- ML models should outperform simple baselines
  to justify their additional complexity.

- Lower MAE/RMSE is better.

- Higher R² is better.

- Directional Accuracy around 50% means the
  model has little directional advantage.

- A baseline that performs as well as or better
  than an ML model means the ML model needs
  improvement before production use.
""")

    print("=" * 70)


if __name__ == "__main__":
    main()