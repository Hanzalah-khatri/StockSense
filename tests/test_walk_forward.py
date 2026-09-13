"""
StockSense — Walk-Forward Validation
-------------------------------------

Tests whether the selected feature set and models
generalize across multiple historical time periods.

No production files are modified by this test.
"""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from services.market_data import fetch_stock_data
from services.prediction_service import PredictionService


# =========================================================
# SETTINGS
# =========================================================

TICKER = "AAPL"
PERIOD = "2y"
N_SPLITS = 5


# =========================================================
# DIRECTIONAL ACCURACY
# =========================================================

def directional_accuracy(y_actual, y_pred):
    """
    Measures how often the model correctly predicts
    whether the next return is positive or negative.
    """

    actual_direction = np.sign(np.asarray(y_actual))
    predicted_direction = np.sign(np.asarray(y_pred))

    return np.mean(
        actual_direction == predicted_direction
    ) * 100


# =========================================================
# EVALUATE ONE MODEL
# =========================================================

def evaluate_model(model, X, y, splitter):

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Fresh model for every fold
        current_model = clone(model)

        current_model.fit(
            X_train,
            y_train
        )

        predictions = current_model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        direction = directional_accuracy(
            y_test,
            predictions
        )

        fold_results.append({
            "Fold": fold,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Directional Accuracy": direction
        })

    return pd.DataFrame(fold_results)


# =========================================================
# PRINT RESULTS
# =========================================================

def print_results(title, results):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nAverage:")
    print(
        f"MAE:                  {results['MAE'].mean():.4f}"
    )

    print(
        f"RMSE:                 {results['RMSE'].mean():.4f}"
    )

    print(
        f"R²:                   {results['R2'].mean():.4f}"
    )

    print(
        f"Directional Accuracy: "
        f"{results['Directional Accuracy'].mean():.2f}%"
    )

    print("\nStd Dev:")

    print(
        f"MAE:                  {results['MAE'].std():.4f}"
    )

    print(
        f"RMSE:                 {results['RMSE'].std():.4f}"
    )

    print(
        f"R²:                   {results['R2'].std():.4f}"
    )

    print(
        f"Directional Accuracy: "
        f"{results['Directional Accuracy'].std():.2f}%"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("StockSense — Walk-Forward Validation")
    print("=" * 70)

    # -----------------------------------------------------
    # Load market data
    # -----------------------------------------------------

    data = fetch_stock_data(
        TICKER,
        period=PERIOD
    )

    print(f"\nLoaded {len(data)} rows")

    # -----------------------------------------------------
    # Prepare features
    # -----------------------------------------------------

    prediction_service = PredictionService()

    features = prediction_service.prepare_features(
        data
    )

    # -----------------------------------------------------
    # Target = NEXT-DAY RETURN
    # -----------------------------------------------------

    y = (
        features["Close"].shift(-1)
        .div(features["Close"])
        - 1
    )

    valid_rows = y.notna()

    y = y.loc[valid_rows].copy()

    # -----------------------------------------------------
    # Feature sets
    # -----------------------------------------------------

    current_features = (
        prediction_service.get_feature_columns()
    )

    selected_features = (
        prediction_service.get_selected_feature_columns()
    )

    X_current = (
        features.loc[valid_rows, current_features]
        .copy()
    )

    X_selected = (
        features.loc[valid_rows, selected_features]
        .copy()
    )

    # -----------------------------------------------------
    # Models
    # -----------------------------------------------------

    model_manager = prediction_service.model_manager

    models = model_manager.models

    # -----------------------------------------------------
    # Time-Series Split
    # -----------------------------------------------------

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS
    )

    print("\n")
    print("Walk-forward configuration:")
    print(f"Ticker:              {TICKER}")
    print(f"Feature sets:        30 vs 15")
    print(f"Splits:              {N_SPLITS}")
    print("Training:            Expanding window")
    print("Testing:             Future unseen data")

    # =====================================================
    # CURRENT FEATURES
    # =====================================================

    print("\n\n")
    print("#" * 70)
    print("# CURRENT FEATURES — 30")
    print("#" * 70)

    current_summary = []

    for model_name, model in models.items():

        results = evaluate_model(
            model,
            X_current,
            y,
            splitter
        )

        print_results(
            model_name,
            results
        )

        current_summary.append({
            "Feature Set": "Current (30)",
            "Model": model_name,
            "Avg MAE": results["MAE"].mean(),
            "Avg RMSE": results["RMSE"].mean(),
            "Avg R2": results["R2"].mean(),
            "Avg Direction": results[
                "Directional Accuracy"
            ].mean()
        })

    # =====================================================
    # SELECTED FEATURES
    # =====================================================

    print("\n\n")
    print("#" * 70)
    print("# SELECTED FEATURES — 15")
    print("#" * 70)

    selected_summary = []

    for model_name, model in models.items():

        results = evaluate_model(
            model,
            X_selected,
            y,
            splitter
        )

        print_results(
            model_name,
            results
        )

        selected_summary.append({
            "Feature Set": "Selected (15)",
            "Model": model_name,
            "Avg MAE": results["MAE"].mean(),
            "Avg RMSE": results["RMSE"].mean(),
            "Avg R2": results["R2"].mean(),
            "Avg Direction": results[
                "Directional Accuracy"
            ].mean()
        })

    # =====================================================
    # COMPARISON
    # =====================================================

    comparison = pd.DataFrame(
        current_summary + selected_summary
    )

    print("\n\n")
    print("=" * 70)
    print("FINAL WALK-FORWARD COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n")
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print("""
Use the following rules:

1. Lower MAE is better.
2. Lower RMSE is better.
3. Higher R² is better.
4. Directional Accuracy above 50% is potentially useful.
5. A model that performs consistently across folds
   is more trustworthy than one strong single split.
6. If Selected (15) consistently beats Current (30),
   feature selection is helping.
7. If results vary heavily between folds, the model
   may not generalize well.
""")

    print("=" * 70)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()