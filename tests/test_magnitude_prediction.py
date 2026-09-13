import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from services.market_data import fetch_stock_data
from services.prediction_service import PredictionService


# =========================================================
# CONFIGURATION
# =========================================================

TICKER = "AAPL"
PERIOD = "2y"

HORIZONS = [1, 3, 5]

TEST_SIZE = 0.20


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n" + "=" * 80)
    print("STOCKSENSE — MOVEMENT MAGNITUDE VALIDATION 2.3")
    print("=" * 80)

    print(f"\nTicker: {TICKER}")
    print(f"Period: {PERIOD}")
    print("Target: absolute future price movement")
    print("Validation: chronological 80/20 split")

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    print("\nFetching market data...")

    df = fetch_stock_data(
        TICKER,
        period=PERIOD
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    # -----------------------------------------------------
    # BUILD FEATURES
    # -----------------------------------------------------

    service = PredictionService()

    features = service.prepare_features(
        df
    )

    print(
        f"Feature rows: {len(features)}"
    )

    feature_columns = (
        service.get_feature_columns()
    )

    missing_features = [
        column
        for column in feature_columns
        if column not in features.columns
    ]

    if missing_features:

        print("\nERROR: Missing features:")

        for column in missing_features:
            print(
                f" - {column}"
            )

        return

    close = features["Close"]

    all_results = []

    # =====================================================
    # TEST EACH HORIZON
    # =====================================================

    for horizon in HORIZONS:

        print("\n" + "=" * 80)
        print(
            f"HORIZON: NEXT {horizon} DAY(S)"
        )
        print("=" * 80)

        # -------------------------------------------------
        # ABSOLUTE RETURN TARGET
        # -------------------------------------------------

        future_return = (
            close.shift(-horizon)
            .div(close)
            - 1
        )

        target = future_return.abs()

        target_name = (
            f"Absolute_{horizon}D_Return"
        )

        dataset = features[
            feature_columns
        ].copy()

        dataset[target_name] = target

        dataset = dataset.replace(
            [np.inf, -np.inf],
            np.nan
        )

        dataset = dataset.dropna()

        X = dataset[
            feature_columns
        ]

        y = dataset[
            target_name
        ]

        print(
            f"Usable rows: {len(dataset)}"
        )

        print(
            f"Target mean: {y.mean():.6f}"
        )

        print(
            f"Target std:  {y.std():.6f}"
        )

        print(
            f"Target min:  {y.min():.6f}"
        )

        print(
            f"Target max:  {y.max():.6f}"
        )

        # -------------------------------------------------
        # CHRONOLOGICAL SPLIT
        # -------------------------------------------------

        split_index = int(
            len(X) * (1 - TEST_SIZE)
        )

        X_train = X.iloc[
            :split_index
        ]

        X_test = X.iloc[
            split_index:
        ]

        y_train = y.iloc[
            :split_index
        ]

        y_test = y.iloc[
            split_index:
        ]

        print(
            f"\nTrain rows: {len(X_train)}"
        )

        print(
            f"Test rows:  {len(X_test)}"
        )

        # -------------------------------------------------
        # BASELINE
        #
        # Predict the average movement from training data
        # -------------------------------------------------

        baseline_value = (
            y_train.mean()
        )

        baseline_prediction = np.full(
            len(y_test),
            baseline_value
        )

        baseline_metrics = calculate_metrics(
            y_test.to_numpy(),
            baseline_prediction
        )

        print("\nBASELINE")
        print(
            f"Prediction: {baseline_value:.6f}"
        )

        print(
            f"MAE:  {baseline_metrics['MAE']:.6f}"
        )

        print(
            f"RMSE: {baseline_metrics['RMSE']:.6f}"
        )

        print(
            f"R²:   {baseline_metrics['R2']:.6f}"
        )

        # -------------------------------------------------
        # MODELS
        # -------------------------------------------------

        models = {

            "Ridge": Ridge(
                alpha=1.0
            ),

            "Random Forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            ),

            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        }

        # -------------------------------------------------
        # TRAIN + EVALUATE
        # -------------------------------------------------

        for model_name, model in models.items():

            print(
                f"\nTraining {model_name}..."
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            metrics = calculate_metrics(
                y_test.to_numpy(),
                predictions
            )

            mae_improvement = (
                baseline_metrics["MAE"]
                - metrics["MAE"]
            )

            rmse_improvement = (
                baseline_metrics["RMSE"]
                - metrics["RMSE"]
            )

            all_results.append({
                "Horizon": f"{horizon}D",
                "Model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "MAE Improvement":
                    mae_improvement,
                "RMSE Improvement":
                    rmse_improvement
            })

            print(
                f"MAE:  {metrics['MAE']:.6f}"
            )

            print(
                f"RMSE: {metrics['RMSE']:.6f}"
            )

            print(
                f"R²:   {metrics['R2']:.6f}"
            )

            print(
                f"MAE improvement: "
                f"{mae_improvement:+.6f}"
            )

            print(
                f"RMSE improvement: "
                f"{rmse_improvement:+.6f}"
            )

    # =====================================================
    # FINAL COMPARISON
    # =====================================================

    results_df = pd.DataFrame(
        all_results
    )

    print("\n" + "=" * 80)
    print("FINAL MAGNITUDE COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "MAE":
                    "{:.6f}".format,
                "RMSE":
                    "{:.6f}".format,
                "R2":
                    "{:.6f}".format,
                "MAE Improvement":
                    "{:+.6f}".format,
                "RMSE Improvement":
                    "{:+.6f}".format
            }
        )
    )

    # =====================================================
    # BEST RESULTS
    # =====================================================

    print("\n" + "=" * 80)
    print("BEST RESULTS")
    print("=" * 80)

    best_mae = results_df.loc[
        results_df["MAE"].idxmin()
    ]

    best_rmse = results_df.loc[
        results_df["RMSE"].idxmin()
    ]

    best_r2 = results_df.loc[
        results_df["R2"].idxmax()
    ]

    print(
        "\nBest MAE:"
    )

    print(
        f"{best_mae['Horizon']} "
        f"{best_mae['Model']} "
        f"-> "
        f"{best_mae['MAE']:.6f}"
    )

    print(
        "\nBest RMSE:"
    )

    print(
        f"{best_rmse['Horizon']} "
        f"{best_rmse['Model']} "
        f"-> "
        f"{best_rmse['RMSE']:.6f}"
    )

    print(
        "\nBest R²:"
    )

    print(
        f"{best_r2['Horizon']} "
        f"{best_r2['Model']} "
        f"-> "
        f"{best_r2['R2']:.6f}"
    )

    # =====================================================
    # BASELINE WINNERS
    # =====================================================

    print("\n" + "=" * 80)
    print("BASELINE COMPARISON")
    print("=" * 80)

    for horizon in HORIZONS:

        horizon_df = results_df[
            results_df["Horizon"]
            == f"{horizon}D"
        ]

        mae_winners = horizon_df[
            horizon_df[
                "MAE Improvement"
            ] > 0
        ]

        rmse_winners = horizon_df[
            horizon_df[
                "RMSE Improvement"
            ] > 0
        ]

        print(
            f"\n{horizon}-DAY MAGNITUDE"
        )

        if len(mae_winners) > 0:

            print(
                "✓ Models beating baseline MAE:"
            )

            for _, row in mae_winners.iterrows():

                print(
                    f"  {row['Model']} "
                    f"({row['MAE Improvement']:+.6f})"
                )

        else:

            print(
                "✗ No model beats baseline MAE"
            )

        if len(rmse_winners) > 0:

            print(
                "✓ Models beating baseline RMSE:"
            )

            for _, row in rmse_winners.iterrows():

                print(
                    "  "
                    f"{row['Model']} "
                    f"({row['RMSE Improvement']:+.6f})"
                )

        else:

            print(
                "✗ No model beats baseline RMSE"
            )

    print("\n" + "=" * 80)
    print("MOVEMENT MAGNITUDE VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()