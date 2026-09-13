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
# HELPERS
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

    # Directional accuracy
    actual_direction = np.sign(y_true)
    predicted_direction = np.sign(y_pred)

    valid = predicted_direction != 0

    if valid.sum() > 0:
        directional_accuracy = (
            actual_direction[valid]
            == predicted_direction[valid]
        ).mean() * 100
    else:
        directional_accuracy = 0.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional Accuracy": directional_accuracy
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n" + "=" * 80)
    print("STOCKSENSE — REGRESSION HORIZON VALIDATION 2.2")
    print("=" * 80)

    print(f"\nTicker: {TICKER}")
    print(f"Period: {PERIOD}")
    print("Validation: chronological 80/20 split")

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    print("\nFetching market data...")

    df = fetch_stock_data(
        TICKER,
        period=PERIOD
    )

    print(f"Rows loaded: {len(df)}")

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

    # Make sure all required columns exist
    missing_features = [
        column
        for column in feature_columns
        if column not in features.columns
    ]

    if missing_features:

        print("\nERROR: Missing features:")
        for column in missing_features:
            print(" -", column)

        return

    # -----------------------------------------------------
    # PREPARE HORIZON TARGETS
    # -----------------------------------------------------

    close = features["Close"]

    results = []

    # -----------------------------------------------------
    # TEST EACH HORIZON
    # -----------------------------------------------------

    for horizon in HORIZONS:

        print("\n" + "=" * 80)
        print(
            f"HORIZON: NEXT {horizon} DAY(S)"
        )
        print("=" * 80)

        target_name = (
            f"Next_{horizon}D_Return"
        )

        target = (
            close.shift(-horizon)
            .div(close)
            - 1
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
            f"Train rows: {len(X_train)}"
        )

        print(
            f"Test rows:  {len(X_test)}"
        )

        # -------------------------------------------------
        # BASELINE
        # -------------------------------------------------

        baseline_prediction = np.zeros(
            len(y_test)
        )

        baseline_metrics = calculate_metrics(
            y_test.to_numpy(),
            baseline_prediction
        )

        print("\nBASELINE")
        print(
            f"MAE:  {baseline_metrics['MAE']:.6f}"
        )
        print(
            f"RMSE: {baseline_metrics['RMSE']:.6f}"
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

        horizon_results = []

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

            horizon_results.append({
                "Horizon": f"{horizon}D",
                "Model": model_name,
                "Directional Accuracy":
                    metrics[
                        "Directional Accuracy"
                    ],
                "MAE":
                    metrics["MAE"],
                "RMSE":
                    metrics["RMSE"],
                "R2":
                    metrics["R2"],
                "MAE Improvement":
                    mae_improvement,
                "RMSE Improvement":
                    rmse_improvement
            })

            print(
                f"Directional Accuracy: "
                f"{metrics['Directional Accuracy']:.2f}%"
            )

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
                f"MAE improvement vs baseline: "
                f"{mae_improvement:+.6f}"
            )

            print(
                f"RMSE improvement vs baseline: "
                f"{rmse_improvement:+.6f}"
            )

        results.extend(
            horizon_results
        )

    # =====================================================
    # FINAL COMPARISON
    # =====================================================

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 80)
    print("FINAL HORIZON COMPARISON")
    print("=" * 80)

    display_columns = [
        "Horizon",
        "Model",
        "Directional Accuracy",
        "MAE",
        "RMSE",
        "R2",
        "MAE Improvement",
        "RMSE Improvement"
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            formatters={
                "Directional Accuracy":
                    "{:.2f}".format,
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

    best_direction = results_df.loc[
        results_df[
            "Directional Accuracy"
        ].idxmax()
    ]

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
        "\nBest Directional Accuracy:"
    )
    print(
        f"{best_direction['Horizon']} "
        f"{best_direction['Model']} "
        f"-> "
        f"{best_direction['Directional Accuracy']:.2f}%"
    )

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
    print("BASELINE WINNERS")
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
            f"\n{horizon}-DAY HORIZON"
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
                    f"  {row['Model']} "
                    f"({row['RMSE Improvement']:+.6f})"
                )

        else:

            print(
                "✗ No model beats baseline RMSE"
            )

    print("\n" + "=" * 80)
    print("HORIZON VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()