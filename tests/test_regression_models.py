import sys
from pathlib import Path

import pandas as pd

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.market_data import fetch_stock_data
from services.prediction_service import PredictionService


def main():

    print("\n" + "=" * 75)
    print("STOCKSENSE — REGRESSION MODEL VALIDATION 2.0")
    print("=" * 75)

    ticker = "AAPL"

    print(f"\nTicker: {ticker}")
    print("Period: 2 years")
    print("Validation: chronological 80/20 split")

    # =====================================================
    # 1. FETCH DATA
    # =====================================================

    print("\nFetching market data...")

    df = fetch_stock_data(
        ticker,
        period="2y"
    )

    if df.empty:
        raise ValueError(
            f"No market data found for {ticker}."
        )

    print(f"Rows loaded: {len(df)}")

    # =====================================================
    # 2. PREPARE FEATURES
    # =====================================================

    service = PredictionService()

    features = service.prepare_features(
        df
    )

    print(
        f"Feature rows: {len(features)}"
    )

    # =====================================================
    # 3. PREPARE ML DATA
    # =====================================================

    X, y = service.prepare_ml_data(
        features
    )

    print(
        f"Usable ML rows: {len(X)}"
    )

    # =====================================================
    # 4. EVALUATE MODELS
    # =====================================================

    results, best_model = (
        service.evaluate_models(
            X,
            y
        )
    )

    # =====================================================
    # 5. DISPLAY RESULTS
    # =====================================================

    print("\n" + "=" * 75)
    print("MODEL RESULTS")
    print("=" * 75)

    display_columns = [
        "Directional Accuracy",
        "MAE",
        "RMSE",
        "R2"
    ]

    available_columns = [
        column
        for column in display_columns
        if column in results.columns
    ]

    display = results[
        available_columns
    ].copy()

    print(
        display.round(4).to_string()
    )

    # =====================================================
    # 6. BASELINE COMPARISON
    # =====================================================

    print("\n" + "=" * 75)
    print("BASELINE COMPARISON")
    print("=" * 75)

    if "Naive Baseline" in results.index:

        baseline = results.loc[
            "Naive Baseline"
        ]

        print(
            f"Baseline MAE: "
            f"{baseline['MAE']:.4f}"
        )

        print(
            f"Baseline RMSE: "
            f"{baseline['RMSE']:.4f}"
        )

        print(
            f"Baseline R²: "
            f"{baseline['R2']:.4f}"
        )

    # =====================================================
    # 7. BEST MODEL
    # =====================================================

    print("\n" + "=" * 75)
    print("SELECTED MODEL")
    print("=" * 75)

    print(
        f"Best model: {best_model}"
    )

    if best_model in results.index:

        best = results.loc[
            best_model
        ]

        print(
            f"Directional Accuracy: "
            f"{best['Directional Accuracy']:.2f}%"
        )

        print(
            f"MAE: "
            f"{best['MAE']:.4f}"
        )

        print(
            f"RMSE: "
            f"{best['RMSE']:.4f}"
        )

        print(
            f"R²: "
            f"{best['R2']:.4f}"
        )

    # =====================================================
    # 8. BASELINE TEST
    # =====================================================

    print("\n" + "=" * 75)
    print("BASELINE TEST")
    print("=" * 75)

    if "Naive Baseline" in results.index:

        best_mae = results.loc[
            best_model,
            "MAE"
        ]

        baseline_mae = results.loc[
            "Naive Baseline",
            "MAE"
        ]

        best_rmse = results.loc[
            best_model,
            "RMSE"
        ]

        baseline_rmse = results.loc[
            "Naive Baseline",
            "RMSE"
        ]

        print(
            f"MAE improvement: "
            f"{baseline_mae - best_mae:.4f}"
        )

        print(
            f"RMSE improvement: "
            f"{baseline_rmse - best_rmse:.4f}"
        )

        if best_mae < baseline_mae:
            print(
                "✓ Model beats baseline on MAE"
            )
        else:
            print(
                "✗ Model does NOT beat baseline on MAE"
            )

        if best_rmse < baseline_rmse:
            print(
                "✓ Model beats baseline on RMSE"
            )
        else:
            print(
                "✗ Model does NOT beat baseline on RMSE"
            )

    print("\n" + "=" * 75)
    print("VALIDATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()