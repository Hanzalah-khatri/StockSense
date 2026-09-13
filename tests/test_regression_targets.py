import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.market_data import fetch_stock_data
from services.prediction_service import PredictionService


def main():

    print("\n" + "=" * 75)
    print("STOCKSENSE — REGRESSION TARGET ANALYSIS")
    print("=" * 75)

    ticker = "AAPL"

    # =====================================================
    # 1. FETCH DATA
    # =====================================================

    df = fetch_stock_data(
        ticker,
        period="2y"
    )

    print(
        f"\nTicker: {ticker}"
    )

    print(
        f"Rows: {len(df)}"
    )

    # =====================================================
    # 2. PREPARE FEATURES
    # =====================================================

    service = PredictionService()

    features = service.prepare_features(
        df
    )

    # =====================================================
    # 3. BUILD DIFFERENT TARGETS
    # =====================================================

    close = features["Close"]

    targets = pd.DataFrame(index=features.index)

    # Next-day return
    targets["Next_1D_Return"] = (
        close.shift(-1)
        .div(close)
        - 1
    )

    # 3-day forward return
    targets["Next_3D_Return"] = (
        close.shift(-3)
        .div(close)
        - 1
    )

    # 5-day forward return
    targets["Next_5D_Return"] = (
        close.shift(-5)
        .div(close)
        - 1
    )

    # 5-day average forward return
    targets["Next_5D_Avg_Return"] = (
        close.shift(-5)
        .div(close)
        - 1
    ) / 5

    # Next-day direction
    targets["Next_1D_Direction"] = (
        targets["Next_1D_Return"] > 0
    ).astype(int)

    # =====================================================
    # 4. REMOVE INVALID ROWS
    # =====================================================

    print("\n" + "=" * 75)
    print("TARGET DISTRIBUTIONS")
    print("=" * 75)

    for column in targets.columns:

        series = targets[column].dropna()

        print(
            f"\n{column}"
        )

        print(
            f"Rows: {len(series)}"
        )

        print(
            f"Mean: {series.mean():.6f}"
        )

        print(
            f"Std: {series.std():.6f}"
        )

        print(
            f"Min: {series.min():.6f}"
        )

        print(
            f"Max: {series.max():.6f}"
        )

    # =====================================================
    # 5. NEXT-DAY RETURN BASELINE
    # =====================================================

    print("\n" + "=" * 75)
    print("NEXT-DAY RETURN BASELINE")
    print("=" * 75)

    y = targets[
        "Next_1D_Return"
    ].dropna()

    zero_prediction = np.zeros(
        len(y)
    )

    mae = np.mean(
        np.abs(
            y.to_numpy()
            -
            zero_prediction
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                y.to_numpy()
                -
                zero_prediction
            ) ** 2
        )
    )

    print(
        f"Zero-return MAE:  {mae:.6f}"
    )

    print(
        f"Zero-return RMSE: {rmse:.6f}"
    )

    # =====================================================
    # 6. DIRECTION BALANCE
    # =====================================================

    direction = targets[
        "Next_1D_Direction"
    ].dropna()

    up_percentage = (
        direction.mean()
        * 100
    )

    down_percentage = (
        100
        -
        up_percentage
    )

    print("\n" + "=" * 75)
    print("DIRECTION DISTRIBUTION")
    print("=" * 75)

    print(
        f"UP:   {up_percentage:.2f}%"
    )

    print(
        f"DOWN: {down_percentage:.2f}%"
    )

    print("\n" + "=" * 75)
    print("TARGET ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()