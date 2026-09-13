"""
StockSense Market Data Service
------------------------------
Handles downloading, validating, and cleaning stock market data.
"""

from datetime import datetime

import pandas as pd
import yfinance as yf


def fetch_stock_data(
    ticker: str,
    start_date=None,
    end_date=None,
    period: str = "1y"
) -> pd.DataFrame:
    """
    Fetch historical OHLCV stock data from Yahoo Finance.

    Returns:
        pandas.DataFrame containing:
        Date, Open, High, Low, Close, Volume

    Raises:
        ValueError: If ticker is invalid or no market data is returned.
    """

    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Stock ticker cannot be empty.")

    try:
        # =========================================================
        # FETCH DATA
        # =========================================================

        if start_date is None or end_date is None:

            data = yf.download(
                ticker,
                period=period,
                auto_adjust=False,
                progress=False,
                threads=False
            )

        else:

            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
                threads=False
            )

        # =========================================================
        # VALIDATE
        # =========================================================

        if data is None or data.empty:
            raise ValueError(
                f"No market data was returned for ticker '{ticker}'."
            )

        # =========================================================
        # HANDLE YFINANCE MULTIINDEX
        # =========================================================

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # =========================================================
        # RESET INDEX
        # =========================================================

        data = data.reset_index()

        # yfinance normally calls this "Date"
        if "Date" not in data.columns:

            if "Datetime" in data.columns:
                data = data.rename(
                    columns={"Datetime": "Date"}
                )

            else:
                raise ValueError(
                    "Market data does not contain a Date column."
                )

        # =========================================================
        # REQUIRED COLUMNS
        # =========================================================

        required_columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        missing = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Market data is missing columns: {missing}"
            )

        data = data[required_columns].copy()

        # =========================================================
        # CLEAN NUMERIC DATA
        # =========================================================

        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce"
        )

        numeric_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        for column in numeric_columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

        # =========================================================
        # REMOVE INVALID ROWS
        # =========================================================

        data = data.dropna(
            subset=[
                "Date",
                "Open",
                "High",
                "Low",
                "Close"
            ]
        )

        data = data.sort_values("Date")

        data = data.drop_duplicates(
            subset=["Date"],
            keep="last"
        )

        data = data.reset_index(drop=True)

        # =========================================================
        # FINAL VALIDATION
        # =========================================================

        if data.empty:
            raise ValueError(
                f"Market data for '{ticker}' contains no usable rows."
            )

        # =========================================================
        # METADATA
        # =========================================================

        data.attrs["ticker"] = ticker
        data.attrs["source"] = "Yahoo Finance"
        data.attrs["retrieved_at"] = datetime.now().isoformat()

        return data

    except ValueError:
        raise

    except Exception as exc:

        raise ValueError(
            f"Unable to retrieve market data for '{ticker}': {exc}"
        ) from exc