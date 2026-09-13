"""
StockSense Feature Engineering
------------------------------
Creates machine-learning features from historical OHLCV data.
"""

import numpy as np
import pandas as pd


class FeatureEngineering:
    """
    Generates technical and statistical features for ML models.
    """

    def __init__(self, dataframe: pd.DataFrame):

        if dataframe is None or dataframe.empty:
            raise ValueError(
                "Cannot engineer features from empty data."
            )

        self.df = dataframe.copy()

    # =========================================================
    # PRICE FEATURES
    # =========================================================

    def add_price_features(self):

        self.df["Daily_Return"] = (
            self.df["Close"].pct_change()
        )

        self.df["Price_Change"] = (
            self.df["Close"] - self.df["Open"]
        )

        self.df["High_Low_Range"] = (
            self.df["High"] - self.df["Low"]
        )

        self.df["High_Low_Pct"] = (
            (self.df["High"] - self.df["Low"])
            / self.df["Close"]
        )

        return self

    # =========================================================
    # MOVING AVERAGES
    # =========================================================

    def add_moving_averages(self):

        self.df["SMA_5"] = (
            self.df["Close"].rolling(5).mean()
        )

        self.df["SMA_10"] = (
            self.df["Close"].rolling(10).mean()
        )

        self.df["SMA_20"] = (
            self.df["Close"].rolling(20).mean()
        )

        self.df["SMA_50"] = (
            self.df["Close"].rolling(50).mean()
        )

        self.df["SMA_200"] = (
            self.df["Close"].rolling(200).mean()
        )

        self.df["EMA_12"] = (
            self.df["Close"]
            .ewm(span=12, adjust=False)
            .mean()
        )

        self.df["EMA_26"] = (
            self.df["Close"]
            .ewm(span=26, adjust=False)
            .mean()
        )

        return self

    # =========================================================
    # RSI
    # =========================================================

    def add_rsi(self, period=14):

        delta = self.df["Close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)

        self.df["RSI"] = (
            100 - (100 / (1 + rs))
        )

        return self

    # =========================================================
    # MACD
    # =========================================================

    def add_macd(self):

        ema_12 = (
            self.df["Close"]
            .ewm(span=12, adjust=False)
            .mean()
        )

        ema_26 = (
            self.df["Close"]
            .ewm(span=26, adjust=False)
            .mean()
        )

        self.df["MACD"] = ema_12 - ema_26

        self.df["MACD_Signal"] = (
            self.df["MACD"]
            .ewm(span=9, adjust=False)
            .mean()
        )

        self.df["MACD_Histogram"] = (
            self.df["MACD"]
            - self.df["MACD_Signal"]
        )

        return self

    # =========================================================
    # BOLLINGER BANDS
    # =========================================================

    def add_bollinger_bands(self, period=20):

        middle = (
            self.df["Close"]
            .rolling(period)
            .mean()
        )

        std = (
            self.df["Close"]
            .rolling(period)
            .std()
        )

        self.df["BB_Middle"] = middle

        self.df["BB_Upper"] = (
            middle + (2 * std)
        )

        self.df["BB_Lower"] = (
            middle - (2 * std)
        )

        band_width = (
            self.df["BB_Upper"]
            - self.df["BB_Lower"]
        )

        self.df["BB_Position"] = (
            (self.df["Close"] - self.df["BB_Lower"])
            / band_width.replace(0, np.nan)
        )

        return self

    # =========================================================
    # VOLATILITY
    # =========================================================

    def add_volatility(self, period=20):

        self.df["Volatility"] = (
            self.df["Daily_Return"]
            .rolling(period)
            .std()
        )

        return self

    # =========================================================
    # MOMENTUM
    # =========================================================

    def add_momentum(self):

        self.df["Momentum_5"] = (
            self.df["Close"].pct_change(5)
        )

        self.df["Momentum_10"] = (
            self.df["Close"].pct_change(10)
        )

        self.df["Momentum_20"] = (
            self.df["Close"].pct_change(20)
        )

        return self

    # =========================================================
    # VOLUME FEATURES
    # =========================================================

    def add_volume_features(self):

        if "Volume" not in self.df.columns:
            return self

        self.df["Volume_Change"] = (
            self.df["Volume"].pct_change()
        )

        self.df["Volume_SMA_20"] = (
            self.df["Volume"]
            .rolling(20)
            .mean()
        )

        self.df["Relative_Volume"] = (
            self.df["Volume"]
            / self.df["Volume_SMA_20"].replace(0, np.nan)
        )

        return self

    # =========================================================
    # LAG FEATURES
    # =========================================================

    def add_lag_features(self):

        for lag in [1, 2, 3, 5, 10]:

            self.df[f"Return_Lag_{lag}"] = (
                self.df["Daily_Return"]
                .shift(lag)
            )

            self.df[f"Close_Lag_{lag}"] = (
                self.df["Close"]
                .shift(lag)
            )

        return self

    # =========================================================
    # COMPLETE PIPELINE
    # =========================================================

    def build_features(self):

        self.add_price_features()
        self.add_moving_averages()
        self.add_rsi()
        self.add_macd()
        self.add_bollinger_bands()
        self.add_volatility()
        self.add_momentum()
        self.add_volume_features()
        self.add_lag_features()

        # Replace infinite values
        self.df = self.df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Only drop rows after ALL features are created
        self.df = self.df.dropna().copy()

        if self.df.empty:
            raise ValueError(
                "Feature engineering produced no usable data. "
                "At least 1 year of historical market data is recommended."
            )

        return self.df