"""
predictor.py
------------
Machine-learning logic for forecasting future closing prices.

Approach:
  - Feature engineering: day index, rolling moving averages (5-day, 10-day)
    as momentum/trend features.
  - Model: scikit-learn LinearRegression trained on the engineered features
    against the closing price. Simple and fast, chosen deliberately for
    interpretability (matches the "explain your model" submission
    requirement) over a black-box model.
  - Confidence: derived from the model's R^2 score on a held-out slice of
    the historical data (last 20% of days), mapped into a 0-100% indicator.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["day_index"] = np.arange(len(data))
    data["ma_5"] = data["Close"].rolling(window=5, min_periods=1).mean()
    data["ma_10"] = data["Close"].rolling(window=10, min_periods=1).mean()
    return data


def predict_future_prices(df: pd.DataFrame, days_ahead: int = 7):
    """
    Trains a linear regression model on historical closing prices and
    forecasts `days_ahead` future trading days.

    Returns a dict with:
      - future_dates: list of ISO date strings
      - predicted_prices: list of floats
      - confidence: float (0-100), higher = model fit historical data better
      - trend: "up" | "down" | "flat"
    """
    data = _build_features(df)
    feature_cols = ["day_index", "ma_5", "ma_10"]

    X = data[feature_cols].values
    y = data["Close"].values

    # Hold out the last 20% of points to estimate how well the model
    # generalizes -> used purely as a confidence proxy.
    split = max(int(len(X) * 0.8), 1)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    if len(X_test) >= 2:
        y_pred_test = model.predict(X_test)
        r2 = r2_score(y_test, y_pred_test)
        confidence = max(0.0, min(100.0, (r2 + 1) / 2 * 100))  # map [-1,1] -> [0,100]
    else:
        confidence = 50.0  # not enough data to score, neutral confidence

    # Refit on ALL available data before forecasting forward
    model.fit(X, y)

    last_index = data["day_index"].iloc[-1]
    last_ma5 = data["ma_5"].iloc[-1]
    last_ma10 = data["ma_10"].iloc[-1]
    last_close = data["Close"].iloc[-1]
    last_date = pd.to_datetime(data["Date"].iloc[-1])

    future_dates = []
    predicted_prices = []
    running_closes = list(data["Close"].values[-10:])  # for rolling MA continuation

    for step in range(1, days_ahead + 1):
        idx = last_index + step
        ma5 = np.mean(running_closes[-5:])
        ma10 = np.mean(running_closes[-10:])
        pred = model.predict([[idx, ma5, ma10]])[0]
        pred = float(max(pred, 0.01))  # price can't go negative

        next_date = last_date + pd.tseries.offsets.BDay(step)
        future_dates.append(next_date.strftime("%Y-%m-%d"))
        predicted_prices.append(round(pred, 2))

        running_closes.append(pred)

    first_pred = predicted_prices[0]
    last_pred = predicted_prices[-1]
    if last_pred > last_close * 1.01:
        trend = "up"
    elif last_pred < last_close * 0.99:
        trend = "down"
    else:
        trend = "flat"

    return {
        "future_dates": future_dates,
        "predicted_prices": predicted_prices,
        "confidence": round(confidence, 1),
        "trend": trend,
        "last_actual_close": round(float(last_close), 2),
    }
