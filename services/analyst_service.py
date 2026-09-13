import pandas as pd


class StockAnalyst:
    """
    Deterministic AI-style stock analyst.

    Combines:
    - Technical indicators
    - Momentum
    - Trend structure
    - Bollinger Bands
    - Direction classifier probability

    This service does NOT make a separate ML prediction.
    The locked Random Forest classifier is the primary
    direction model.
    """

    def analyze(
        self,
        features: pd.DataFrame,
        direction_result=None
    ):
        if features is None or features.empty:
            raise ValueError(
                "No feature data available for analysis."
            )

        row = features.iloc[-1]

        # =================================================
        # CURRENT MARKET DATA
        # =================================================

        current_price = self._value(
            row,
            "Close"
        )

        sma_20 = self._value(
            row,
            "SMA_20"
        )

        sma_50 = self._value(
            row,
            "SMA_50"
        )

        sma_200 = self._value(
            row,
            "SMA_200"
        )

        rsi = self._value(
            row,
            "RSI"
        )

        macd = self._value(
            row,
            "MACD"
        )

        macd_signal = self._value(
            row,
            "MACD_Signal"
        )

        macd_histogram = self._value(
            row,
            "MACD_Histogram"
        )

        momentum_5 = self._value(
            row,
            "Momentum_5"
        )

        momentum_20 = self._value(
            row,
            "Momentum_20"
        )

        bb_upper = self._value(
            row,
            "BB_Upper"
        )

        bb_lower = self._value(
            row,
            "BB_Lower"
        )

        volatility = self._value(
            row,
            "Volatility"
        )

        # =================================================
        # SCORING
        # =================================================

        bullish_score = 0
        bearish_score = 0

        bullish_points = []
        bearish_points = []
        risks = []

        # -------------------------------------------------
        # SMA TREND
        # -------------------------------------------------

        if (
            sma_20 is not None
            and sma_50 is not None
        ):

            if sma_20 > sma_50:

                bullish_score += 10

                bullish_points.append(
                    "Short-term trend is above the medium-term trend."
                )

            else:

                bearish_score += 10

                bearish_points.append(
                    "Short-term trend is below the medium-term trend."
                )

        if (
            sma_50 is not None
            and sma_200 is not None
        ):

            if sma_50 > sma_200:

                bullish_score += 15

                bullish_points.append(
                    "Medium-term trend is above the long-term trend."
                )

            else:

                bearish_score += 15

                bearish_points.append(
                    "Medium-term trend is below the long-term trend."
                )

        # -------------------------------------------------
        # RSI
        # -------------------------------------------------

        if rsi is not None:

            if rsi < 30:

                bullish_score += 15

                bullish_points.append(
                    f"RSI is oversold at {rsi:.1f}, "
                    "which can support a rebound."
                )

            elif rsi > 70:

                bearish_score += 15

                bearish_points.append(
                    f"RSI is overbought at {rsi:.1f}, "
                    "which increases pullback risk."
                )

            elif rsi >= 50:

                bullish_score += 8

                bullish_points.append(
                    f"RSI is moderately bullish at {rsi:.1f}."
                )

            else:

                bearish_score += 5

                bearish_points.append(
                    f"RSI is below 50 at {rsi:.1f}."
                )

        # -------------------------------------------------
        # MACD
        # -------------------------------------------------

        if (
            macd is not None
            and macd_signal is not None
        ):

            if macd > macd_signal:

                bullish_score += 15

                bullish_points.append(
                    "MACD is above its signal line."
                )

            else:

                bearish_score += 15

                bearish_points.append(
                    "MACD is below its signal line."
                )

        if macd_histogram is not None:

            if macd_histogram > 0:

                bullish_score += 5

                bullish_points.append(
                    "MACD histogram is positive."
                )

            elif macd_histogram < 0:

                bearish_score += 5

                bearish_points.append(
                    "MACD histogram is negative."
                )

        # -------------------------------------------------
        # MOMENTUM
        # -------------------------------------------------

        if momentum_5 is not None:

            if momentum_5 > 0:

                bullish_score += 8

                bullish_points.append(
                    "Recent 5-day momentum is positive."
                )

            else:

                bearish_score += 8

                bearish_points.append(
                    "Recent 5-day momentum is negative."
                )

        if momentum_20 is not None:

            if momentum_20 > 0:

                bullish_score += 7

                bullish_points.append(
                    "20-day momentum remains positive."
                )

            else:

                bearish_score += 7

                bearish_points.append(
                    "20-day momentum is negative."
                )

        # -------------------------------------------------
        # BOLLINGER BANDS
        # -------------------------------------------------

        if (
            current_price is not None
            and bb_upper is not None
            and bb_lower is not None
        ):

            if current_price > bb_upper:

                bearish_score += 8

                bearish_points.append(
                    "Price is above the upper Bollinger Band."
                )

                risks.append(
                    "Price may be extended relative to recent volatility."
                )

            elif current_price < bb_lower:

                bullish_score += 8

                bullish_points.append(
                    "Price is below the lower Bollinger Band."
                )

                risks.append(
                    "Price is under pressure and volatility may remain elevated."
                )

        # =================================================
        # CLASSIFIER INTEGRATION
        # =================================================

        classifier_direction = None
        probability_up = None
        probability_down = None
        classifier_confidence = None

        if direction_result:

            classifier_direction = (
                direction_result.get(
                    "direction"
                )
            )

            probability_up = (
                direction_result.get(
                    "probability_up"
                )
            )

            probability_down = (
                direction_result.get(
                    "probability_down"
                )
            )

            classifier_confidence = (
                direction_result.get(
                    "confidence_percent"
                )
            )

            # ---------------------------------------------
            # Add classifier signal to analyst score
            # ---------------------------------------------

            if probability_up is not None:

                if probability_up >= 0.55:

                    bullish_score += 15

                    bullish_points.append(
                        f"Direction classifier favors UP "
                        f"({probability_up:.1%} probability)."
                    )

                elif probability_up <= 0.45:

                    bearish_score += 15

                    bearish_points.append(
                        f"Direction classifier favors DOWN "
                        f"({probability_down:.1%} probability)."
                    )

                else:

                    risks.append(
                        "Direction classifier is close to neutral."
                    )

        # =================================================
        # FINAL SIGNAL
        # =================================================

        total_score = (
            bullish_score
            + bearish_score
        )

        if total_score > 0:

            bullish_percentage = (
                bullish_score
                / total_score
                * 100
            )

        else:

            bullish_percentage = 50.0

        bearish_percentage = (
            100
            - bullish_percentage
        )

        difference = abs(
            bullish_percentage
            - bearish_percentage
        )

        if difference >= 30:

            signal_strength = "STRONG"

        elif difference >= 15:

            signal_strength = "MODERATE"

        else:

            signal_strength = "WEAK"

        if bullish_percentage >= 60:

            signal = "BULLISH"

        elif bearish_percentage >= 60:

            signal = "BEARISH"

        else:

            signal = "NEUTRAL"

        # =================================================
        # SUMMARY
        # =================================================

        if signal == "BULLISH":

            summary = (
                "Technical conditions currently lean bullish, "
                "with multiple indicators supporting upward momentum."
            )

        elif signal == "BEARISH":

            summary = (
                "Technical conditions currently lean bearish, "
                "with several indicators pointing toward downside pressure."
            )

        else:

            summary = (
                "Technical conditions are mixed, "
                "so the current market setup does not show a strong directional edge."
            )

        # =================================================
        # RISK ANALYSIS
        # =================================================

        if volatility is not None:

            if volatility >= 0.02:

                risks.append(
                    f"Recent volatility is elevated at "
                    f"{volatility:.2%}."
                )

            elif volatility >= 0.01:

                risks.append(
                    f"Market volatility is moderate at "
                    f"{volatility:.2%}."
                )

        # Remove duplicate risks
        risks = list(
            dict.fromkeys(
                risks
            )
        )

        # =================================================
        # TECHNICAL DATA
        # =================================================

        technical_indicators = {
            "price": self._round(
                current_price
            ),
            "rsi": self._round(
                rsi
            ),
            "sma_20": self._round(
                sma_20
            ),
            "sma_50": self._round(
                sma_50
            ),
            "sma_200": self._round(
                sma_200
            ),
            "macd": self._round(
                macd
            ),
            "macd_signal": self._round(
                macd_signal
            ),
            "macd_histogram": self._round(
                macd_histogram
            ),
            "momentum_5": self._round(
                momentum_5
            ),
            "momentum_20": self._round(
                momentum_20
            ),
            "volatility": self._round(
                volatility
            )
        }

        # =================================================
        # FINAL RESPONSE
        # =================================================

        return {
            "signal": signal,

            "signal_strength": signal_strength,

            "score": {
                "bullish": round(
                    bullish_percentage,
                    2
                ),
                "bearish": round(
                    bearish_percentage,
                    2
                )
            },

            "summary": summary,

            "bullish_points": bullish_points,

            "bearish_points": bearish_points,

            "reasons": (
                bullish_points
                + bearish_points
            ),

            "risks": risks,

            "direction_classifier": {
                "direction":
                    classifier_direction,

                "probability_up":
                    self._round(
                        probability_up
                    ),

                "probability_down":
                    self._round(
                        probability_down
                    ),

                "confidence":
                    self._round(
                        classifier_confidence
                    )
            },

            "technical_indicators":
                technical_indicators
        }

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _value(
        row,
        column
    ):

        if column not in row.index:
            return None

        value = row[column]

        if pd.isna(value):
            return None

        try:
            return float(value)

        except (
            TypeError,
            ValueError
        ):
            return None

    @staticmethod
    def _round(
        value,
        digits=4
    ):

        if value is None:
            return None

        return round(
            float(value),
            digits
        )