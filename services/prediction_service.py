# =========================================================
# StockSense — Prediction Service
# Handles feature preparation, model evaluation,
# training, and recursive return-based forecasting.
# =========================================================


import pandas as pd

from models.feature_engineering import FeatureEngineering
from models.model_evaluation import ModelEvaluator
from models.model_manager import ModelManager


class PredictionService:

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(self):

        self.model_manager = ModelManager()

        self.evaluator = ModelEvaluator(
            test_size=0.2
        )


    # =====================================================
    # FEATURE PREPARATION
    # =====================================================

    def prepare_features(self, market_data):

        feature_engineer = FeatureEngineering(
            market_data
        )

        features = feature_engineer.build_features()

        if features.empty:

            raise ValueError(
                "Feature engineering produced no usable data."
            )

        return features


    # =====================================================
    # ML FEATURE COLUMNS
    # =====================================================
    

    @staticmethod
    def get_feature_columns():

        return [

            "Daily_Return",
            "Price_Change",
            "High_Low_Range",
            "High_Low_Pct",

            "SMA_5",
            "SMA_10",
            "SMA_20",
            "SMA_50",
            "SMA_200",

            "EMA_12",
            "EMA_26",

            "RSI",

            "MACD",
            "MACD_Signal",
            "MACD_Histogram",

            "BB_Middle",
            "BB_Upper",
            "BB_Lower",
            "BB_Position",

            "Volatility",

            "Momentum_5",
            "Momentum_10",
            "Momentum_20",

            "Volume_Change",
            "Relative_Volume",

            "Return_Lag_1",
            "Return_Lag_2",
            "Return_Lag_3",
            "Return_Lag_5",
            "Return_Lag_10",

        ]


    # =====================================================
    # SELECTED FEATURE SET
    # =====================================================

    @staticmethod
    def get_selected_feature_columns():

        return [

            # Price / volatility
            "Daily_Return",
            "High_Low_Range",
            "High_Low_Pct",

            # Short-term trend
            "SMA_5",
            "SMA_10",
            "SMA_20",

            # Momentum
            "Momentum_5",

            # Volatility
            "Volatility",

            # Recent return behavior
            "Return_Lag_1",
            "Return_Lag_3",
            "Return_Lag_5",
            "Return_Lag_10",

            # Volume
            "Volume_Change",
            "Relative_Volume",

            # Bollinger position
            "BB_Position",
        ]
    # =====================================================
    # PREPARE ML DATA
    # =====================================================

    def prepare_ml_data(self, features):

        """
        Prepare ML data for next-day return prediction.

        Features from Day T are used to predict the
        percentage return from Day T to Day T+1.

        Example:

        Day T Close       = 100
        Day T+1 Close     = 102

        Target return     = 0.02
        Target return     = 2%
        """

        feature_columns = (
            self.get_feature_columns()
        )


        # -------------------------------------------------
        # Validate required features
        # -------------------------------------------------

        missing = [

            column

            for column in feature_columns

            if column not in features.columns

        ]


        if missing:

            raise ValueError(
                f"Missing ML features: {missing}"
            )


        # -------------------------------------------------
        # Features from current day
        # -------------------------------------------------

        X = features[
            feature_columns
        ].copy()


        # -------------------------------------------------
        # NEXT-DAY RETURN TARGET
        #
        # Return = (Tomorrow Close / Today Close) - 1
        # -------------------------------------------------

        y = (
            features["Close"]
            .shift(-1)
            .div(features["Close"])
            - 1
        )


        # -------------------------------------------------
        # Remove final row because it has no next-day target
        # -------------------------------------------------

        valid_rows = y.notna()


        X = X.loc[
            valid_rows
        ].copy()


        y = y.loc[
            valid_rows
        ].copy()


        return X, y


    # =====================================================
    # MODEL EVALUATION
    # =====================================================

    def evaluate_models(
        self,
        X,
        y
    ):

        results = self.evaluator.evaluate_all(

            self.model_manager.models,
            X,
            y

        )


        best_model = (
            self.evaluator.select_best_model(
                results
            )
        )


        return results, best_model


    # =====================================================
    # TRAIN BEST MODEL
    # =====================================================

    def train_best_model(
        self,
        best_model,
        X,
        y
    ):

        self.model_manager.train_model(
            best_model,
            X,
            y
        )


        return (
            self.model_manager.trained_models[
                best_model
            ]
        )


    # =====================================================
    # CREATE FUTURE OHLCV ROW
    # =====================================================

    @staticmethod
    def create_future_row(
        previous_row,
        predicted_close,
        future_date
    ):

        # -------------------------------------------------
        # Use previous close as synthetic future open
        # -------------------------------------------------

        previous_close = float(
            previous_row["Close"]
        )


        future_open = previous_close


        # -------------------------------------------------
        # Synthetic high / low
        # -------------------------------------------------

        future_high = max(
            future_open,
            predicted_close
        )


        future_low = min(
            future_open,
            predicted_close
        )


        # -------------------------------------------------
        # Keep latest known volume as proxy
        # -------------------------------------------------

        future_volume = float(
            previous_row["Volume"]
        )


        return pd.DataFrame({

            "Date": [
                future_date
            ],

            "Open": [
                future_open
            ],

            "High": [
                future_high
            ],

            "Low": [
                future_low
            ],

            "Close": [
                predicted_close
            ],

            "Volume": [
                future_volume
            ]

        })


    # =====================================================
    # RECURSIVE RETURN-BASED FORECASTING
    # =====================================================

    def predict_future(
        self,
        model,
        market_data,
        days_ahead
    ):

        """
        Predict future prices recursively.

        The model predicts the next-day RETURN.

        The predicted return is then converted into
        a predicted closing price.

        Example:

        Current Close = 100
        Predicted Return = 0.02

        Predicted Close =
            100 * (1 + 0.02)
            = 102
        """


        # -------------------------------------------------
        # Work on a copy so original data is not modified
        # -------------------------------------------------

        working_data = (
            market_data.copy()
        )


        # -------------------------------------------------
        # Validate required columns
        # -------------------------------------------------

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

            if column not in working_data.columns

        ]


        if missing:

            raise ValueError(
                f"Missing market columns: {missing}"
            )


        predictions = []


        # =================================================
        # FORECAST ONE DAY AT A TIME
        # =================================================

        for _ in range(days_ahead):


            # ---------------------------------------------
            # Rebuild features using current data
            # ---------------------------------------------

            features = self.prepare_features(
                working_data
            )


            # ---------------------------------------------
            # Latest available feature row
            # ---------------------------------------------

            latest_row = (
                features
                .iloc[-1:]
                .copy()
            )


            # ---------------------------------------------
            # Current real/synthetic close
            # ---------------------------------------------

            current_close = float(
                latest_row["Close"].iloc[0]
            )


            # ---------------------------------------------
            # Predict NEXT-DAY RETURN
            # ---------------------------------------------

            predicted_return = float(

                model.predict(
                    latest_row[
                        self.get_feature_columns()
                    ]
                )[0]

            )


            # ---------------------------------------------
            # Convert return into predicted price
            #
            # predicted_close =
            # current_close * (1 + predicted_return)
            # ---------------------------------------------

            predicted_close = (

                current_close
                * (1 + predicted_return)

            )


            # ---------------------------------------------
            # Safety check
            # ---------------------------------------------

            if not pd.notna(
                predicted_close
            ):

                raise ValueError(
                    "Model produced an invalid future prediction."
                )


            predicted_close = float(
                predicted_close
            )


            # ---------------------------------------------
            # Store predicted price
            # ---------------------------------------------

            predictions.append(
                predicted_close
            )


            # ---------------------------------------------
            # Determine next business day
            # ---------------------------------------------

            last_date = pd.Timestamp(
                working_data["Date"].iloc[-1]
            )


            future_date = (
                last_date
                + pd.tseries.offsets.BDay(1)
            )


            # ---------------------------------------------
            # Create synthetic future market row
            # ---------------------------------------------

            previous_row = (
                working_data.iloc[-1]
            )


            future_row = self.create_future_row(

                previous_row,

                predicted_close,

                future_date

            )


            # ---------------------------------------------
            # Add future row to working dataset
            # ---------------------------------------------

            working_data = pd.concat(

                [
                    working_data,
                    future_row
                ],

                ignore_index=True

            )


        return predictions


    # =====================================================
    # COMPLETE PREDICTION PIPELINE
    # =====================================================

    def run_prediction(
        self,
        market_data,
        days_ahead=7
    ):

        # =================================================
        # 1. FEATURE ENGINEERING
        # =================================================

        features = self.prepare_features(
            market_data
        )


        # =================================================
        # 2. PREPARE ML DATA
        # =================================================

        X, y = self.prepare_ml_data(
            features
        )

        # =================================================
        # 4. VALIDATE DATASET
        # =================================================

        if len(X) < 30:

            raise ValueError(
                "Not enough usable data for ML prediction."
            )


        # =================================================
        # 5. EVALUATE ALL MODELS
        # =================================================

        results, best_model = (
            self.evaluate_models(
                X,
                y
            )
        )


        # =================================================
        # 6. TRAIN BEST MODEL ON FULL DATA
        # =================================================

        model = self.train_best_model(

            best_model,

            X,

            y

        )


        # =================================================
        # 7. PREDICT FUTURE PRICES
        # =================================================

        predictions = self.predict_future(

            model,

            market_data,

            days_ahead

        )


        # =================================================
        # 8. RETURN RESULTS
        # =================================================

        return {

            "predictions": predictions,

            "best_model": best_model,

            "evaluation": results,

            "features": features

        }