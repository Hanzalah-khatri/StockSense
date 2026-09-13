"""
StockSense Model Evaluation
---------------------------
Evaluates ML models using chronological time-series validation
and compares them against a naive zero-return baseline.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class ModelEvaluator:
    """
    Evaluates stock prediction models without shuffling
    the historical time series.
    """

    def __init__(self, test_size=0.2):

        if not 0 < test_size < 1:
            raise ValueError(
                "test_size must be between 0 and 1."
            )

        self.test_size = test_size


    # =========================================================
    # CHRONOLOGICAL SPLIT
    # =========================================================

    def split_data(self, X, y):

        split_index = int(
            len(X) * (1 - self.test_size)
        )

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )


    # =========================================================
    # DIRECTIONAL ACCURACY
    # =========================================================

    @staticmethod
    def directional_accuracy(
        y_actual,
        y_pred
    ):
        """
        Calculate directional accuracy for return prediction.

        Positive return  -> Up
        Negative return  -> Down
        Zero             -> Neutral

        Zero-return predictions are not counted as correct.
        """

        y_actual = np.asarray(y_actual)
        y_pred = np.asarray(y_pred)

        actual_direction = np.sign(
            y_actual
        )

        predicted_direction = np.sign(
            y_pred
        )

        # Ignore cases where the model predicts exactly zero.
        valid = predicted_direction != 0

        if not np.any(valid):
            return 0.0

        return (
            np.mean(
                actual_direction[valid]
                ==
                predicted_direction[valid]
            )
            * 100
        )


    # =========================================================
    # EVALUATE ONE MODEL
    # =========================================================

    def evaluate_model(
        self,
        model,
        X,
        y
    ):
        """
        Train on earlier observations and evaluate
        on later observations.
        """

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = self.split_data(
            X,
            y
        )

        # -----------------------------------------------------
        # Train
        # -----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # -----------------------------------------------------
        # Predict
        # -----------------------------------------------------

        predictions = model.predict(
            X_test
        )

        # -----------------------------------------------------
        # MAE
        # -----------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        # -----------------------------------------------------
        # RMSE
        # -----------------------------------------------------

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        # -----------------------------------------------------
        # R²
        # -----------------------------------------------------

        r2 = r2_score(
            y_test,
            predictions
        )

        # -----------------------------------------------------
        # Directional Accuracy
        # -----------------------------------------------------

        direction = self.directional_accuracy(
            y_test.to_numpy(),
            predictions
        )

        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),

            # Main metric
            "Directional Accuracy": float(
                direction
            ),

            # Frontend-friendly alias
            "Accuracy": float(
                direction
            )
        }


    # =========================================================
    # NAIVE BASELINE
    # =========================================================

    @staticmethod
    def evaluate_naive_baseline(y):
        """
        Naive baseline:

        Predict that tomorrow's return will be 0%.

        This provides a simple benchmark that the ML models
        should ideally outperform.
        """

        y = np.asarray(
            y
        )

        predictions = np.zeros_like(
            y,
            dtype=float
        )

        mae = mean_absolute_error(
            y,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y,
                predictions
            )
        )

        r2 = r2_score(
            y,
            predictions
        )

        # A zero-return prediction has no direction.
        direction = 0.0

        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "Directional Accuracy": float(
                direction
            ),
            "Accuracy": float(
                direction
            )
        }


    # =========================================================
    # EVALUATE ALL MODELS
    # =========================================================

    def evaluate_all(
        self,
        models,
        X,
        y
    ):
        """
        Evaluate all ML models and the naive baseline
        using the same chronological test set.
        """

        results = {}

        # -----------------------------------------------------
        # Evaluate ML models
        # -----------------------------------------------------

        for name, model in models.items():

            results[name] = self.evaluate_model(
                model,
                X,
                y
            )

        # -----------------------------------------------------
        # Evaluate naive baseline
        # -----------------------------------------------------

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = self.split_data(
            X,
            y
        )

        baseline_metrics = (
            self.evaluate_naive_baseline(
                y_test
            )
        )

        results[
            "Naive Baseline"
        ] = baseline_metrics

        return pd.DataFrame(
            results
        ).T


    # =========================================================
    # SELECT BEST MODEL
    # =========================================================

    @staticmethod
    def select_best_model(
        results
    ):
        """
        Select the best ML model.

        The Naive Baseline is excluded from model selection.

        Selection considers:

        1. Directional Accuracy
        2. MAE
        3. RMSE
        4. R²

        Directional accuracy receives additional weight because
        StockSense is ultimately trying to identify useful
        future market direction.
        """

        # -----------------------------------------------------
        # Remove baseline
        # -----------------------------------------------------

        model_results = results.drop(
            index="Naive Baseline",
            errors="ignore"
        ).copy()

        if model_results.empty:
            raise ValueError(
                "No ML models available for selection."
            )

        # -----------------------------------------------------
        # Rank each metric
        # -----------------------------------------------------

        model_results[
            "MAE_Rank"
        ] = model_results[
            "MAE"
        ].rank(
            ascending=True,
            method="min"
        )

        model_results[
            "RMSE_Rank"
        ] = model_results[
            "RMSE"
        ].rank(
            ascending=True,
            method="min"
        )

        model_results[
            "R2_Rank"
        ] = model_results[
            "R2"
        ].rank(
            ascending=False,
            method="min"
        )

        model_results[
            "Direction_Rank"
        ] = model_results[
            "Directional Accuracy"
        ].rank(
            ascending=False,
            method="min"
        )

        # -----------------------------------------------------
        # Weighted overall score
        #
        # Directional accuracy gets the highest weight.
        # -----------------------------------------------------

        model_results[
            "Overall Score"
        ] = (

            model_results[
                "Direction_Rank"
            ] * 2.0

            +

            model_results[
                "MAE_Rank"
            ] * 1.0

            +

            model_results[
                "RMSE_Rank"
            ] * 1.0

            +

            model_results[
                "R2_Rank"
            ] * 1.0
        )

        # -----------------------------------------------------
        # Best = lowest score
        # -----------------------------------------------------

        best_model = (
            model_results[
                "Overall Score"
            ]
            .idxmin()
        )

        return best_model