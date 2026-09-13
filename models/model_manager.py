"""
StockSense Model Manager
------------------------
Handles training, prediction, and feature importance
analysis for multiple machine-learning models.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class ModelManager:
    """
    Manages multiple regression models used by StockSense.
    """

    def __init__(self):
        self.models = {
            "Ridge": Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0))
            ]),

            "Random Forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),

            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        }

        self.trained_models = {}

    # =========================================================
    # TRAIN
    # =========================================================

    def train_model(self, name, X, y):
        """
        Train a single model.
        """

        if name not in self.models:
            raise ValueError(f"Unknown model: {name}")

        model = self.models[name]

        model.fit(X, y)

        self.trained_models[name] = model

        return model

    # =========================================================
    # TRAIN ALL
    # =========================================================

    def train_all(self, X, y):
        """
        Train every available model.
        """

        for name in self.models:
            self.train_model(name, X, y)

        return self.trained_models

    # =========================================================
    # PREDICT
    # =========================================================

    def predict(self, name, X):
        """
        Generate predictions using a trained model.
        """

        if name not in self.trained_models:
            raise ValueError(
                f"Model '{name}' has not been trained."
            )

        return self.trained_models[name].predict(X)

    # =========================================================
    # PREDICT ALL
    # =========================================================

    def predict_all(self, X):
        """
        Generate predictions from every trained model.
        """

        predictions = {}

        for name in self.trained_models:
            predictions[name] = self.predict(name, X)

        return predictions

    # =========================================================
    # FEATURE IMPORTANCE
    # =========================================================

    def get_feature_importance(self, name, feature_names):
        """
        Return feature importance or coefficient strength
        for a trained model.

        Tree models:
            Uses feature_importances_

        Ridge:
            Uses absolute standardized coefficients.
        """

        if name not in self.trained_models:
            raise ValueError(
                f"Model '{name}' has not been trained."
            )

        model = self.trained_models[name]

        # -----------------------------------------------------
        # Ridge Pipeline
        # -----------------------------------------------------

        if name == "Ridge":

            ridge_model = model.named_steps["model"]

            coefficients = ridge_model.coef_

            importance = np.abs(coefficients)

        # -----------------------------------------------------
        # Tree-Based Models
        # -----------------------------------------------------

        elif hasattr(model, "feature_importances_"):

            importance = model.feature_importances_

        else:
            raise ValueError(
                f"Feature importance is not supported for '{name}'."
            )

        result = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })

        result = result.sort_values(
            by="Importance",
            ascending=False
        ).reset_index(drop=True)

        return result

    # =========================================================
    # ALL FEATURE IMPORTANCE
    # =========================================================

    def get_all_feature_importance(self, feature_names):
        """
        Return feature importance for every trained model.
        """

        results = {}

        for name in self.trained_models:

            try:
                results[name] = self.get_feature_importance(
                    name,
                    feature_names
                )

            except ValueError:
                continue

        return results

    # =========================================================
    # MODEL NAMES
    # =========================================================

    def get_model_names(self):
        """Return available model names."""

        return list(self.models.keys())