import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


class ValidationService:
    """
    Walk-forward validation for the StockSense direction classifier.

    The model is repeatedly trained on historical observations and
    evaluated only on later unseen observations.

    This service is intentionally separate from the production
    ClassifierService. It is used to measure historical reliability
    rather than to generate the live production prediction.
    """

    def __init__(self):
        pass

    # =========================================================
    # FEATURES
    # =========================================================

    @staticmethod
    def get_feature_columns():
        """
        Features used by the validation model.

        These must remain aligned with the feature engineering
        pipeline used to generate the validation dataset.
        """

        return [
            "Daily_Return",
            "High_Low_Range",
            "High_Low_Pct",
            "SMA_5",
            "SMA_10",
            "SMA_20",
            "Momentum_5",
            "Volatility",
            "Return_Lag_1",
            "Return_Lag_3",
            "Return_Lag_5",
            "Return_Lag_10",
            "Volume_Change",
            "Relative_Volume",
            "BB_Position",
        ]

    # =========================================================
    # TARGET
    # =========================================================

    @staticmethod
    def create_target(features):
        """
        Create the next-day direction target.

        1 = UP
        0 = DOWN

        The final observation is removed because it does not have
        a known next-day closing price.
        """

        target = (
            features["Close"]
            .shift(-1)
            .gt(features["Close"])
            .astype(int)
        )

        return target.iloc[:-1]

    # =========================================================
    # DATA PREPARATION
    # =========================================================

    def prepare_data(self, features):
        """
        Prepare validation features and target.

        Rows containing missing feature values are removed.
        X and y are kept perfectly aligned.
        """

        feature_columns = self.get_feature_columns()

        # -----------------------------------------------------
        # Validate required columns
        # -----------------------------------------------------

        missing = [
            column
            for column in feature_columns
            if column not in features.columns
        ]

        if missing:
            raise ValueError(
                f"Missing validation features: {missing}"
            )

        if "Close" not in features.columns:
            raise ValueError(
                "Missing required target column: Close"
            )

        # -----------------------------------------------------
        # Feature matrix
        # -----------------------------------------------------

        X = features[
            feature_columns
        ].copy()

        # -----------------------------------------------------
        # Target
        # -----------------------------------------------------

        y = self.create_target(
            features
        )

        # -----------------------------------------------------
        # Remove final X row to align with target
        # -----------------------------------------------------

        X = X.iloc[:-1].copy()

        # -----------------------------------------------------
        # Remove invalid rows
        # -----------------------------------------------------

        valid_rows = (
            X.notna().all(axis=1)
            & y.notna()
        )

        X = X.loc[
            valid_rows
        ].copy()

        y = y.loc[
            valid_rows
        ].copy()

        # -----------------------------------------------------
        # Minimum dataset requirement
        # -----------------------------------------------------

        if len(X) < 100:
            raise ValueError(
                "Not enough usable observations for validation."
            )

        # -----------------------------------------------------
        # Ensure target is integer
        # -----------------------------------------------------

        y = y.astype(int)

        return X, y

    # =========================================================
    # MODEL
    # =========================================================

    @staticmethod
    def create_model():
        """
        Create the validation Random Forest.

        This configuration matches the established StockSense
        validation configuration.
        """

        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

    # =========================================================
    # WALK-FORWARD VALIDATION
    # =========================================================

    def walk_forward_validate(
        self,
        X,
        y,
        n_splits=5,
        min_train_size=150,
    ):
        """
        Perform expanding-window walk-forward validation.

        Example:

            Fold 1:
                Train → 150
                Test  → 30

            Fold 2:
                Train → 180
                Test  → 30

            Fold 3:
                Train → 210
                Test  → 30

            ...

        No future observations are used to train an earlier fold.
        """

        total_rows = len(X)

        if total_rows <= (
            min_train_size + n_splits
        ):
            raise ValueError(
                "Dataset is too small for requested "
                "walk-forward validation."
            )

        # -----------------------------------------------------
        # Determine test window
        # -----------------------------------------------------

        test_size = (
            total_rows - min_train_size
        ) // n_splits

        if test_size < 1:
            raise ValueError(
                "Unable to create validation windows."
            )

        fold_results = []

        all_actual = []
        all_predictions = []
        all_probabilities = []

        # -----------------------------------------------------
        # Walk-forward folds
        # -----------------------------------------------------

        for fold in range(n_splits):

            train_end = (
                min_train_size
                + fold * test_size
            )

            if fold == n_splits - 1:
                test_end = total_rows
            else:
                test_end = (
                    train_end
                    + test_size
                )

            X_train = X.iloc[
                :train_end
            ]

            y_train = y.iloc[
                :train_end
            ]

            X_test = X.iloc[
                train_end:test_end
            ]

            y_test = y.iloc[
                train_end:test_end
            ]

            if X_test.empty:
                continue

            # -------------------------------------------------
            # Train fresh model for this fold
            # -------------------------------------------------

            model = self.create_model()

            model.fit(
                X_train,
                y_train,
            )

            # -------------------------------------------------
            # Predictions
            # -------------------------------------------------

            predictions = model.predict(
                X_test
            )

            probabilities = (
                model.predict_proba(
                    X_test
                )[:, 1]
            )

            # -------------------------------------------------
            # Fold metrics
            # -------------------------------------------------

            fold_accuracy = accuracy_score(
                y_test,
                predictions,
            )

            fold_precision = precision_score(
                y_test,
                predictions,
                zero_division=0,
            )

            fold_recall = recall_score(
                y_test,
                predictions,
                zero_division=0,
            )

            fold_f1 = f1_score(
                y_test,
                predictions,
                zero_division=0,
            )

            try:
                fold_auc = roc_auc_score(
                    y_test,
                    probabilities,
                )
            except ValueError:
                fold_auc = None

            # -------------------------------------------------
            # Store fold result
            # -------------------------------------------------

            fold_results.append(
                {
                    "fold": fold + 1,

                    "train_rows": int(
                        len(X_train)
                    ),

                    "test_rows": int(
                        len(X_test)
                    ),

                    "accuracy": float(
                        fold_accuracy
                    ),

                    "precision": float(
                        fold_precision
                    ),

                    "recall": float(
                        fold_recall
                    ),

                    "f1": float(
                        fold_f1
                    ),

                    "roc_auc": (
                        float(fold_auc)
                        if fold_auc is not None
                        else None
                    ),
                }
            )

            # -------------------------------------------------
            # Store out-of-sample predictions
            # -------------------------------------------------

            all_actual.extend(
                y_test.tolist()
            )

            all_predictions.extend(
                predictions.tolist()
            )

            all_probabilities.extend(
                probabilities.tolist()
            )

        # -----------------------------------------------------
        # Convert to NumPy arrays
        # -----------------------------------------------------

        actual = np.asarray(
            all_actual,
            dtype=int,
        )

        predictions = np.asarray(
            all_predictions,
            dtype=int,
        )

        probabilities = np.asarray(
            all_probabilities,
            dtype=float,
        )

        # -----------------------------------------------------
        # Integrity check
        # -----------------------------------------------------

        total_test_rows = sum(
            fold["test_rows"]
            for fold in fold_results
        )

        if total_test_rows != len(actual):
            raise RuntimeError(
                "Validation integrity error: fold test rows "
                "do not match collected validation observations."
            )

        if len(actual) != len(predictions):
            raise RuntimeError(
                "Validation integrity error: actual and "
                "prediction arrays have different lengths."
            )

        if len(actual) != len(probabilities):
            raise RuntimeError(
                "Validation integrity error: actual and "
                "probability arrays have different lengths."
            )

        return (
            fold_results,
            actual,
            predictions,
            probabilities,
        )

    # =========================================================
    # AGGREGATE METRICS
    # =========================================================

    @staticmethod
    def calculate_metrics(
        actual,
        predictions,
        probabilities,
    ):
        """
        Calculate aggregate metrics using ONLY the
        out-of-sample validation observations.
        """

        actual = np.asarray(
            actual
        )

        predictions = np.asarray(
            predictions
        )

        probabilities = np.asarray(
            probabilities
        )

        if not (
            len(actual)
            == len(predictions)
            == len(probabilities)
        ):
            raise ValueError(
                "Metric calculation requires actual values, "
                "predictions, and probabilities to have "
                "the same length."
            )

        if len(actual) == 0:
            raise ValueError(
                "Cannot calculate metrics with no observations."
            )

        # -----------------------------------------------------
        # Classification metrics
        # -----------------------------------------------------

        accuracy = accuracy_score(
            actual,
            predictions,
        )

        precision = precision_score(
            actual,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            actual,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            actual,
            predictions,
            zero_division=0,
        )

        # -----------------------------------------------------
        # ROC-AUC
        # -----------------------------------------------------

        try:
            roc_auc = roc_auc_score(
                actual,
                probabilities,
            )
        except ValueError:
            roc_auc = None

        # -----------------------------------------------------
        # Confusion matrix
        # -----------------------------------------------------

        matrix = confusion_matrix(
            actual,
            predictions,
            labels=[0, 1],
        )

        true_negative = int(
            matrix[0][0]
        )

        false_positive = int(
            matrix[0][1]
        )

        false_negative = int(
            matrix[1][0]
        )

        true_positive = int(
            matrix[1][1]
        )

        confusion_total = (
            true_negative
            + false_positive
            + false_negative
            + true_positive
        )

        if confusion_total != len(actual):
            raise RuntimeError(
                "Validation integrity error: confusion matrix "
                "total does not match validation observations."
            )

        return {
            "accuracy": float(
                accuracy
            ),

            "precision": float(
                precision
            ),

            "recall": float(
                recall
            ),

            "f1": float(
                f1
            ),

            "roc_auc": (
                float(roc_auc)
                if roc_auc is not None
                else None
            ),

            "confusion_matrix": {
                "true_negative":
                    true_negative,

                "false_positive":
                    false_positive,

                "false_negative":
                    false_negative,

                "true_positive":
                    true_positive,
            },
        }

    # =========================================================
    # BASELINE
    # =========================================================

    @staticmethod
    def calculate_baseline(actual):
        """
        Calculate a majority-class baseline using EXACTLY the
        same out-of-sample observations used by the model.

        This prevents an invalid comparison between the model
        and a baseline calculated over a different dataset.
        """

        actual = np.asarray(
            actual
        )

        if len(actual) == 0:
            raise ValueError(
                "Cannot calculate baseline with no observations."
            )

        # -----------------------------------------------------
        # Class counts
        # -----------------------------------------------------

        down_count = int(
            np.sum(actual == 0)
        )

        up_count = int(
            np.sum(actual == 1)
        )

        # -----------------------------------------------------
        # Majority class
        # -----------------------------------------------------

        if up_count >= down_count:

            majority_class = 1
            majority_label = "UP"

        else:

            majority_class = 0
            majority_label = "DOWN"

        # -----------------------------------------------------
        # Baseline predictions
        # -----------------------------------------------------

        baseline_predictions = np.full(
            len(actual),
            majority_class,
            dtype=int,
        )

        baseline_accuracy = accuracy_score(
            actual,
            baseline_predictions,
        )

        # -----------------------------------------------------
        # Integrity check
        # -----------------------------------------------------

        if len(baseline_predictions) != len(actual):
            raise RuntimeError(
                "Baseline integrity error: baseline prediction "
                "count does not match validation observations."
            )

        return {
            "strategy":
                "Always predict the majority class",

            "majority_class":
                majority_label,

            "accuracy":
                float(
                    baseline_accuracy
                ),

            "observations":
                int(
                    len(actual)
                ),

            "up_count":
                up_count,

            "down_count":
                down_count,
        }

    # =========================================================
    # CLASS DISTRIBUTION
    # =========================================================

    @staticmethod
    def calculate_class_distribution(actual):
        """
        Calculate UP/DOWN distribution on the same validation
        observations used by the model.
        """

        actual = np.asarray(
            actual
        )

        total = len(actual)

        if total == 0:
            raise ValueError(
                "Cannot calculate class distribution "
                "with no observations."
            )

        up_count = int(
            np.sum(actual == 1)
        )

        down_count = int(
            np.sum(actual == 0)
        )

        return {
            "UP": up_count,
            "DOWN": down_count,
        }, {
            "UP": float(
                up_count / total
            ),
            "DOWN": float(
                down_count / total
            ),
        }

    # =========================================================
    # FOLD DIAGNOSTICS
    # =========================================================

    @staticmethod
    def calculate_fold_diagnostics(
        fold_results
    ):
        """
        Identify the strongest and weakest validation folds
        and summarize consistency across folds.
        """

        if not fold_results:
            return {
                "strongest_fold": None,
                "weakest_fold": None,
                "average_accuracy": None,
                "accuracy_std": None,
                "accuracy_range": None,
            }

        accuracies = np.asarray(
            [
                fold["accuracy"]
                for fold in fold_results
            ],
            dtype=float,
        )

        strongest = max(
            fold_results,
            key=lambda fold:
                fold["accuracy"]
        )

        weakest = min(
            fold_results,
            key=lambda fold:
                fold["accuracy"]
        )

        return {
            "strongest_fold": {
                "fold": int(
                    strongest["fold"]
                ),
                "accuracy": float(
                    strongest["accuracy"]
                ),
            },

            "weakest_fold": {
                "fold": int(
                    weakest["fold"]
                ),
                "accuracy": float(
                    weakest["accuracy"]
                ),
            },

            "average_accuracy": float(
                np.mean(accuracies)
            ),

            "accuracy_std": float(
                np.std(
                    accuracies
                )
            ),

            "accuracy_range": float(
                np.max(accuracies)
                - np.min(accuracies)
            ),
        }

    # =========================================================
    # VALIDATION INTEGRITY
    # =========================================================

    @staticmethod
    def calculate_integrity(
        actual,
        predictions,
        fold_results,
        baseline,
        metrics,
    ):
        """
        Verify that every major validation quantity is based
        on the same out-of-sample observation set.
        """

        observations = len(
            actual
        )

        prediction_count = len(
            predictions
        )

        fold_test_rows = sum(
            fold["test_rows"]
            for fold in fold_results
        )

        confusion = (
            metrics["confusion_matrix"]
        )

        confusion_total = (
            confusion["true_negative"]
            + confusion["false_positive"]
            + confusion["false_negative"]
            + confusion["true_positive"]
        )

        baseline_observations = int(
            baseline["observations"]
        )

        checks = {
            "observations_match_predictions":
                observations
                == prediction_count,

            "observations_match_folds":
                observations
                == fold_test_rows,

            "observations_match_confusion_matrix":
                observations
                == confusion_total,

            "observations_match_baseline":
                observations
                == baseline_observations,
        }

        passed = all(
            checks.values()
        )

        if not passed:
            failed_checks = [
                name
                for name, result
                in checks.items()
                if not result
            ]

            raise RuntimeError(
                "Validation integrity check failed: "
                + ", ".join(
                    failed_checks
                )
            )

        return {
            "passed": True,

            "checks": checks,

            "observations":
                int(observations),

            "fold_test_rows":
                int(fold_test_rows),

            "confusion_matrix_total":
                int(confusion_total),

            "baseline_observations":
                int(baseline_observations),
        }

    # =========================================================
    # RELIABILITY
    # =========================================================

    @staticmethod
    def calculate_reliability(
        metrics,
        baseline,
    ):
        """
        Interpret historical validation performance.

        Reliability is deliberately conservative.

        It is NOT a prediction of future investment returns.
        """

        accuracy = metrics[
            "accuracy"
        ]

        f1 = metrics[
            "f1"
        ]

        auc = metrics[
            "roc_auc"
        ]

        baseline_accuracy = baseline[
            "accuracy"
        ]

        improvement = (
            accuracy
            - baseline_accuracy
        )

        # -----------------------------------------------------
        # Conservative reliability classification
        # -----------------------------------------------------

        if (
            accuracy >= 0.60
            and f1 >= 0.58
            and auc is not None
            and auc >= 0.60
            and improvement >= 0.05
        ):

            level = "GOOD"

        elif (
            accuracy >= 0.55
            and f1 >= 0.50
            and auc is not None
            and auc >= 0.55
            and improvement >= 0.02
        ):

            level = "MODERATE"

        else:

            level = "WEAK"

        # -----------------------------------------------------
        # Interpretation
        # -----------------------------------------------------

        if improvement > 0:

            baseline_interpretation = (
                "The classifier performs above the "
                "majority-class baseline on the historical "
                "validation observations."
            )

        elif improvement < 0:

            baseline_interpretation = (
                "The classifier performs below the "
                "majority-class baseline on the historical "
                "validation observations."
            )

        else:

            baseline_interpretation = (
                "The classifier matches the majority-class "
                "baseline on the historical validation "
                "observations."
            )

        return {
            "level":
                level,

            "baseline_improvement":
                float(
                    improvement
                ),

            "baseline_interpretation":
                baseline_interpretation,

            "note":
                (
                    "Reliability reflects historical "
                    "walk-forward validation performance "
                    "only. It does not predict future "
                    "investment returns."
                ),
        }

    # =========================================================
    # COMPLETE VALIDATION
    # =========================================================

    def validate(
        self,
        features,
        n_splits=5,
    ):
        """
        Run the complete StockSense validation pipeline.
        """

        # -----------------------------------------------------
        # Prepare data
        # -----------------------------------------------------

        X, y = self.prepare_data(
            features
        )

        # -----------------------------------------------------
        # Walk-forward validation
        # -----------------------------------------------------

        (
            fold_results,
            actual,
            predictions,
            probabilities,
        ) = self.walk_forward_validate(
            X,
            y,
            n_splits=n_splits,
        )

        # -----------------------------------------------------
        # Aggregate metrics
        # -----------------------------------------------------

        metrics = self.calculate_metrics(
            actual,
            predictions,
            probabilities,
        )

        # -----------------------------------------------------
        # Majority baseline
        # -----------------------------------------------------

        baseline = self.calculate_baseline(
            actual
        )

        # -----------------------------------------------------
        # Class distribution
        # -----------------------------------------------------

        (
            class_distribution,
            class_percentages,
        ) = self.calculate_class_distribution(
            actual
        )

        # -----------------------------------------------------
        # Fold diagnostics
        # -----------------------------------------------------

        fold_diagnostics = (
            self.calculate_fold_diagnostics(
                fold_results
            )
        )

        # -----------------------------------------------------
        # Reliability
        # -----------------------------------------------------

        reliability = (
            self.calculate_reliability(
                metrics,
                baseline,
            )
        )

        # -----------------------------------------------------
        # Integrity checks
        # -----------------------------------------------------

        integrity = (
            self.calculate_integrity(
                actual,
                predictions,
                fold_results,
                baseline,
                metrics,
            )
        )

        # -----------------------------------------------------
        # Final response
        # -----------------------------------------------------

        return {
            "status":
                "validated",

            "validation_type":
                "Expanding walk-forward validation",

            "splits":
                int(
                    len(fold_results)
                ),

            "observations":
                int(
                    len(actual)
                ),

            "features":
                int(
                    X.shape[1]
                ),

            "metrics":
                metrics,

            "baseline":
                baseline,

            "reliability":
                reliability,

            "class_distribution":
                class_distribution,

            "class_percentages":
                class_percentages,

            "fold_diagnostics":
                fold_diagnostics,

            "integrity":
                integrity,

            "folds":
                fold_results,
        }