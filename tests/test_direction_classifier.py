"""
StockSense — Direction Classification Experiment
------------------------------------------------

Predicts whether the NEXT trading day's return will be:

    1 = UP
    0 = DOWN

Models:
    1. Logistic Regression
    2. Random Forest Classifier
    3. Gradient Boosting Classifier

Validation:
    Walk-forward / TimeSeriesSplit

Features:
    - Current 30-feature set
    - Selected 15-feature set

Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
"""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from services.market_data import fetch_stock_data
from services.prediction_service import PredictionService


# =========================================================
# SETTINGS
# =========================================================

TICKER = "AAPL"
PERIOD = "2y"
N_SPLITS = 5


# =========================================================
# CLASSIFICATION MODELS
# =========================================================

def create_models():

    return {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(y_actual, y_pred, y_probability):

    accuracy = accuracy_score(
        y_actual,
        y_pred
    )

    precision = precision_score(
        y_actual,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_actual,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_actual,
        y_pred,
        zero_division=0
    )

    # ROC-AUC requires both classes to exist
    if len(np.unique(y_actual)) == 2:
        roc_auc = roc_auc_score(
            y_actual,
            y_probability
        )
    else:
        roc_auc = np.nan

    return {
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1 Score": f1 * 100,
        "ROC-AUC": roc_auc
    }


# =========================================================
# EVALUATE MODEL WITH WALK-FORWARD VALIDATION
# =========================================================

def evaluate_model(model, X, y, splitter):

    fold_results = []

    all_actual = []
    all_predictions = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Fresh model for every fold
        current_model = clone(model)

        current_model.fit(
            X_train,
            y_train
        )

        # Predicted class
        predictions = current_model.predict(
            X_test
        )

        # Probability of UP
        probabilities = current_model.predict_proba(
            X_test
        )[:, 1]

        metrics = calculate_metrics(
            y_test,
            predictions,
            probabilities
        )

        fold_results.append({
            "Fold": fold,
            **metrics
        })

        all_actual.extend(
            y_test.tolist()
        )

        all_predictions.extend(
            predictions.tolist()
        )

    return (
        pd.DataFrame(fold_results),
        np.asarray(all_actual),
        np.asarray(all_predictions)
    )


# =========================================================
# PRINT MODEL RESULTS
# =========================================================

def print_results(
    model_name,
    feature_set_name,
    results
):

    print("\n")
    print("=" * 70)
    print(
        f"{model_name} — {feature_set_name}"
    )
    print("=" * 70)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print("\nAverage:")

    print(
        f"Accuracy:   "
        f"{results['Accuracy'].mean():.2f}%"
    )

    print(
        f"Precision:  "
        f"{results['Precision'].mean():.2f}%"
    )

    print(
        f"Recall:     "
        f"{results['Recall'].mean():.2f}%"
    )

    print(
        f"F1 Score:   "
        f"{results['F1 Score'].mean():.2f}%"
    )

    print(
        f"ROC-AUC:    "
        f"{results['ROC-AUC'].mean():.4f}"
    )

    print("\nStd Dev:")

    print(
        f"Accuracy:   "
        f"{results['Accuracy'].std():.2f}%"
    )

    print(
        f"Precision:  "
        f"{results['Precision'].std():.2f}%"
    )

    print(
        f"Recall:     "
        f"{results['Recall'].std():.2f}%"
    )

    print(
        f"F1 Score:   "
        f"{results['F1 Score'].std():.2f}%"
    )

    print(
        f"ROC-AUC:    "
        f"{results['ROC-AUC'].std():.4f}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("StockSense — Direction Classification")
    print("=" * 70)

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    data = fetch_stock_data(
        TICKER,
        period=PERIOD
    )

    print(
        f"\nLoaded {len(data)} rows"
    )

    # -----------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------

    prediction_service = PredictionService()

    features = prediction_service.prepare_features(
        data
    )

    # -----------------------------------------------------
    # NEXT-DAY RETURN
    # -----------------------------------------------------

    next_day_return = (
        features["Close"].shift(-1)
        .div(features["Close"])
        - 1
    )

    valid_rows = next_day_return.notna()

    next_day_return = (
        next_day_return
        .loc[valid_rows]
        .copy()
    )

    # -----------------------------------------------------
    # CLASSIFICATION TARGET
    # -----------------------------------------------------
    #
    # Positive return  -> 1 (UP)
    # Zero/negative    -> 0 (DOWN)
    #

    y = (
        next_day_return > 0
    ).astype(int)

    print("\nTarget distribution:")

    print(
        y.value_counts()
        .sort_index()
        .rename({
            0: "DOWN",
            1: "UP"
        })
        .to_string()
    )

    print(
        f"\nUP percentage: "
        f"{(y == 1).mean() * 100:.2f}%"
    )

    print(
        f"DOWN percentage: "
        f"{(y == 0).mean() * 100:.2f}%"
    )

    # =====================================================
    # FEATURE SETS
    # =====================================================

    current_feature_names = (
        prediction_service
        .get_feature_columns()
    )

    selected_feature_names = (
        prediction_service
        .get_selected_feature_columns()
    )

    X_current = (
        features.loc[
            valid_rows,
            current_feature_names
        ]
        .copy()
    )

    X_selected = (
        features.loc[
            valid_rows,
            selected_feature_names
        ]
        .copy()
    )

    # =====================================================
    # MODELS
    # =====================================================

    models = create_models()

    # =====================================================
    # TIME-SERIES VALIDATION
    # =====================================================

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS
    )

    print("\n")
    print("=" * 70)
    print("VALIDATION CONFIGURATION")
    print("=" * 70)

    print(
        f"Ticker:              {TICKER}"
    )

    print(
        f"Rows:                {len(y)}"
    )

    print(
        f"Current features:    "
        f"{len(current_feature_names)}"
    )

    print(
        f"Selected features:   "
        f"{len(selected_feature_names)}"
    )

    print(
        f"Splits:              {N_SPLITS}"
    )

    print(
        "Validation:           "
        "Expanding walk-forward"
    )

    # =====================================================
    # STORE SUMMARY
    # =====================================================

    summary = []

    # =====================================================
    # CURRENT FEATURES
    # =====================================================

    print("\n\n")
    print("#" * 70)
    print("# CURRENT FEATURES")
    print("#" * 70)

    for model_name, model in models.items():

        results, actual, predictions = (
            evaluate_model(
                model,
                X_current,
                y,
                splitter
            )
        )

        print_results(
            model_name,
            "Current (30)",
            results
        )

        summary.append({
            "Feature Set": "Current (30)",
            "Model": model_name,
            "Accuracy": results[
                "Accuracy"
            ].mean(),
            "Precision": results[
                "Precision"
            ].mean(),
            "Recall": results[
                "Recall"
            ].mean(),
            "F1 Score": results[
                "F1 Score"
            ].mean(),
            "ROC-AUC": results[
                "ROC-AUC"
            ].mean()
        })

    # =====================================================
    # SELECTED FEATURES
    # =====================================================

    print("\n\n")
    print("#" * 70)
    print("# SELECTED FEATURES")
    print("#" * 70)

    for model_name, model in models.items():

        results, actual, predictions = (
            evaluate_model(
                model,
                X_selected,
                y,
                splitter
            )
        )

        print_results(
            model_name,
            "Selected (15)",
            results
        )

        summary.append({
            "Feature Set": "Selected (15)",
            "Model": model_name,
            "Accuracy": results[
                "Accuracy"
            ].mean(),
            "Precision": results[
                "Precision"
            ].mean(),
            "Recall": results[
                "Recall"
            ].mean(),
            "F1 Score": results[
                "F1 Score"
            ].mean(),
            "ROC-AUC": results[
                "ROC-AUC"
            ].mean()
        })

    # =====================================================
    # BASELINE
    # =====================================================

    print("\n\n")
    print("=" * 70)
    print("CLASSIFICATION BASELINES")
    print("=" * 70)

    # -----------------------------------------------------
    # Majority-class baseline
    # -----------------------------------------------------

    majority_class = (
        y.value_counts()
        .idxmax()
    )

    majority_predictions = np.full(
        len(y),
        majority_class
    )

    majority_accuracy = (
        accuracy_score(
            y,
            majority_predictions
        ) * 100
    )

    print(
        f"\nMajority Class Baseline: "
        f"{'UP' if majority_class == 1 else 'DOWN'}"
    )

    print(
        f"Accuracy: "
        f"{majority_accuracy:.2f}%"
    )

    summary.append({
        "Feature Set": "Baseline",
        "Model": "Majority Class",
        "Accuracy": majority_accuracy,
        "Precision": np.nan,
        "Recall": np.nan,
        "F1 Score": np.nan,
        "ROC-AUC": np.nan
    })

    # =====================================================
    # FINAL COMPARISON
    # =====================================================

    summary_df = pd.DataFrame(
        summary
    )

    print("\n\n")
    print("=" * 70)
    print("FINAL CLASSIFICATION COMPARISON")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # =====================================================
    # BEST MODELS
    # =====================================================

    ml_summary = summary_df[
        summary_df["Feature Set"] != "Baseline"
    ]

    best_accuracy = ml_summary.loc[
        ml_summary["Accuracy"].idxmax()
    ]

    best_f1 = ml_summary.loc[
        ml_summary["F1 Score"].idxmax()
    ]

    best_auc = ml_summary.loc[
        ml_summary["ROC-AUC"].idxmax()
    ]

    print("\n")
    print("=" * 70)
    print("BEST RESULTS")
    print("=" * 70)

    print(
        f"\nBest Accuracy:"
        f"\n  {best_accuracy['Model']}"
        f" — {best_accuracy['Feature Set']}"
        f" — {best_accuracy['Accuracy']:.2f}%"
    )

    print(
        f"\nBest F1 Score:"
        f"\n  {best_f1['Model']}"
        f" — {best_f1['Feature Set']}"
        f" — {best_f1['F1 Score']:.2f}%"
    )

    print(
        f"\nBest ROC-AUC:"
        f"\n  {best_auc['Model']}"
        f" — {best_auc['Feature Set']}"
        f" — {best_auc['ROC-AUC']:.4f}"
    )

    # =====================================================
    # INTERPRETATION
    # =====================================================

    print("\n")
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print("""
Important:

1. Accuracy near 50% means the model has little
   directional advantage.

2. The model should beat the Majority Class baseline.

3. ROC-AUC above 0.50 suggests some ability to
   distinguish UP from DOWN days.

4. F1 Score is useful when UP and DOWN classes
   are not perfectly balanced.

5. Consistent performance across all five folds
   is more important than one excellent fold.

6. We should NOT integrate a classifier into the
   production application unless it shows a
   reasonably consistent advantage.

7. This experiment uses chronological validation,
   so future data is never used to train earlier folds.
""")

    print("=" * 70)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()