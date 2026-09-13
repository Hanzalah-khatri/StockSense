"""
StockSense - Milestone 4, Step 3
Classifier Algorithm Comparison

Compares multiple classifiers using:
- Top 5 selected features
- Expanding walk-forward validation
- Identical train/test folds
- Identical evaluation metrics
- Majority-class baseline

No production model is modified by this script.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from services.market_data import fetch_stock_data
from services.prediction_service import PredictionService
from services.validation_service import ValidationService


# =========================================================
# CONFIGURATION
# =========================================================

TICKER = "AAPL"

N_SPLITS = 5
MIN_TRAIN_SIZE = 150

TOP_FEATURES = [
    "Return_Lag_1",
    "Return_Lag_3",
    "Return_Lag_5",
    "Volatility",
    "Volume_Change",
]


# =========================================================
# MODEL FACTORIES
# =========================================================

def create_models():
    """
    Return fresh model instances for every validation fold.
    """

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),

        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=42,
        ),
    }


# =========================================================
# DATA PREPARATION
# =========================================================

def load_data():
    print("=" * 75)
    print("LOADING MARKET DATA")
    print("=" * 75)

    market_data = fetch_stock_data(
        TICKER,
        period="2y"
    )

    print(f"Loaded {len(market_data)} raw rows.")

    prediction_service = PredictionService()

    features = prediction_service.prepare_features(
        market_data
    )

    validation_service = ValidationService()

    X, y = validation_service.prepare_data(
        features
    )

    # Keep only the selected Top 5 features.
    X = X[TOP_FEATURES].copy()

    print(f"Usable observations: {len(X)}")
    print(f"Selected features: {len(TOP_FEATURES)}")

    print("\nFeatures:")
    print(", ".join(TOP_FEATURES))

    return X, y


# =========================================================
# WALK-FORWARD FOLDS
# =========================================================

def create_walk_forward_folds(
    total_rows,
    n_splits,
    min_train_size
):
    """
    Create the exact expanding-window structure used
    throughout Milestone 4.
    """

    test_size = (
        total_rows - min_train_size
    ) // n_splits

    folds = []

    for fold in range(n_splits):

        train_end = (
            min_train_size
            + fold * test_size
        )

        if fold == n_splits - 1:
            test_end = total_rows
        else:
            test_end = train_end + test_size

        train_indices = np.arange(
            0,
            train_end
        )

        test_indices = np.arange(
            train_end,
            test_end
        )

        if len(test_indices) == 0:
            continue

        folds.append(
            (
                fold + 1,
                train_indices,
                test_indices,
            )
        )

    return folds


# =========================================================
# MODEL EVALUATION
# =========================================================

def evaluate_model(
    model_name,
    model_factory,
    X,
    y,
    folds
):
    """
    Evaluate one classifier using identical
    walk-forward folds.
    """

    actual_all = []
    predicted_all = []
    probability_all = []

    fold_results = []

    print("\n" + "=" * 75)
    print(f"TESTING: {model_name}")
    print("=" * 75)

    for (
        fold_number,
        train_indices,
        test_indices,
    ) in folds:

        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]

        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        # Fresh model for every fold.
        model = model_factory()

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        # Probability for ROC-AUC.
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                X_test
            )[:, 1]

        elif hasattr(model, "decision_function"):

            probabilities = model.decision_function(
                X_test
            )

        else:

            probabilities = predictions.astype(float)

        actual_all.extend(
            y_test.tolist()
        )

        predicted_all.extend(
            predictions.tolist()
        )

        probability_all.extend(
            probabilities.tolist()
        )

        fold_accuracy = accuracy_score(
            y_test,
            predictions
        )

        fold_precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        fold_recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        fold_f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        try:
            fold_auc = roc_auc_score(
                y_test,
                probabilities
            )
        except ValueError:
            fold_auc = np.nan

        fold_results.append(
            {
                "fold": fold_number,
                "train_size": len(train_indices),
                "test_size": len(test_indices),
                "accuracy": fold_accuracy,
                "precision": fold_precision,
                "recall": fold_recall,
                "f1": fold_f1,
                "roc_auc": fold_auc,
            }
        )

        print(
            f"Fold {fold_number}: "
            f"Accuracy={fold_accuracy:.4f}, "
            f"F1={fold_f1:.4f}, "
            f"ROC-AUC={fold_auc:.4f}"
        )

    # =====================================================
    # AGGREGATED OUT-OF-SAMPLE METRICS
    # =====================================================

    accuracy = accuracy_score(
        actual_all,
        predicted_all
    )

    precision = precision_score(
        actual_all,
        predicted_all,
        zero_division=0
    )

    recall = recall_score(
        actual_all,
        predicted_all,
        zero_division=0
    )

    f1 = f1_score(
        actual_all,
        predicted_all,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        actual_all,
        probability_all
    )

    fold_df = pd.DataFrame(
        fold_results
    )

    print("\nAggregated out-of-sample results:")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1:        {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print(
        f"Fold accuracy mean: "
        f"{fold_df['accuracy'].mean() * 100:.2f}%"
    )

    print(
        f"Fold accuracy std: "
        f"{fold_df['accuracy'].std(ddof=0) * 100:.2f} pp"
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "fold_accuracy_mean": fold_df["accuracy"].mean(),
        "fold_accuracy_std": fold_df["accuracy"].std(ddof=0),
        "fold_accuracy_min": fold_df["accuracy"].min(),
        "fold_accuracy_max": fold_df["accuracy"].max(),
        "folds": fold_results,
    }


# =========================================================
# BASELINE
# =========================================================

def calculate_baseline(y, folds):
    """
    Calculate the majority baseline using the exact
    out-of-sample observations from the validation folds.
    """

    test_indices = np.concatenate(
        [
            test_indices
            for _, _, test_indices in folds
        ]
    )

    actual = y.iloc[test_indices]

    majority_class = (
        actual.value_counts()
        .idxmax()
    )

    baseline_predictions = np.full(
        len(actual),
        majority_class
    )

    accuracy = accuracy_score(
        actual,
        baseline_predictions
    )

    return {
        "accuracy": accuracy,
        "majority_class": int(majority_class),
        "observations": len(actual),
    }


# =========================================================
# MAIN
# =========================================================

def main():

    X, y = load_data()

    folds = create_walk_forward_folds(
        total_rows=len(X),
        n_splits=N_SPLITS,
        min_train_size=MIN_TRAIN_SIZE,
    )

    print("\n" + "=" * 75)
    print("WALK-FORWARD VALIDATION")
    print("=" * 75)

    for (
        fold_number,
        train_indices,
        test_indices,
    ) in folds:

        print(
            f"Fold {fold_number}: "
            f"Train={len(train_indices)} "
            f"Test={len(test_indices)}"
        )

    baseline = calculate_baseline(
        y,
        folds
    )

    print("\n" + "=" * 75)
    print("MAJORITY BASELINE")
    print("=" * 75)

    print(
        f"Accuracy: "
        f"{baseline['accuracy'] * 100:.2f}%"
    )

    print(
        f"Majority class: "
        f"{'UP' if baseline['majority_class'] == 1 else 'DOWN'}"
    )

    print(
        f"Observations: "
        f"{baseline['observations']}"
    )

    # =====================================================
    # FACTORIES
    # =====================================================

    model_factories = {
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),

        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),

        "Gradient Boosting": lambda: GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),

        "Extra Trees": lambda: ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),

        "Hist Gradient Boosting": lambda: HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=42,
        ),
    }

    # =====================================================
    # RUN COMPARISON
    # =====================================================

    results = []

    for model_name, factory in model_factories.items():

        result = evaluate_model(
            model_name,
            factory,
            X,
            y,
            folds,
        )

        results.append(
            {
                "Model": result["model"],
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1": result["f1"],
                "ROC-AUC": result["roc_auc"],
                "Fold Mean Accuracy": result[
                    "fold_accuracy_mean"
                ],
                "Fold Std": result[
                    "fold_accuracy_std"
                ],
                "Fold Min Accuracy": result[
                    "fold_accuracy_min"
                ],
                "Fold Max Accuracy": result[
                    "fold_accuracy_max"
                ],
            }
        )

    results_df = pd.DataFrame(
        results
    )

    # =====================================================
    # BASELINE COMPARISON
    # =====================================================

    results_df["vs Baseline"] = (
        results_df["Accuracy"]
        - baseline["accuracy"]
    )

    results_df = results_df.sort_values(
        by=["Accuracy", "ROC-AUC"],
        ascending=False
    ).reset_index(
        drop=True
    )

    # =====================================================
    # FINAL RESULTS
    # =====================================================

    print("\n\n" + "=" * 75)
    print("FINAL CLASSIFIER COMPARISON")
    print("=" * 75)

    print(
        f"{'Model':<24}"
        f"{'Accuracy':>12}"
        f"{'F1':>12}"
        f"{'ROC-AUC':>12}"
        f"{'vs Base':>12}"
    )

    print("-" * 75)

    for _, row in results_df.iterrows():

        print(
            f"{row['Model']:<24}"
            f"{row['Accuracy'] * 100:>11.2f}%"
            f"{row['F1'] * 100:>11.2f}%"
            f"{row['ROC-AUC']:>12.4f}"
            f"{row['vs Baseline'] * 100:>10.2f} pp"
        )

    # =====================================================
    # BEST MODEL
    # =====================================================

    best = results_df.iloc[0]

    print("\n" + "=" * 75)
    print("BEST CLASSIFIER")
    print("=" * 75)

    print(f"Model: {best['Model']}")
    print(
        f"Accuracy: "
        f"{best['Accuracy'] * 100:.2f}%"
    )
    print(
        f"F1: "
        f"{best['F1'] * 100:.2f}%"
    )
    print(
        f"ROC-AUC: "
        f"{best['ROC-AUC']:.4f}"
    )
    print(
        f"vs Baseline: "
        f"{best['vs Baseline'] * 100:+.2f} pp"
    )

    print(
        f"Fold mean accuracy: "
        f"{best['Fold Mean Accuracy'] * 100:.2f}%"
    )

    print(
        f"Fold accuracy range: "
        f"{best['Fold Min Accuracy'] * 100:.2f}% - "
        f"{best['Fold Max Accuracy'] * 100:.2f}%"
    )

    # =====================================================
    # SAVE
    # =====================================================

    reports_dir = Path("reports")
    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        reports_dir
        / "classifier_comparison.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 75)
    print("RESULTS SAVED")
    print("=" * 75)

    print(output_path.resolve())

    print("\n" + "=" * 75)
    print("Classifier comparison complete.")
    print("=" * 75)


if __name__ == "__main__":
    main()