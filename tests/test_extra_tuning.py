"""
StockSense - Milestone 4, Step 4
Extra Trees Hyperparameter Tuning

Uses:
- Top 5 selected features
- Expanding walk-forward validation
- Same folds for every configuration
- Majority-class baseline
- No production model modification
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
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
# HYPERPARAMETER SEARCH SPACE
# =========================================================

N_ESTIMATORS_VALUES = [
    100,
    200,
    300,
]

MAX_DEPTH_VALUES = [
    5,
    8,
    10,
    15,
    None,
]

MIN_SAMPLES_SPLIT_VALUES = [
    2,
    5,
    10,
]


# =========================================================
# DATA
# =========================================================

def load_data():

    print("=" * 75)
    print("LOADING MARKET DATA")
    print("=" * 75)

    market_data = fetch_stock_data(
        TICKER,
        period="2y"
    )

    print(
        f"Loaded {len(market_data)} raw rows."
    )

    prediction_service = PredictionService()

    features = prediction_service.prepare_features(
        market_data
    )

    validation_service = ValidationService()

    X, y = validation_service.prepare_data(
        features
    )

    X = X[TOP_FEATURES].copy()

    print(
        f"Usable observations: {len(X)}"
    )

    print(
        f"Selected features: {len(TOP_FEATURES)}"
    )

    print("\nFeatures:")
    print(", ".join(TOP_FEATURES))

    return X, y


# =========================================================
# WALK-FORWARD FOLDS
# =========================================================

def create_folds(
    total_rows,
    n_splits,
    min_train_size
):

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
            test_end = (
                train_end + test_size
            )

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
# BASELINE
# =========================================================

def calculate_baseline(y, folds):

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

    return accuracy


# =========================================================
# EVALUATE CONFIGURATION
# =========================================================

def evaluate_configuration(
    n_estimators,
    max_depth,
    min_samples_split,
    X,
    y,
    folds
):

    actual_all = []
    predicted_all = []
    probability_all = []

    fold_accuracies = []

    for (
        fold_number,
        train_indices,
        test_indices,
    ) in folds:

        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]

        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

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

        fold_accuracies.append(
            fold_accuracy
        )

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

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "fold_mean_accuracy": np.mean(
            fold_accuracies
        ),
        "fold_std_accuracy": np.std(
            fold_accuracies
        ),
        "fold_min_accuracy": np.min(
            fold_accuracies
        ),
        "fold_max_accuracy": np.max(
            fold_accuracies
        ),
    }


# =========================================================
# MAIN
# =========================================================

def main():

    X, y = load_data()

    folds = create_folds(
        total_rows=len(X),
        n_splits=N_SPLITS,
        min_train_size=MIN_TRAIN_SIZE,
    )

    print("\n" + "=" * 75)
    print("WALK-FORWARD FOLDS")
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
        f"{baseline * 100:.2f}%"
    )

    # =====================================================
    # SEARCH
    # =====================================================

    total_combinations = (
        len(N_ESTIMATORS_VALUES)
        * len(MAX_DEPTH_VALUES)
        * len(MIN_SAMPLES_SPLIT_VALUES)
    )

    print("\n" + "=" * 75)
    print("EXTRA TREES HYPERPARAMETER SEARCH")
    print("=" * 75)

    print(
        f"Configurations to test: "
        f"{total_combinations}"
    )

    results = []

    counter = 0

    for n_estimators in N_ESTIMATORS_VALUES:

        for max_depth in MAX_DEPTH_VALUES:

            for min_samples_split in (
                MIN_SAMPLES_SPLIT_VALUES
            ):

                counter += 1

                print(
                    f"\n[{counter}/{total_combinations}] "
                    f"Testing: "
                    f"trees={n_estimators}, "
                    f"depth={max_depth}, "
                    f"min_split={min_samples_split}"
                )

                result = evaluate_configuration(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    X=X,
                    y=y,
                    folds=folds,
                )

                result["vs_baseline"] = (
                    result["accuracy"]
                    - baseline
                )

                results.append(
                    result
                )

                print(
                    f"Accuracy: "
                    f"{result['accuracy'] * 100:.2f}% | "
                    f"F1: "
                    f"{result['f1'] * 100:.2f}% | "
                    f"ROC-AUC: "
                    f"{result['roc_auc']:.4f} | "
                    f"vs Base: "
                    f"{result['vs_baseline'] * 100:+.2f} pp"
                )

    # =====================================================
    # RESULTS
    # =====================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by=[
            "accuracy",
            "roc_auc",
            "f1",
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    print("\n\n" + "=" * 75)
    print("TOP EXTRA TREES CONFIGURATIONS")
    print("=" * 75)

    print(
        f"{'Trees':>7}"
        f"{'Depth':>9}"
        f"{'MinSplit':>10}"
        f"{'Accuracy':>12}"
        f"{'F1':>10}"
        f"{'ROC-AUC':>12}"
        f"{'vs Base':>12}"
    )

    print("-" * 75)

    for _, row in results_df.head(10).iterrows():

        depth = (
            "None"
            if pd.isna(row["max_depth"])
            else str(int(row["max_depth"]))
        )

        print(
            f"{int(row['n_estimators']):>7}"
            f"{depth:>9}"
            f"{int(row['min_samples_split']):>10}"
            f"{row['accuracy'] * 100:>11.2f}%"
            f"{row['f1'] * 100:>9.2f}%"
            f"{row['roc_auc']:>12.4f}"
            f"{row['vs_baseline'] * 100:>10.2f} pp"
        )

    # =====================================================
    # BEST CONFIGURATION
    # =====================================================

    best = results_df.iloc[0]

    print("\n" + "=" * 75)
    print("BEST EXTRA TREES CONFIGURATION")
    print("=" * 75)

    print(
        f"Trees: "
        f"{int(best['n_estimators'])}"
    )

    if pd.isna(best["max_depth"]):
        print("Max depth: None")
    else:
        print(
            f"Max depth: "
            f"{int(best['max_depth'])}"
        )

    print(
        f"Min samples split: "
        f"{int(best['min_samples_split'])}"
    )

    print(
        f"Accuracy: "
        f"{best['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{best['precision'] * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{best['recall'] * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{best['f1'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC: "
        f"{best['roc_auc']:.4f}"
    )

    print(
        f"vs Baseline: "
        f"{best['vs_baseline'] * 100:+.2f} pp"
    )

    print(
        f"Fold mean accuracy: "
        f"{best['fold_mean_accuracy'] * 100:.2f}%"
    )

    print(
        f"Fold accuracy range: "
        f"{best['fold_min_accuracy'] * 100:.2f}% - "
        f"{best['fold_max_accuracy'] * 100:.2f}%"
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
        / "extra_trees_tuning.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 75)
    print("RESULTS SAVED")
    print("=" * 75)

    print(
        output_path.resolve()
    )

    print("\n" + "=" * 75)
    print("Extra Trees tuning complete.")
    print("=" * 75)


if __name__ == "__main__":
    main()