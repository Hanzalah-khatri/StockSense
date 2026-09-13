"""
StockSense — Milestone 3.14
Feature Subset Validation

Goal:
Compare different feature subsets using expanding walk-forward validation.

Feature sets:
1. Full 30 features
2. Top 15 Random Forest features
3. Top 5 candidate features
4. Practical technical-analysis subset

Important:
- Uses PredictionService for the official feature pipeline.
- Converts regression-style next-day return target into UP/DOWN classification.
- Uses chronological validation.
- Keeps the final 61-row holdout untouched.
- Feature selection is based ONLY on the training portion.
"""

import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from sklearn.model_selection import TimeSeriesSplit

from services.market_data import fetch_stock_data
from services.prediction_service import PredictionService


# =========================================================
# CONFIGURATION
# =========================================================

TICKER = "AAPL"

TRAIN_SIZE = 240
N_SPLITS = 5

# Same tuned Random Forest configuration from Milestone 3.10+
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1,
}


warnings.filterwarnings("ignore")


# =========================================================
# DISPLAY HELPERS
# =========================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_subheader(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


# =========================================================
# RANDOM FOREST FACTORY
# =========================================================

def create_model():
    """
    Create the tuned Random Forest classifier.
    """
    return RandomForestClassifier(**RF_PARAMS)


# =========================================================
# SAFE ROC-AUC
# =========================================================

def calculate_roc_auc(y_true, probabilities):
    """
    Calculate ROC-AUC safely.

    ROC-AUC cannot be calculated if a validation fold
    contains only one class.
    """
    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, probabilities)


# =========================================================
# FEATURE RANKING
# =========================================================

def get_training_feature_importance(X_train, y_train, feature_columns):
    """
    Train a Random Forest ONLY on the training portion.

    This prevents the untouched final holdout from being used
    to select features.
    """

    print_subheader("Training-Only Feature Importance")

    model = create_model()

    model.fit(
        X_train[feature_columns],
        y_train,
    )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance": model.feature_importances_,
        }
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False,
    ).reset_index(drop=True)

    print()
    print("Feature ranking generated using training data only:")
    print()

    for i, row in importance_df.iterrows():
        print(
            f"{i + 1:2d}. "
            f"{row['Feature']:<20} "
            f"{row['Importance'] * 100:6.2f}%"
        )

    return importance_df


# =========================================================
# WALK-FORWARD VALIDATION
# =========================================================

def evaluate_feature_set(
    X_train,
    y_train,
    feature_list,
    feature_set_name,
):
    """
    Evaluate one feature subset using expanding
    walk-forward validation.
    """

    print_subheader(
        f"Evaluating: {feature_set_name}"
    )

    print(
        f"Features used: {len(feature_list)}"
    )

    print(
        "Features:",
        ", ".join(feature_list)
    )

    tscv = TimeSeriesSplit(
        n_splits=N_SPLITS
    )

    fold_results = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        tscv.split(X_train),
        start=1,
    ):

        X_fold_train = X_train.iloc[
            train_indices
        ][feature_list]

        X_fold_validation = X_train.iloc[
            validation_indices
        ][feature_list]

        y_fold_train = y_train.iloc[
            train_indices
        ]

        y_fold_validation = y_train.iloc[
            validation_indices
        ]

        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        model = create_model()

        model.fit(
            X_fold_train,
            y_fold_train,
        )

        # -------------------------------------------------
        # Predict
        # -------------------------------------------------

        predictions = model.predict(
            X_fold_validation
        )

        probabilities = model.predict_proba(
            X_fold_validation
        )[:, 1]

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        accuracy = accuracy_score(
            y_fold_validation,
            predictions,
        )

        f1 = f1_score(
            y_fold_validation,
            predictions,
            zero_division=0,
        )

        roc_auc = calculate_roc_auc(
            y_fold_validation,
            probabilities,
        )

        fold_results.append(
            {
                "Fold": fold_number,
                "Accuracy": accuracy,
                "F1": f1,
                "ROC_AUC": roc_auc,
                "Train_Size": len(train_indices),
                "Validation_Size": len(validation_indices),
            }
        )

        auc_display = (
            f"{roc_auc:.4f}"
            if not np.isnan(roc_auc)
            else "N/A"
        )

        print(
            f"Fold {fold_number}: "
            f"Accuracy={accuracy:.2%} | "
            f"F1={f1:.4f} | "
            f"ROC-AUC={auc_display} | "
            f"Train={len(train_indices)} | "
            f"Validation={len(validation_indices)}"
        )

    # -----------------------------------------------------
    # Convert results to DataFrame
    # -----------------------------------------------------

    results_df = pd.DataFrame(
        fold_results
    )

    # -----------------------------------------------------
    # Average metrics
    # -----------------------------------------------------

    avg_accuracy = results_df[
        "Accuracy"
    ].mean()

    std_accuracy = results_df[
        "Accuracy"
    ].std()

    avg_f1 = results_df[
        "F1"
    ].mean()

    std_f1 = results_df[
        "F1"
    ].std()

    avg_auc = results_df[
        "ROC_AUC"
    ].mean()

    std_auc = results_df[
        "ROC_AUC"
    ].std()

    print()
    print(
        f"Average Accuracy : {avg_accuracy:.2%}"
    )

    print(
        f"Accuracy Std     : {std_accuracy:.2%}"
    )

    print(
        f"Average F1       : {avg_f1:.4f}"
    )

    print(
        f"F1 Std           : {std_f1:.4f}"
    )

    print(
        f"Average ROC-AUC  : {avg_auc:.4f}"
    )

    print(
        f"ROC-AUC Std      : {std_auc:.4f}"
    )

    return {
        "Feature_Set": feature_set_name,
        "Feature_Count": len(feature_list),
        "Features": feature_list,
        "Accuracy": avg_accuracy,
        "Accuracy_Std": std_accuracy,
        "F1": avg_f1,
        "F1_Std": std_f1,
        "ROC_AUC": avg_auc,
        "ROC_AUC_Std": std_auc,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print_header(
        "STOCKSENSE — MILESTONE 3.14"
    )

    print(
        "Feature Subset Validation"
    )

    print()
    print(
        f"Ticker: {TICKER}"
    )

    print(
        f"Training rows: {TRAIN_SIZE}"
    )

    print(
        f"Walk-forward splits: {N_SPLITS}"
    )

    print(
        "Validation type: Expanding TimeSeriesSplit"
    )

    print(
        "Model: Tuned Random Forest"
    )


    # =====================================================
    # LOAD DATA
    # =====================================================

    print_header(
        "1. LOADING MARKET DATA"
    )

    market_data = fetch_stock_data(
        TICKER,
        period="2y",
    )

    print(
        f"Loaded {len(market_data)} market rows"
    )


    # =====================================================
    # PREPARE FEATURES
    # =====================================================

    print_header(
        "2. PREPARING FEATURES"
    )

    prediction_service = PredictionService()

    features = prediction_service.prepare_features(
        market_data
    )

    X, y = prediction_service.prepare_ml_data(
        features
    )

    # -----------------------------------------------------
    # Convert next-day return into direction
    #
    # > 0  = UP
    # <= 0 = DOWN
    # -----------------------------------------------------

    y = (y > 0).astype(int)

    # Ensure feature names are available
    feature_columns = (
        prediction_service.get_feature_columns()
    )

    # Make sure X contains the official feature columns
    X = X[feature_columns]

    print(
        f"Usable rows: {len(X)}"
    )

    print(
        f"Total features: {len(feature_columns)}"
    )

    print()
    print("Target distribution:")

    print(
        pd.Series(y)
        .map({
            0: "DOWN",
            1: "UP",
        })
        .value_counts()
    )

    print()

    print(
        f"UP percentage: {y.mean() * 100:.2f}%"
    )

    print(
        f"DOWN percentage: {(1 - y.mean()) * 100:.2f}%"
    )


    # =====================================================
    # CHRONOLOGICAL TRAIN / FINAL HOLDOUT SPLIT
    # =====================================================

    print_header(
        "3. CHRONOLOGICAL DATA SPLIT"
    )

    if len(X) <= TRAIN_SIZE:
        raise ValueError(
            f"Not enough data. "
            f"Need more than {TRAIN_SIZE} rows, "
            f"but only {len(X)} are available."
        )

    X_train = X.iloc[
        :TRAIN_SIZE
    ].copy()

    y_train = y.iloc[
        :TRAIN_SIZE
    ].copy()

    X_holdout = X.iloc[
        TRAIN_SIZE:
    ].copy()

    y_holdout = y.iloc[
        TRAIN_SIZE:
    ].copy()

    print(
        f"Training rows : {len(X_train)}"
    )

    print(
        f"Holdout rows  : {len(X_holdout)}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The final holdout is NOT used during "
        "feature subset validation."
    )


    # =====================================================
    # TRAINING-ONLY FEATURE IMPORTANCE
    # =====================================================

    importance_df = get_training_feature_importance(
        X_train,
        y_train,
        feature_columns,
    )


    # =====================================================
    # FEATURE SET A — FULL 30
    # =====================================================

    full_features = list(
        feature_columns
    )


    # =====================================================
    # FEATURE SET B — TOP 15 RF FEATURES
    # =====================================================

    top_15_features = (
        importance_df
        .head(15)["Feature"]
        .tolist()
    )


    # =====================================================
    # FEATURE SET C — TOP 5 RF FEATURES
    #
    # NOTE:
    # These are selected from training-only RF importance,
    # not from the final holdout.
    # =====================================================

    top_5_features = (
        importance_df
        .head(5)["Feature"]
        .tolist()
    )


    # =====================================================
    # FEATURE SET D — PRACTICAL TECHNICAL SUBSET
    # =====================================================

    practical_candidates = [
        "Relative_Volume",
        "Return_Lag_10",
        "MACD",
        "Daily_Return",
        "EMA_12",
        "RSI",
        "Volatility",
        "Momentum_20",
        "SMA_20",
        "SMA_50",
        "BB_Position",
    ]

    # -----------------------------------------------------
    # Only keep features that actually exist.
    # -----------------------------------------------------

    practical_features = [
        feature
        for feature in practical_candidates
        if feature in feature_columns
    ]

    missing_practical = [
        feature
        for feature in practical_candidates
        if feature not in feature_columns
    ]

    if missing_practical:
        print()
        print(
            "WARNING — Practical subset features "
            "not found:"
        )

        for feature in missing_practical:
            print(
                f"  - {feature}"
            )


    # =====================================================
    # DISPLAY FEATURE SETS
    # =====================================================

    print_header(
        "4. FEATURE SETS"
    )

    print(
        f"A. Full 30 features      : "
        f"{len(full_features)}"
    )

    print(
        f"B. Top 15 RF features    : "
        f"{len(top_15_features)}"
    )

    print(
        f"C. Top 5 RF features     : "
        f"{len(top_5_features)}"
    )

    print(
        f"D. Practical subset      : "
        f"{len(practical_features)}"
    )

    print()

    print(
        "Top 15 RF features:"
    )

    for i, feature in enumerate(
        top_15_features,
        start=1,
    ):
        print(
            f"{i:2d}. {feature}"
        )

    print()

    print(
        "Top 5 RF features:"
    )

    for i, feature in enumerate(
        top_5_features,
        start=1,
    ):
        print(
            f"{i:2d}. {feature}"
        )

    print()

    print(
        "Practical technical subset:"
    )

    for i, feature in enumerate(
        practical_features,
        start=1,
    ):
        print(
            f"{i:2d}. {feature}"
        )


    # =====================================================
    # VALIDATION
    # =====================================================

    print_header(
        "5. WALK-FORWARD VALIDATION"
    )

    all_results = []

    # -----------------------------------------------------
    # A. Full 30
    # -----------------------------------------------------

    result_full = evaluate_feature_set(
        X_train,
        y_train,
        full_features,
        "Full 30 Features",
    )

    all_results.append(
        result_full
    )


    # -----------------------------------------------------
    # B. Top 15
    # -----------------------------------------------------

    result_top15 = evaluate_feature_set(
        X_train,
        y_train,
        top_15_features,
        "Top 15 RF Features",
    )

    all_results.append(
        result_top15
    )


    # -----------------------------------------------------
    # C. Top 5
    # -----------------------------------------------------

    result_top5 = evaluate_feature_set(
        X_train,
        y_train,
        top_5_features,
        "Top 5 RF Features",
    )

    all_results.append(
        result_top5
    )


    # -----------------------------------------------------
    # D. Practical subset
    # -----------------------------------------------------

    result_practical = evaluate_feature_set(
        X_train,
        y_train,
        practical_features,
        "Practical Technical Subset",
    )

    all_results.append(
        result_practical
    )


    # =====================================================
    # RESULTS TABLE
    # =====================================================

    print_header(
        "6. FINAL WALK-FORWARD RESULTS"
    )

    results_df = pd.DataFrame(
        all_results
    )

    display_df = results_df[
        [
            "Feature_Set",
            "Feature_Count",
            "Accuracy",
            "Accuracy_Std",
            "F1",
            "F1_Std",
            "ROC_AUC",
            "ROC_AUC_Std",
        ]
    ].copy()

    # Convert metrics to percentage for display
    display_df[
        "Accuracy"
    ] *= 100

    display_df[
        "Accuracy_Std"
    ] *= 100

    display_df[
        "F1"
    ] *= 100

    display_df[
        "F1_Std"
    ] *= 100

    display_df[
        "ROC_AUC"
    ] *= 100

    display_df[
        "ROC_AUC_Std"
    ] *= 100

    print()

    print(
        display_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.2f}%".format,
                "Accuracy_Std": "{:.2f}%".format,
                "F1": "{:.2f}%".format,
                "F1_Std": "{:.2f}%".format,
                "ROC_AUC": "{:.2f}%".format,
                "ROC_AUC_Std": "{:.2f}%".format,
            },
        )
    )


    # =====================================================
    # MAJORITY BASELINE
    # =====================================================

    majority_class = (
        y_train.value_counts()
        .idxmax()
    )

    majority_accuracy = (
        y_train == majority_class
    ).mean()

    print_header(
        "7. MAJORITY BASELINE"
    )

    print(
        f"Majority class: "
        f"{'UP' if majority_class == 1 else 'DOWN'}"
    )

    print(
        f"Training baseline accuracy: "
        f"{majority_accuracy:.2%}"
    )


    # =====================================================
    # FIND WINNERS
    # =====================================================

    print_header(
        "8. WINNER ANALYSIS"
    )

    # Best Accuracy
    best_accuracy_row = results_df.loc[
        results_df["Accuracy"].idxmax()
    ]

    # Best F1
    best_f1_row = results_df.loc[
        results_df["F1"].idxmax()
    ]

    # Best ROC-AUC
    best_auc_row = results_df.loc[
        results_df["ROC_AUC"].idxmax()
    ]

    print()
    print(
        f"Best Accuracy : "
        f"{best_accuracy_row['Feature_Set']} "
        f"({best_accuracy_row['Accuracy']:.2%})"
    )

    print(
        f"Best F1       : "
        f"{best_f1_row['Feature_Set']} "
        f"({best_f1_row['F1']:.4f})"
    )

    print(
        f"Best ROC-AUC  : "
        f"{best_auc_row['Feature_Set']} "
        f"({best_auc_row['ROC_AUC']:.4f})"
    )


    # =====================================================
    # SCORE FEATURE SETS
    #
    # We use average ROC-AUC as the primary selection
    # criterion because this is a directional classifier
    # and accuracy alone can be misleading.
    # =====================================================

    ranked_results = (
        results_df
        .sort_values(
            by="ROC_AUC",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print()
    print(
        "Ranking by average ROC-AUC:"
    )

    print()

    for i, row in ranked_results.iterrows():

        print(
            f"{i + 1}. "
            f"{row['Feature_Set']:<28} "
            f"ROC-AUC={row['ROC_AUC']:.4f} | "
            f"Accuracy={row['Accuracy']:.2%} | "
            f"F1={row['F1']:.4f}"
        )


    # =====================================================
    # SAVE RESULTS
    # =====================================================

    output_path = (
        "tests/feature_subset_validation_results.csv"
    )

    save_df = results_df[
        [
            "Feature_Set",
            "Feature_Count",
            "Accuracy",
            "Accuracy_Std",
            "F1",
            "F1_Std",
            "ROC_AUC",
            "ROC_AUC_Std",
        ]
    ].copy()

    save_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Results saved to: {output_path}"
    )


    # =====================================================
    # SAVE FEATURE RANKING
    # =====================================================

    importance_path = (
        "tests/training_only_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    print(
        f"Feature ranking saved to: "
        f"{importance_path}"
    )


    # =====================================================
    # IMPORTANT CONCLUSION
    # =====================================================

    print_header(
        "9. MILESTONE 3.14 CONCLUSION"
    )

    winner = ranked_results.iloc[0]

    print()

    print(
        f"🏆 WALK-FORWARD WINNER: "
        f"{winner['Feature_Set']}"
    )

    print(
        f"   Features : "
        f"{int(winner['Feature_Count'])}"
    )

    print(
        f"   Accuracy : "
        f"{winner['Accuracy']:.2%}"
    )

    print(
        f"   F1       : "
        f"{winner['F1']:.4f}"
    )

    print(
        f"   ROC-AUC  : "
        f"{winner['ROC_AUC']:.4f}"
    )

    print()

    print(
        "The final holdout was NOT used to select "
        "the winning feature subset."
    )

    print()

    print(
        "NEXT STEP:"
    )

    print(
        "Milestone 3.15 — Final validation of the "
        "winning feature set on the untouched holdout."
    )

    print()

    print(
        "=" * 70
    )

    print(
        "Milestone 3.14 complete."
    )

    print(
        "=" * 70
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()