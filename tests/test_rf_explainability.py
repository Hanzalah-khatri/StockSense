import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

from services.market_data import fetch_stock_data
from services.prediction_service import PredictionService


# =========================================================
# CONFIGURATION
# =========================================================

TICKER = "AAPL"

TRAIN_SIZE = 240

N_ESTIMATORS = 200
MAX_DEPTH = 15
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 1

RANDOM_STATE = 42

TOP_N = 15

PERMUTATION_REPEATS = 20


# =========================================================
# HEADER
# =========================================================

print("=" * 70)
print("MILESTONE 3.13 — RANDOM FOREST EXPLAINABILITY")
print("=" * 70)

print("\nPurpose:")
print("Identify which technical features drive the")
print("StockSense direction classifier.")


# =========================================================
# 1. LOAD MARKET DATA
# =========================================================

print("\n[1/8] Loading AAPL market data...")

market_data = fetch_stock_data(
    TICKER,
    period="2y"
)

print(f"Loaded {len(market_data)} rows")


# =========================================================
# 2. PREPARE TECHNICAL FEATURES
# =========================================================

print("\n[2/8] Preparing technical features...")

prediction_service = PredictionService()

features = prediction_service.prepare_features(
    market_data
)

print(
    f"Feature dataframe shape: {features.shape}"
)


# =========================================================
# 3. PREPARE ML DATA
# =========================================================

print("\n[3/8] Preparing machine-learning data...")

X, y = prediction_service.prepare_ml_data(
    features
)

# Convert continuous next-day return
# into classification labels.
#
# 1 = UP
# 0 = DOWN

y = (y > 0).astype(int)

feature_columns = (
    prediction_service.get_feature_columns()
)

print(
    f"Features available: {len(feature_columns)}"
)

print(
    f"ML rows: {len(X)}"
)

print("\nFeatures used:")

for i, feature in enumerate(
    feature_columns,
    start=1
):
    print(
        f"{i:2d}. {feature}"
    )


# =========================================================
# 4. CHRONOLOGICAL TRAIN / TEST SPLIT
# =========================================================

print(
    "\n[4/8] Creating chronological train/test split..."
)

X_train = X.iloc[:TRAIN_SIZE].copy()
X_test = X.iloc[TRAIN_SIZE:].copy()

y_train = y.iloc[:TRAIN_SIZE].copy()
y_test = y.iloc[TRAIN_SIZE:].copy()

print(
    f"\nTraining rows : {len(X_train)}"
)

print(
    f"Test rows     : {len(X_test)}"
)

print(
    f"Training UP   : {(y_train == 1).sum()}"
)

print(
    f"Training DOWN : {(y_train == 0).sum()}"
)

print(
    f"Test UP       : {(y_test == 1).sum()}"
)

print(
    f"Test DOWN     : {(y_test == 0).sum()}"
)


# =========================================================
# 5. TRAIN TUNED RANDOM FOREST
# =========================================================

print(
    "\n[5/8] Training tuned Random Forest..."
)

model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print(
    "Random Forest trained successfully."
)

print("\nConfiguration:")

print(
    f"n_estimators      = {N_ESTIMATORS}"
)

print(
    f"max_depth         = {MAX_DEPTH}"
)

print(
    f"min_samples_split = {MIN_SAMPLES_SPLIT}"
)

print(
    f"min_samples_leaf  = {MIN_SAMPLES_LEAF}"
)


# =========================================================
# 6. VERIFY HOLDOUT PERFORMANCE
# =========================================================

print(
    "\n[6/8] Verifying final holdout performance..."
)

test_predictions = model.predict(
    X_test
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print(
    f"\nFinal holdout accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# =========================================================
# 7. BUILT-IN RANDOM FOREST FEATURE IMPORTANCE
# =========================================================

print(
    "\n[7/8] Calculating built-in feature importance..."
)

feature_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

feature_importance[
    "Importance_Percent"
] = (
    feature_importance["Importance"] * 100
)


print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

print(
    f"\n{'Rank':<7}"
    f"{'Feature':<25}"
    f"{'Importance':<15}"
)

print("-" * 50)

for rank, row in feature_importance.head(
    TOP_N
).iterrows():

    print(
        f"{rank + 1:<7}"
        f"{row['Feature']:<25}"
        f"{row['Importance_Percent']:>8.2f}%"
    )


# =========================================================
# TOP 5 FEATURES
# =========================================================

print("\n" + "=" * 70)
print("TOP 5 FEATURES")
print("=" * 70)

for rank, row in feature_importance.head(
    5
).iterrows():

    print(
        f"{rank + 1}. "
        f"{row['Feature']} "
        f"({row['Importance_Percent']:.2f}%)"
    )


# =========================================================
# IMPORTANCE CONCENTRATION
# =========================================================

top_5_importance = (
    feature_importance
    .head(5)["Importance"]
    .sum()
)

top_10_importance = (
    feature_importance
    .head(10)["Importance"]
    .sum()
)

print("\n" + "=" * 70)
print("IMPORTANCE CONCENTRATION")
print("=" * 70)

print(
    f"\nTop 5 features account for  "
    f"{top_5_importance * 100:.2f}% "
    f"of total importance."
)

print(
    f"Top 10 features account for "
    f"{top_10_importance * 100:.2f}% "
    f"of total importance."
)


# =========================================================
# 8. PERMUTATION IMPORTANCE
# =========================================================

print(
    "\n[8/8] Calculating permutation importance..."
)

print(
    "\nThis evaluates how much holdout accuracy"
)

print(
    "drops when each feature is randomly shuffled."
)

print(
    f"\nPermutation repeats: "
    f"{PERMUTATION_REPEATS}"
)

permutation_result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="accuracy",
    n_repeats=PERMUTATION_REPEATS,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

permutation_importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Permutation_Mean":
        permutation_result.importances_mean,
    "Permutation_STD":
        permutation_result.importances_std
})

permutation_importance_df = (
    permutation_importance_df
    .sort_values(
        "Permutation_Mean",
        ascending=False
    )
    .reset_index(drop=True)
)


# =========================================================
# PERMUTATION RESULTS
# =========================================================

print("\n" + "=" * 70)
print("PERMUTATION FEATURE IMPORTANCE")
print("=" * 70)

print(
    f"\n{'Rank':<7}"
    f"{'Feature':<25}"
    f"{'Mean Accuracy Drop':<22}"
    f"{'Std':<12}"
)

print("-" * 70)

for rank, row in permutation_importance_df.head(
    TOP_N
).iterrows():

    print(
        f"{rank + 1:<7}"
        f"{row['Feature']:<25}"
        f"{row['Permutation_Mean'] * 100:>10.2f}%"
        f"{row['Permutation_STD'] * 100:>15.2f}%"
    )


# =========================================================
# POSITIVE PERMUTATION FEATURES
# =========================================================

positive_permutation = (
    permutation_importance_df[
        permutation_importance_df[
            "Permutation_Mean"
        ] > 0
    ]
)

print("\n" + "=" * 70)
print("FEATURES WITH POSITIVE PERMUTATION IMPORTANCE")
print("=" * 70)

print(
    f"\n{len(positive_permutation)} "
    f"of {len(feature_columns)} features "
    f"produced a positive average accuracy contribution."
)

if len(positive_permutation) > 0:

    for rank, row in positive_permutation.head(
        10
    ).iterrows():

        print(
            f"{rank + 1}. "
            f"{row['Feature']} "
            f"→ "
            f"{row['Permutation_Mean'] * 100:.2f}% "
            f"accuracy drop"
        )


# =========================================================
# NEGATIVE PERMUTATION FEATURES
# =========================================================

negative_permutation = (
    permutation_importance_df[
        permutation_importance_df[
            "Permutation_Mean"
        ] < 0
    ]
)

print("\n" + "=" * 70)
print("FEATURES WITH NEGATIVE PERMUTATION IMPORTANCE")
print("=" * 70)

print(
    f"\n{len(negative_permutation)} "
    f"features produced a negative average "
    f"accuracy contribution."
)

if len(negative_permutation) > 0:

    print(
        "\nThese features may be adding noise "
        "on the final holdout."
    )

    for rank, row in negative_permutation.tail(
        10
    ).iterrows():

        print(
            f"- {row['Feature']} "
            f"→ "
            f"{row['Permutation_Mean'] * 100:.2f}%"
        )


# =========================================================
# COMPARE BOTH IMPORTANCE METHODS
# =========================================================

comparison = feature_importance.merge(
    permutation_importance_df,
    on="Feature"
)

comparison = comparison[
    [
        "Feature",
        "Importance",
        "Permutation_Mean",
        "Permutation_STD"
    ]
]

comparison = comparison.sort_values(
    "Importance",
    ascending=False
)


print("\n" + "=" * 70)
print("FEATURE IMPORTANCE COMPARISON")
print("=" * 70)

print(
    "\nThe two methods answer different questions:"
)

print(
    "\nRandom Forest importance:"
)

print(
    "How often the feature helps split "
    "the decision trees."
)

print(
    "\nPermutation importance:"
)

print(
    "How much holdout accuracy changes "
    "when the feature is shuffled."
)


print(
    f"\n{'Feature':<25}"
    f"{'RF Importance':<18}"
    f"{'Permutation Drop':<20}"
)

print("-" * 65)

for _, row in comparison.head(
    TOP_N
).iterrows():

    print(
        f"{row['Feature']:<25}"
        f"{row['Importance'] * 100:>8.2f}%"
        f"{row['Permutation_Mean'] * 100:>15.2f}%"
    )


# =========================================================
# FINAL INTERPRETATION
# =========================================================

print("\n" + "=" * 70)
print("MILESTONE 3.13 — INTERPRETATION")
print("=" * 70)

top_rf_feature = (
    feature_importance.iloc[0]["Feature"]
)

top_rf_value = (
    feature_importance.iloc[0][
        "Importance_Percent"
    ]
)

top_perm_feature = (
    permutation_importance_df.iloc[0]["Feature"]
)

top_perm_value = (
    permutation_importance_df.iloc[0][
        "Permutation_Mean"
    ]
)

print(
    f"\nMost important RF feature:"
)

print(
    f"  {top_rf_feature} "
    f"({top_rf_value:.2f}%)"
)

print(
    f"\nStrongest permutation feature:"
)

print(
    f"  {top_perm_feature} "
    f"({top_perm_value * 100:.2f}% "
    f"average accuracy drop)"
)


if top_perm_value > 0:

    print(
        "\nThe strongest permutation feature "
        "provides measurable predictive information "
        "on the final holdout."
    )

else:

    print(
        "\nThe strongest permutation feature "
        "does not provide a positive average "
        "accuracy contribution on the holdout."
    )


if top_5_importance > 0.50:

    print(
        "\nObservation:"
    )

    print(
        "The model's tree-based importance is "
        "highly concentrated in the top 5 features."
    )

else:

    print(
        "\nObservation:"
    )

    print(
        "Feature importance is relatively "
        "distributed across multiple features."
    )


print(
    "\nImportant:"
)

print(
    "Feature importance does NOT prove that a "
    "feature causes stock movements."
)

print(
    "It only describes how the trained model "
    "uses the available features."
)


# =========================================================
# SAVE RESULTS
# =========================================================

feature_importance.to_csv(
    "tests/rf_feature_importance.csv",
    index=False
)

permutation_importance_df.to_csv(
    "tests/rf_permutation_importance.csv",
    index=False
)

print(
    "\nSaved:"
)

print(
    "  tests/rf_feature_importance.csv"
)

print(
    "  tests/rf_permutation_importance.csv"
)


# =========================================================
# COMPLETE
# =========================================================

print("\n" + "=" * 70)
print("MILESTONE 3.13 COMPLETE")
print("=" * 70)