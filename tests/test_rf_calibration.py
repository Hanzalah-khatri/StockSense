import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss
)

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

CALIBRATION_FOLDS = 3


# =========================================================
# HEADER
# =========================================================

print("=" * 70)
print("MILESTONE 3.12 — RANDOM FOREST PROBABILITY CALIBRATION")
print("=" * 70)


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
# 2. PREPARE FEATURES
# =========================================================

print("\n[2/8] Preparing technical features...")

prediction_service = PredictionService()

features = prediction_service.prepare_features(
    market_data
)

print(f"Feature dataframe shape: {features.shape}")


# =========================================================
# 3. PREPARE ML DATA
# =========================================================

print("\n[3/8] Preparing machine-learning data...")

X, y = prediction_service.prepare_ml_data(
    features
)

# Convert continuous next-day return
# into classification labels:
#
# 1 = UP
# 0 = DOWN

y = (y > 0).astype(int)

print(f"Features available: {X.shape[1]}")
print(f"ML rows: {len(X)}")

print("\nTarget distribution:")
print(f"UP   : {(y == 1).sum()}")
print(f"DOWN : {(y == 0).sum()}")


# =========================================================
# 4. CHRONOLOGICAL SPLIT
# =========================================================

print("\n[4/8] Creating chronological train/test split...")

X_train = X.iloc[:TRAIN_SIZE].copy()
X_test = X.iloc[TRAIN_SIZE:].copy()

y_train = y.iloc[:TRAIN_SIZE].copy()
y_test = y.iloc[TRAIN_SIZE:].copy()

print(f"\nTraining rows : {len(X_train)}")
print(f"Test rows     : {len(X_test)}")

print(f"Training UP   : {(y_train == 1).sum()}")
print(f"Training DOWN : {(y_train == 0).sum()}")

print(f"Test UP       : {(y_test == 1).sum()}")
print(f"Test DOWN     : {(y_test == 0).sum()}")


# =========================================================
# 5. TRAIN BASE RANDOM FOREST
# =========================================================

print("\n[5/8] Training tuned Random Forest...")

base_rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=42,
    n_jobs=-1
)

base_rf.fit(
    X_train,
    y_train
)

print("Base Random Forest trained successfully.")


# =========================================================
# 6. CALIBRATE USING TRAINING DATA ONLY
# =========================================================

print("\n[6/8] Calibrating Random Forest probabilities...")

print(
    f"Calibration method : sigmoid"
)

print(
    f"Calibration folds  : {CALIBRATION_FOLDS}"
)

calibrated_rf = CalibratedClassifierCV(
    estimator=base_rf,
    method="sigmoid",
    cv=CALIBRATION_FOLDS
)

calibrated_rf.fit(
    X_train,
    y_train
)

print("Calibrated Random Forest trained successfully.")


# =========================================================
# 7. EVALUATE ON UNTOUCHED HOLDOUT
# =========================================================

print("\n[7/8] Evaluating on untouched final holdout...")

# ---------------------------------------------------------
# BASE MODEL
# ---------------------------------------------------------

base_probabilities = base_rf.predict_proba(
    X_test
)[:, 1]

base_predictions = (
    base_probabilities >= 0.50
).astype(int)

base_accuracy = accuracy_score(
    y_test,
    base_predictions
)

base_auc = roc_auc_score(
    y_test,
    base_probabilities
)

base_brier = brier_score_loss(
    y_test,
    base_probabilities
)


# ---------------------------------------------------------
# CALIBRATED MODEL
# ---------------------------------------------------------

calibrated_probabilities = calibrated_rf.predict_proba(
    X_test
)[:, 1]

calibrated_predictions = (
    calibrated_probabilities >= 0.50
).astype(int)

calibrated_accuracy = accuracy_score(
    y_test,
    calibrated_predictions
)

calibrated_auc = roc_auc_score(
    y_test,
    calibrated_probabilities
)

calibrated_brier = brier_score_loss(
    y_test,
    calibrated_probabilities
)


# =========================================================
# PERFORMANCE COMPARISON
# =========================================================

print("\n" + "=" * 70)
print("BASE RF vs CALIBRATED RF")
print("=" * 70)

print(
    f"\n{'Metric':<25}"
    f"{'Base RF':>15}"
    f"{'Calibrated RF':>20}"
)

print("-" * 60)

print(
    f"{'Accuracy':<25}"
    f"{base_accuracy * 100:>14.2f}%"
    f"{calibrated_accuracy * 100:>19.2f}%"
)

print(
    f"{'ROC-AUC':<25}"
    f"{base_auc:>15.4f}"
    f"{calibrated_auc:>20.4f}"
)

print(
    f"{'Brier Score':<25}"
    f"{base_brier:>15.4f}"
    f"{calibrated_brier:>20.4f}"
)


# =========================================================
# 8. CALIBRATED PROBABILITY ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("CALIBRATED PROBABILITY DISTRIBUTION")
print("=" * 70)

result = pd.DataFrame({
    "Actual": y_test.values,
    "Probability_UP": calibrated_probabilities
})

result["Predicted_Direction"] = np.where(
    result["Probability_UP"] >= 0.50,
    "UP",
    "DOWN"
)

result["Confidence"] = np.maximum(
    result["Probability_UP"],
    1 - result["Probability_UP"]
)

result["Correct"] = (
    result["Predicted_Direction"]
    ==
    np.where(
        result["Actual"] == 1,
        "UP",
        "DOWN"
    )
)


print(
    f"\nAverage UP probability : "
    f"{result['Probability_UP'].mean() * 100:.2f}%"
)

print(
    f"Minimum UP probability : "
    f"{result['Probability_UP'].min() * 100:.2f}%"
)

print(
    f"Maximum UP probability : "
    f"{result['Probability_UP'].max() * 100:.2f}%"
)


# =========================================================
# CONFIDENCE BANDS
# =========================================================

print("\n" + "=" * 70)
print("CALIBRATED CONFIDENCE BAND ANALYSIS")
print("=" * 70)

bands = [
    (0.50, 0.55, "50-55%"),
    (0.55, 0.60, "55-60%"),
    (0.60, 0.65, "60-65%"),
    (0.65, 0.70, "65-70%"),
    (0.70, 0.75, "70-75%"),
    (0.75, 1.01, "75%+"),
]

print(
    f"\n{'Confidence':<15}"
    f"{'Predictions':<15}"
    f"{'Accuracy':<15}"
    f"{'Avg Confidence':<20}"
)

print("-" * 65)

for lower, upper, label in bands:

    mask = (
        (result["Confidence"] >= lower)
        &
        (result["Confidence"] < upper)
    )

    subset = result[mask]

    count = len(subset)

    if count > 0:

        accuracy = subset["Correct"].mean()

        avg_confidence = subset["Confidence"].mean()

        print(
            f"{label:<15}"
            f"{count:<15}"
            f"{accuracy * 100:>7.2f}%"
            f"{avg_confidence * 100:>15.2f}%"
        )

    else:

        print(
            f"{label:<15}"
            f"{0:<15}"
            f"{'N/A':>10}"
            f"{'N/A':>15}"
        )


# =========================================================
# CALIBRATION BINS
# =========================================================

print("\n" + "=" * 70)
print("CALIBRATED PROBABILITY CHECK")
print("=" * 70)

probability_bins = [
    (0.50, 0.60, "50-60%"),
    (0.60, 0.70, "60-70%"),
    (0.70, 0.80, "70-80%"),
    (0.80, 0.90, "80-90%"),
    (0.90, 1.01, "90-100%"),
]

print(
    f"\n{'Probability UP':<18}"
    f"{'Count':<10}"
    f"{'Avg Probability':<20}"
    f"{'Actual UP Rate':<20}"
)

print("-" * 68)

for lower, upper, label in probability_bins:

    mask = (
        (result["Probability_UP"] >= lower)
        &
        (result["Probability_UP"] < upper)
    )

    subset = result[mask]

    count = len(subset)

    if count > 0:

        avg_probability = subset[
            "Probability_UP"
        ].mean()

        actual_up_rate = subset[
            "Actual"
        ].mean()

        print(
            f"{label:<18}"
            f"{count:<10}"
            f"{avg_probability * 100:>8.2f}%"
            f"{actual_up_rate * 100:>17.2f}%"
        )

    else:

        print(
            f"{label:<18}"
            f"{0:<10}"
            f"{'N/A':>15}"
            f"{'N/A':>20}"
        )


# =========================================================
# HIGH CONFIDENCE
# =========================================================

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE ANALYSIS")
print("=" * 70)

for threshold in [0.55, 0.60, 0.65, 0.70]:

    subset = result[
        result["Confidence"] >= threshold
    ]

    count = len(subset)

    coverage = count / len(result)

    if count > 0:

        accuracy = subset["Correct"].mean()

        print(
            f"\nConfidence >= {threshold * 100:.0f}%"
        )

        print(
            f"Predictions : {count} / {len(result)}"
        )

        print(
            f"Coverage    : {coverage * 100:.2f}%"
        )

        print(
            f"Accuracy    : {accuracy * 100:.2f}%"
        )

    else:

        print(
            f"\nConfidence >= {threshold * 100:.0f}%"
        )

        print("Predictions : 0")


# =========================================================
# FINAL CONCLUSION
# =========================================================

print("\n" + "=" * 70)
print("MILESTONE 3.12 CONCLUSION")
print("=" * 70)

if calibrated_brier < base_brier:

    print(
        "\nProbability calibration IMPROVED the Brier Score."
    )

else:

    print(
        "\nProbability calibration did NOT improve the Brier Score."
    )

if calibrated_auc > base_auc:

    print(
        "Calibrated ROC-AUC improved."
    )

elif calibrated_auc < base_auc:

    print(
        "Calibrated ROC-AUC decreased slightly."
    )

else:

    print(
        "ROC-AUC remained unchanged."
    )

print(
    "\nThe final 61-row holdout remained untouched during calibration."
)

print(
    "\nMILESTONE 3.12 COMPLETE"
)