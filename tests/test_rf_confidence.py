import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

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


# =========================================================
# HEADER
# =========================================================

print("=" * 70)
print("RANDOM FOREST PROBABILITY & CONFIDENCE ANALYSIS")
print("=" * 70)


# =========================================================
# INITIALIZE SERVICES
# =========================================================

prediction_service = PredictionService()


# =========================================================
# LOAD MARKET DATA
# =========================================================

print(f"\n[1/7] Loading {TICKER} market data...")

market_data = fetch_stock_data(
    TICKER,
    period="2y"
)

if market_data is None or market_data.empty:
    raise ValueError("No market data received.")

print(f"Loaded {len(market_data)} rows")


# =========================================================
# PREPARE FEATURES
# =========================================================

print("\n[2/7] Preparing technical features...")

features = prediction_service.prepare_features(
    market_data
)

print(f"Feature dataframe shape: {features.shape}")


# =========================================================
# PREPARE ML DATA
# =========================================================

print("\n[3/7] Preparing machine-learning data...")

X, y = prediction_service.prepare_ml_data(
    features
)
y = (y > 0).astype(int)
feature_columns = prediction_service.get_feature_columns()

print(f"Features available: {len(feature_columns)}")
print(f"ML rows: {len(X)}")


# =========================================================
# VERIFY FEATURE COUNT
# =========================================================

if len(feature_columns) != 30:

    print(
        f"\nWARNING: Expected 30 features, "
        f"but PredictionService returned {len(feature_columns)}."
    )

print("\nFeatures used:")

for i, feature in enumerate(feature_columns, start=1):
    print(f"{i:2}. {feature}")


# =========================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# =========================================================

print("\n[4/7] Creating chronological train/test split...")

X_train = X.iloc[:TRAIN_SIZE].copy()
y_train = y.iloc[:TRAIN_SIZE].copy()

X_test = X.iloc[TRAIN_SIZE:].copy()
y_test = y.iloc[TRAIN_SIZE:].copy()

print(f"\nTraining rows : {len(X_train)}")
print(f"Test rows     : {len(X_test)}")

print(
    f"\nTraining UP   : {int(y_train.sum())}"
)

print(
    f"Training DOWN : {int(len(y_train) - y_train.sum())}"
)

print(
    f"Test UP       : {int(y_test.sum())}"
)

print(
    f"Test DOWN     : {int(len(y_test) - y_test.sum())}"
)


# =========================================================
# TRAIN TUNED RANDOM FOREST
# =========================================================

print("\n[5/7] Training tuned Random Forest...")

model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("\nRandom Forest trained successfully.")

print("\nConfiguration:")

print(f"n_estimators      = {N_ESTIMATORS}")
print(f"max_depth         = {MAX_DEPTH}")
print(f"min_samples_split = {MIN_SAMPLES_SPLIT}")
print(f"min_samples_leaf  = {MIN_SAMPLES_LEAF}")


# =========================================================
# PROBABILITY PREDICTIONS
# =========================================================

print("\n[6/7] Generating probability predictions...")

probabilities = model.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= 0.50
).astype(int)


# =========================================================
# BASIC PERFORMANCE
# =========================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

print("\n" + "=" * 70)
print("FINAL HOLDOUT PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# =========================================================
# BUILD RESULTS DATAFRAME
# =========================================================

results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": predictions,
    "Probability_UP": probabilities
})

results["Predicted_Direction"] = np.where(
    results["Predicted"] == 1,
    "UP",
    "DOWN"
)

# Confidence means probability of the predicted class.
#
# Example:
# UP probability = 0.72
# DOWN probability = 0.28
# prediction = UP
# confidence = 72%
#
# Example:
# UP probability = 0.31
# DOWN probability = 0.69
# prediction = DOWN
# confidence = 69%

results["Confidence"] = np.maximum(
    results["Probability_UP"],
    1 - results["Probability_UP"]
)

results["Correct"] = (
    results["Actual"] ==
    results["Predicted"]
)


# =========================================================
# INDIVIDUAL PREDICTIONS
# =========================================================

print("\n" + "=" * 70)
print("INDIVIDUAL TEST PREDICTIONS")
print("=" * 70)

display_results = results.copy()

display_results["Probability_UP"] = (
    display_results["Probability_UP"] * 100
).round(2)

display_results["Confidence"] = (
    display_results["Confidence"] * 100
).round(2)

print()

print(
    display_results[
        [
            "Actual",
            "Predicted_Direction",
            "Probability_UP",
            "Confidence",
            "Correct"
        ]
    ].to_string(index=False)
)


# =========================================================
# CONFIDENCE BAND ANALYSIS
# =========================================================

bins = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    1.01
]

labels = [
    "50-55%",
    "55-60%",
    "60-65%",
    "65-70%",
    "70-75%",
    "75%+"
]

results["Confidence_Band"] = pd.cut(
    results["Confidence"],
    bins=bins,
    labels=labels,
    right=False
)

print("\n" + "=" * 70)
print("CONFIDENCE BAND ANALYSIS")
print("=" * 70)

print()

print(
    f"{'Confidence':<15}"
    f"{'Predictions':<15}"
    f"{'Accuracy':<15}"
    f"{'Avg Confidence':<18}"
)

print("-" * 63)

for label in labels:

    bucket = results[
        results["Confidence_Band"] == label
    ]

    count = len(bucket)

    if count == 0:

        print(
            f"{label:<15}"
            f"{0:<15}"
            f"{'N/A':<15}"
            f"{'N/A':<18}"
        )

        continue

    bucket_accuracy = (
        bucket["Correct"].mean()
    )

    avg_confidence = (
        bucket["Confidence"].mean()
    )

    print(
        f"{label:<15}"
        f"{count:<15}"
        f"{bucket_accuracy * 100:>6.2f}%{'':<8}"
        f"{avg_confidence * 100:>6.2f}%"
    )


# =========================================================
# HIGH-CONFIDENCE ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE ANALYSIS")
print("=" * 70)

thresholds = [
    0.55,
    0.60,
    0.65,
    0.70,
    0.75
]

for threshold in thresholds:

    high_conf = results[
        results["Confidence"] >= threshold
    ]

    if len(high_conf) == 0:
        continue

    high_accuracy = (
        high_conf["Correct"].mean()
    )

    coverage = (
        len(high_conf) /
        len(results)
    )

    print(
        f"\nConfidence >= {threshold * 100:.0f}%"
    )

    print(
        f"Predictions : "
        f"{len(high_conf)} / {len(results)}"
    )

    print(
        f"Coverage    : "
        f"{coverage * 100:.2f}%"
    )

    print(
        f"Accuracy    : "
        f"{high_accuracy * 100:.2f}%"
    )


# =========================================================
# PROBABILITY CALIBRATION ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("PROBABILITY CALIBRATION CHECK")
print("=" * 70)

print(
    "\nFor UP probabilities, a well-calibrated model should have:"
)

print(
    "60% predicted probability -> roughly 60% actual UP outcomes"
)

print(
    "70% predicted probability -> roughly 70% actual UP outcomes"
)

print(
    "80% predicted probability -> roughly 80% actual UP outcomes"
)


calibration_bins = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.01
]

calibration_labels = [
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%"
]

results["Calibration_Band"] = pd.cut(
    results["Probability_UP"],
    bins=calibration_bins,
    labels=calibration_labels,
    right=False
)

print()

print(
    f"{'Probability UP':<18}"
    f"{'Count':<10}"
    f"{'Avg Probability':<18}"
    f"{'Actual UP Rate':<18}"
)

print("-" * 65)

for label in calibration_labels:

    bucket = results[
        results["Calibration_Band"] == label
    ]

    if len(bucket) == 0:

        print(
            f"{label:<18}"
            f"{0:<10}"
            f"{'N/A':<18}"
            f"{'N/A':<18}"
        )

        continue

    avg_probability = (
        bucket["Probability_UP"].mean()
    )

    actual_up_rate = (
        bucket["Actual"].mean()
    )

    print(
        f"{label:<18}"
        f"{len(bucket):<10}"
        f"{avg_probability * 100:>6.2f}%{'':<11}"
        f"{actual_up_rate * 100:>6.2f}%"
    )


# =========================================================
# BEST CONFIDENCE BAND
# =========================================================

print("\n" + "=" * 70)
print("CONFIDENCE CONCLUSION")
print("=" * 70)

best_bucket = None
best_accuracy = -1

for label in labels:

    bucket = results[
        results["Confidence_Band"] == label
    ]

    # Require at least 5 observations.
    if len(bucket) >= 5:

        bucket_accuracy = (
            bucket["Correct"].mean()
        )

        if bucket_accuracy > best_accuracy:

            best_accuracy = bucket_accuracy
            best_bucket = label


if best_bucket is not None:

    print(
        f"\nBest confidence band "
        f"(minimum 5 predictions): {best_bucket}"
    )

    print(
        f"Accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

else:

    print(
        "\nNo confidence band contained "
        "at least 5 predictions."
    )


# =========================================================
# HIGH-CONFIDENCE DECISION
# =========================================================

high_conf_70 = results[
    results["Confidence"] >= 0.70
]

print("\n" + "=" * 70)
print("SHOULD STOCKSENSE TRUST 70%+ CONFIDENCE?")
print("=" * 70)

if len(high_conf_70) > 0:

    accuracy_70 = (
        high_conf_70["Correct"].mean()
    )

    coverage_70 = (
        len(high_conf_70) /
        len(results)
    )

    print(
        f"\n70%+ predictions : {len(high_conf_70)}"
    )

    print(
        f"Coverage         : "
        f"{coverage_70 * 100:.2f}%"
    )

    print(
        f"Accuracy         : "
        f"{accuracy_70 * 100:.2f}%"
    )

    if accuracy_70 >= 0.65:

        print(
            "\nResult: Higher confidence shows "
            "useful predictive separation."
        )

    else:

        print(
            "\nResult: 70%+ confidence is NOT yet "
            "reliable enough to be treated as strong confidence."
        )

else:

    print(
        "\nThe model produced no predictions "
        "with 70%+ confidence."
    )


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 70)
print("MILESTONE 3.11 COMPLETE")
print("=" * 70)