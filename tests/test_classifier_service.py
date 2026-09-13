"""
StockSense — Milestone 4.1
Production Classifier Service Test
"""

from services.classifier_service import (
    ClassifierService,
)

from services.market_data import (
    fetch_stock_data,
)

from services.prediction_service import (
    PredictionService,
)


def main():

    print("=" * 70)
    print("STOCKSENSE — MILESTONE 4.1")
    print("=" * 70)

    print()
    print("PRODUCTION CLASSIFIER SERVICE TEST")

    # =====================================================
    # 1. LOAD CLASSIFIER
    # =====================================================

    print()
    print("-" * 70)
    print("1. LOADING LOCKED MODEL")
    print("-" * 70)

    classifier = (
        ClassifierService()
    )

    print()
    print(
        "✓ Locked model loaded successfully."
    )

    # =====================================================
    # 2. MODEL INFORMATION
    # =====================================================

    print()
    print("-" * 70)
    print("2. MODEL INFORMATION")
    print("-" * 70)

    info = (
        classifier.get_model_info()
    )

    print()

    print(
        f"Model: "
        f"{info['model_name']}"
    )

    print(
        f"Type: "
        f"{info['model_type']}"
    )

    print(
        f"Features: "
        f"{info['feature_count']}"
    )

    print(
        f"Training rows: "
        f"{info['training_rows']}"
    )

    # =====================================================
    # 3. FETCH MARKET DATA
    # =====================================================

    print()
    print("-" * 70)
    print("3. FETCHING AAPL MARKET DATA")
    print("-" * 70)

    market_data = fetch_stock_data(
        "AAPL",
        period="2y",
    )

    print()

    print(
        f"Market rows: "
        f"{len(market_data)}"
    )

    # =====================================================
    # 4. PREPARE FEATURES
    # =====================================================

    print()
    print("-" * 70)
    print("4. PREPARING FEATURES")
    print("-" * 70)

    prediction_service = (
        PredictionService()
    )

    features = (
        prediction_service.prepare_features(
            market_data
        )
    )

    print()

    print(
        f"Feature rows: "
        f"{len(features)}"
    )

    # =====================================================
    # 5. RUN PREDICTION
    # =====================================================

    print()
    print("-" * 70)
    print("5. RUNNING PRODUCTION PREDICTION")
    print("-" * 70)

    result = classifier.predict(
        features
    )

    print()

    print(
        f"Direction: "
        f"{result['direction']}"
    )

    print(
        f"Probability UP: "
        f"{result['probability_up']:.2%}"
    )

    print(
        f"Probability DOWN: "
        f"{result['probability_down']:.2%}"
    )

    print(
        f"Confidence: "
        f"{result['confidence_percent']:.2f}%"
    )

    # =====================================================
    # 6. VALIDATE RESULT
    # =====================================================

    print()
    print("-" * 70)
    print("6. VALIDATING RESULT")
    print("-" * 70)

    assert result["direction"] in [
        "UP",
        "DOWN",
    ]

    assert (
        0
        <= result["probability_up"]
        <= 1
    )

    assert (
        0
        <= result["probability_down"]
        <= 1
    )

    assert abs(
        (
            result["probability_up"]
            + result["probability_down"]
        ) - 1
    ) < 1e-6

    assert (
        0
        <= result["confidence"]
        <= 1
    )

    print()

    print(
        "✓ Direction valid"
    )

    print(
        "✓ Probabilities valid"
    )

    print(
        "✓ Probabilities sum to 100%"
    )

    print(
        "✓ Confidence valid"
    )

    # =====================================================
    # 7. COMPLETE
    # =====================================================

    print()
    print("=" * 70)
    print("MILESTONE 4.1 COMPLETE")
    print("=" * 70)

    print()

    print(
        "The locked Random Forest is now "
        "available through a production service."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Milestone 4.2 — Integrate classifier "
        "into Flask /api/predict"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()