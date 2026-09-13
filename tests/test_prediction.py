"""
Tests for StockSense machine-learning pipeline.
"""

from services.market_data import fetch_stock_data
from models.feature_engineering import FeatureEngineering
from models.model_manager import ModelManager
from models.model_evaluation import ModelEvaluator
from services.prediction_service import PredictionService

def test_feature_engineering():
    """Verify feature generation."""

    data = fetch_stock_data(
        "AAPL",
        period="1y"
    )

    features = FeatureEngineering(data).build_features()

    assert not features.empty

    expected_features = [
        "Daily_Return",
        "SMA_20",
        "SMA_50",
        "RSI",
        "MACD",
        "BB_Upper",
        "BB_Lower",
        "Volatility",
        "Momentum_20",
    ]

    for feature in expected_features:
        assert feature in features.columns

    print("\nFeature Engineering Test Passed!")
    print(f"Rows: {len(features)}")
    print(f"Features: {len(features.columns)}")


def test_model_manager():
    """Verify all ML models can train and predict."""

    data = fetch_stock_data(
        "AAPL",
        period="1y"
    )

    features = FeatureEngineering(data).build_features()

    feature_columns = [
        "Daily_Return",
        "Price_Change",
        "High_Low_Range",
        "High_Low_Pct",
        "SMA_5",
        "SMA_10",
        "SMA_20",
        "SMA_50",
        "EMA_12",
        "EMA_26",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "BB_Middle",
        "BB_Upper",
        "BB_Lower",
        "BB_Position",
        "Volatility",
        "Momentum_5",
        "Momentum_10",
        "Momentum_20",
        "Volume_Change",
        "Relative_Volume",
        "Return_Lag_1",
        "Return_Lag_2",
        "Return_Lag_3",
        "Return_Lag_5",
        "Return_Lag_10",
    ]

    X = features[feature_columns]

    y = features["Close"]

    manager = ModelManager()

    manager.train_all(X, y)

    predictions = manager.predict_all(X.tail(5))

    assert len(predictions) == 3

    for model_name, prediction in predictions.items():

        assert len(prediction) == 5

        print(
            f"{model_name}: "
            f"{prediction[-1]:.2f}"
        )

    print("\nModel Manager Test Passed!")

def test_model_evaluation():
    """Verify chronological model evaluation."""

    data = fetch_stock_data(
        "AAPL",
        period="1y"
    )

    features = FeatureEngineering(data).build_features()

    feature_columns = [
        "Daily_Return",
        "Price_Change",
        "High_Low_Range",
        "High_Low_Pct",
        "SMA_5",
        "SMA_10",
        "SMA_20",
        "SMA_50",
        "EMA_12",
        "EMA_26",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "BB_Middle",
        "BB_Upper",
        "BB_Lower",
        "BB_Position",
        "Volatility",
        "Momentum_5",
        "Momentum_10",
        "Momentum_20",
        "Volume_Change",
        "Relative_Volume",
        "Return_Lag_1",
        "Return_Lag_2",
        "Return_Lag_3",
        "Return_Lag_5",
        "Return_Lag_10",
    ]

    X = features[feature_columns]
    y = features["Close"]

    manager = ModelManager()

    evaluator = ModelEvaluator(
        test_size=0.2
    )

    results = evaluator.evaluate_all(
        manager.models,
        X,
        y
    )

    print("\nModel Evaluation Results:")
    print(results.round(4))

    best_model = evaluator.select_best_model(
        results
    )

    print(
        f"\nBest Model: {best_model}"
    )

    assert not results.empty
    assert len(results) == 3
    assert best_model in manager.models

    print("\nModel Evaluation Test Passed!")

def test_prediction_service():
    """Verify the complete prediction service."""

    data = fetch_stock_data(
        "AAPL",
        period="1y"
    )

    service = PredictionService()

    result = service.run_prediction(
        data,
        days_ahead=5
    )

    assert "predictions" in result
    assert "best_model" in result
    assert "evaluation" in result

    assert len(
        result["predictions"]
    ) == 5

    assert result["best_model"] in [
        "Ridge",
        "Random Forest",
        "Gradient Boosting"
    ]

    print("\nPrediction Service Test Passed!")

    print(
        f"Best Model: "
        f"{result['best_model']}"
    )

    print(
        "Predictions:",
        [
            round(price, 2)
            for price in result["predictions"]
        ]
    )


if __name__ == "__main__":

    test_feature_engineering()
    test_model_manager()
    test_model_evaluation()
    test_prediction_service()