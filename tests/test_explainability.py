from services.classifier_service import ClassifierService
from services.explainability_service import ExplainabilityService
from services.market_data import fetch_stock_data
from services.prediction_service import PredictionService


print("=" * 70)
print("STOCKSENSE SHAP EXPLAINABILITY TEST")
print("=" * 70)


classifier_service = (
    ClassifierService()
)

prediction_service = (
    PredictionService()
)

explainability_service = (
    ExplainabilityService(
        classifier_service
    )
)


print("\nFetching AAPL data...")

df = fetch_stock_data(
    "AAPL",
    period="2y"
)


print(
    f"Loaded {len(df)} rows."
)


print("\nPreparing features...")

features = (
    prediction_service
    .prepare_features(df)
)


print(
    "Feature columns:"
)

print(
    classifier_service.feature_columns
)


print("\nGenerating SHAP explanation...")

result = (
    explainability_service
    .explain(features)
)


print("\n" + "=" * 70)
print("EXPLANATION RESULT")
print("=" * 70)

print(
    f"Method: {result['method']}"
)

print(
    f"Model: {result['model']}"
)

print(
    f"Features: {result['feature_count']}"
)


print("\nFeature contributions:")

for feature in result["features"]:

    print(
        f"{feature['name']:20s} "
        f"value={feature['value']: .6f} "
        f"contribution={feature['contribution']: .6f} "
        f"impact={feature['impact']}"
    )


print("\n" + "=" * 70)
print("SHAP TEST COMPLETE")
print("=" * 70)