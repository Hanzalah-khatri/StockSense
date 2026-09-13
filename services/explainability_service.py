import numpy as np
import shap


class ExplainabilityService:
    """
    SHAP-based explainability service for the
    StockSense production direction classifier.

    Production model:
        CalibratedClassifierCV
            └── ExtraTreesClassifier
    """

    def __init__(self, classifier_service):

        self.classifier_service = classifier_service

        self.model = (
            classifier_service.model
        )

        self.feature_columns = (
            classifier_service.feature_columns
        )

        self.explainer = None

        self._create_explainer()


    # =====================================================
    # CREATE SHAP EXPLAINER
    # =====================================================

    def _create_explainer(self):

        model = self.model

        # -------------------------------------------------
        # CalibratedClassifierCV
        # -------------------------------------------------

        if hasattr(
            model,
            "calibrated_classifiers_"
        ):

            calibrated_models = (
                model.calibrated_classifiers_
            )

            if not calibrated_models:

                raise ValueError(
                    "Calibrated classifier does not "
                    "contain any fitted base models."
                )

            # Use the first fitted Extra Trees model
            base_model = (
                calibrated_models[0].estimator
            )

            self.explainer = (
                shap.TreeExplainer(
                    base_model
                )
            )

            return


        # -------------------------------------------------
        # Raw ExtraTreesClassifier fallback
        # -------------------------------------------------

        if (
            type(model).__name__
            == "ExtraTreesClassifier"
        ):

            self.explainer = (
                shap.TreeExplainer(
                    model
                )
            )

            return


        raise ValueError(
            "Unsupported model type for SHAP "
            f"explainability: "
            f"{type(model).__name__}"
        )


    # =====================================================
    # EXPLAIN CURRENT PREDICTION
    # =====================================================

    def explain(
        self,
        features,
    ):

        # -------------------------------------------------
        # Validate required features
        # -------------------------------------------------

        self.classifier_service.validate_features(
            features
        )

        # -------------------------------------------------
        # Select exact production features
        # -------------------------------------------------

        X = features[
            self.feature_columns
        ].copy()

        # -------------------------------------------------
        # Use latest available row
        # -------------------------------------------------

        latest_row = X.iloc[
            [-1]
        ]

        # -------------------------------------------------
        # Check missing values
        # -------------------------------------------------

        if latest_row.isnull().any().any():

            missing_columns = (
                latest_row.columns[
                    latest_row.isnull()
                    .any()
                ]
                .tolist()
            )

            raise ValueError(
                "Latest model input contains "
                "missing feature values: "
                + ", ".join(
                    missing_columns
                )
            )

        # -------------------------------------------------
        # Calculate SHAP values
        # -------------------------------------------------

        shap_values = (
            self.explainer.shap_values(
                latest_row
            )
        )

        # -------------------------------------------------
        # Handle SHAP output format
        # -------------------------------------------------

        if isinstance(
            shap_values,
            list
        ):

            # Binary classification:
            # index 1 = UP class
            shap_array = np.asarray(
                shap_values[1]
            )

        else:

            shap_array = np.asarray(
                shap_values
            )

            # New SHAP versions may return:
            # (samples, features, classes)

            if (
                shap_array.ndim == 3
            ):

                shap_array = (
                    shap_array[
                        :, :, 1
                    ]
                )

        # -------------------------------------------------
        # Flatten latest row
        # -------------------------------------------------

        shap_array = np.asarray(
            shap_array
        ).reshape(-1)

        feature_values = (
            latest_row.iloc[0]
            .to_dict()
        )

        # -------------------------------------------------
        # Build feature explanations
        # -------------------------------------------------

        explanations = []

        for index, feature_name in enumerate(
            self.feature_columns
        ):

            value = feature_values[
                feature_name
            ]

            if hasattr(
                value,
                "item"
            ):

                value = value.item()

            contribution = float(
                shap_array[index]
            )

            if contribution > 0:

                impact = "positive"

            elif contribution < 0:

                impact = "negative"

            else:

                impact = "neutral"

            explanations.append(
                {
                    "name": feature_name,

                    "value": float(
                        value
                    ),

                    "contribution": (
                        contribution
                    ),

                    "impact": impact,
                }
            )

        # -------------------------------------------------
        # Sort by absolute contribution
        # -------------------------------------------------

        explanations.sort(
            key=lambda item: abs(
                item["contribution"]
            ),
            reverse=True,
        )

        # -------------------------------------------------
        # Return explanation
        # -------------------------------------------------

        return {
            "features": explanations,

            "feature_count": len(
                explanations
            ),

            "method": "SHAP TreeExplainer",

            "model": (
                type(
                    self.model
                ).__name__
            ),
        }