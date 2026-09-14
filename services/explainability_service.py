import numpy as np


class ExplainabilityService:
    """
    Lightweight model explainability service for the
    StockSense production direction classifier.

    Instead of SHAP TreeExplainer, this production-safe
    implementation uses the native feature_importances_
    values from the underlying ExtraTrees classifier.

    This avoids the heavy SHAP / Numba / llvmlite dependency
    chain required by the previous implementation.
    """

    def __init__(self, classifier_service):

        self.classifier_service = classifier_service

        self.model = (
            classifier_service.model
        )

        self.feature_columns = (
            classifier_service.feature_columns
        )

        self.feature_importances_ = (
            self._get_feature_importances()
        )


    # =====================================================
    # GET FEATURE IMPORTANCES
    # =====================================================

    def _get_feature_importances(self):

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

            if not hasattr(
                base_model,
                "feature_importances_"
            ):

                raise ValueError(
                    "Underlying calibrated model "
                    "does not provide feature_importances_."
                )

            return np.asarray(
                base_model.feature_importances_,
                dtype=float
            )


        # -------------------------------------------------
        # Raw ExtraTreesClassifier fallback
        # -------------------------------------------------

        if (
            type(model).__name__
            == "ExtraTreesClassifier"
        ):

            if not hasattr(
                model,
                "feature_importances_"
            ):

                raise ValueError(
                    "ExtraTreesClassifier does not "
                    "provide feature_importances_."
                )

            return np.asarray(
                model.feature_importances_,
                dtype=float
            )


        raise ValueError(
            "Unsupported model type for "
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
        # Calculate lightweight contributions
        # -------------------------------------------------

        feature_values = (
            latest_row.iloc[0]
            .to_dict()
        )

        contributions = []

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

            importance = float(
                self.feature_importances_[index]
            )

            # -------------------------------------------------
            # Normalize the current feature value so that
            # contribution reflects both importance and
            # whether the current value is above/below the
            # feature's recent mean.
            # -------------------------------------------------

            feature_series = X[
                feature_name
            ]

            feature_mean = float(
                feature_series.mean()
            )

            if feature_mean != 0:

                relative_position = (
                    float(value) - feature_mean
                ) / abs(feature_mean)

            else:

                relative_position = (
                    float(value)
                )

            contribution = (
                importance
                * relative_position
            )

            contributions.append(
                {
                    "name": feature_name,

                    "value": float(
                        value
                    ),

                    "contribution": float(
                        contribution
                    ),

                    "importance": float(
                        importance
                    ),
                }
            )

        # -------------------------------------------------
        # Determine impact
        # -------------------------------------------------

        explanations = []

        for item in contributions:

            contribution = (
                item["contribution"]
            )

            if contribution > 0:

                impact = "positive"

            elif contribution < 0:

                impact = "negative"

            else:

                impact = "neutral"

            explanations.append(
                {
                    "name": item["name"],

                    "value": item["value"],

                    "contribution": (
                        item["contribution"]
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

            "method": (
                "ExtraTrees Feature Importance"
            ),

            "model": (
                type(
                    self.model
                ).__name__
            ),
        }