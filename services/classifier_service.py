import json
import os
import joblib


class ClassifierService:
    """
    Production service for the StockSense
    calibrated Extra Trees direction classifier.

    Production model:
        CalibratedClassifierCV
            └── ExtraTreesClassifier
                └── Sigmoid probability calibration

    The original raw Extra Trees model can still be used by
    passing its model and metadata paths to the constructor.
    """

    def __init__(
        self,
        model_path="models/stocksense_calibrated_classifier.pkl",
        metadata_path="models/calibrated_model_metadata.json",
    ):

        self.model_path = model_path
        self.metadata_path = metadata_path

        self.model = None
        self.metadata = None

        self.feature_columns = []

        self.load_model()
        self.load_metadata()

        self.validate_model_configuration()


    # =====================================================
    # MODEL LOADING
    # =====================================================

    def load_model(self):

        if not os.path.exists(
            self.model_path
        ):
            raise FileNotFoundError(
                f"Model file not found: "
                f"{self.model_path}"
            )

        # Calibrated production model was saved
        # using joblib.dump().
        self.model = joblib.load(
            self.model_path
        )


    # =====================================================
    # METADATA LOADING
    # =====================================================

    def load_metadata(self):

        if not os.path.exists(
            self.metadata_path
        ):
            raise FileNotFoundError(
                f"Metadata file not found: "
                f"{self.metadata_path}"
            )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as file:

            self.metadata = json.load(
                file
            )

        self.feature_columns = (
            self.metadata.get(
                "features",
                [],
            )
        )

        if not self.feature_columns:

            raise ValueError(
                "Model metadata does not contain "
                "any feature columns."
            )


    # =====================================================
    # MODEL CONFIGURATION VALIDATION
    # =====================================================

    def validate_model_configuration(self):

        expected_model_types = [
            "CalibratedClassifierCV",
            "ExtraTreesClassifier",
        ]

        actual_model_type = (
            self.metadata.get(
                "model_type"
            )
        )

        if actual_model_type not in expected_model_types:

            raise ValueError(
                "Incorrect production model type. "
                f"Expected one of "
                f"{expected_model_types}, "
                f"found {actual_model_type}."
            )

        # -------------------------------------------------
        # Validate actual loaded model type
        # -------------------------------------------------

        actual_loaded_type = (
            type(self.model).__name__
        )

        if (
            actual_loaded_type
            != actual_model_type
        ):

            raise ValueError(
                "Loaded model type does not match "
                "the metadata. "
                f"Metadata: {actual_model_type}, "
                f"Loaded: {actual_loaded_type}."
            )

        # -------------------------------------------------
        # Confirm calibrated model uses Extra Trees
        # -------------------------------------------------

        if (
            actual_model_type
            == "CalibratedClassifierCV"
        ):

            base_model_type = (
                self.metadata.get(
                    "base_model_type"
                )
            )

            if (
                base_model_type
                != "ExtraTreesClassifier"
            ):

                raise ValueError(
                    "Calibrated production model "
                    "does not use ExtraTreesClassifier "
                    "as its base model."
                )

        # -------------------------------------------------
        # Confirm exact production features
        # -------------------------------------------------

        expected_features = [
            "Return_Lag_1",
            "Return_Lag_3",
            "Return_Lag_5",
            "Volatility",
            "Volume_Change",
        ]

        if (
            self.feature_columns
            != expected_features
        ):

            raise ValueError(
                "Production feature configuration "
                "does not match the confirmed model."
            )

        # -------------------------------------------------
        # Probability support
        # -------------------------------------------------

        if not hasattr(
            self.model,
            "predict_proba",
        ):

            raise ValueError(
                "Production classifier does not "
                "support probability predictions."
            )


    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    def get_model_info(self):

        validation = self.metadata.get(
            "validation",
            {},
        )

        # Calibrated metadata stores the base model
        # configuration under this name.
        hyperparameters = (
            self.metadata.get(
                "hyperparameters"
            )
        )

        if hyperparameters is None:

            hyperparameters = (
                self.metadata.get(
                    "base_model_hyperparameters",
                    {},
                )
            )

        calibration = (
            self.metadata.get(
                "calibration",
                {},
            )
        )

        return {
            "model_name": self.metadata.get(
                "model_name"
            ),

            "model_type": self.metadata.get(
                "model_type"
            ),

            "base_model_type": self.metadata.get(
                "base_model_type"
            ),

            "feature_count": len(
                self.feature_columns
            ),

            "features": self.feature_columns,

            "hyperparameters": (
                hyperparameters
            ),

            "calibration": calibration,

            "training_rows": self.metadata.get(
                "training_rows"
            ),

            "evaluation_summary": validation,
        }


    # =====================================================
    # FEATURE VALIDATION
    # =====================================================

    def validate_features(
        self,
        features,
    ):

        missing_features = [
            feature
            for feature in self.feature_columns
            if feature not in features.columns
        ]

        if missing_features:

            raise ValueError(
                "Missing required model features: "
                + ", ".join(
                    missing_features
                )
            )

        return True


    # =====================================================
    # PREDICTION
    # =====================================================

    def predict(
        self,
        features,
    ):

        self.validate_features(
            features
        )

        # -------------------------------------------------
        # Select features in EXACT training order
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
        # Check for missing values
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
                "Latest market row contains "
                "missing feature values: "
                + ", ".join(
                    missing_columns
                )
            )

        # -------------------------------------------------
        # Model prediction
        # -------------------------------------------------

        prediction = self.model.predict(
            latest_row
        )[0]

        probabilities = (
            self.model.predict_proba(
                latest_row
            )[0]
        )
        # =========================================================
        # PRODUCTION FEATURE VALUES
        # =========================================================
        latest_features = X.iloc[-1]

        feature_values = {}

        for feature_name in self.feature_columns:

            value = latest_features[feature_name]

            if hasattr(value, "item"):
                value = value.item()

            feature_values[feature_name] = float(value)
        # -------------------------------------------------
        # Convert model output
        # -------------------------------------------------

        prediction = int(
            prediction
        )

        probability_down = float(
            probabilities[0]
        )

        probability_up = float(
            probabilities[1]
        )

        # -------------------------------------------------
        # Safety validation
        # -------------------------------------------------

        if not (
            0.0
            <= probability_down
            <= 1.0
        ):

            raise ValueError(
                "Invalid DOWN probability returned "
                "by production classifier."
            )

        if not (
            0.0
            <= probability_up
            <= 1.0
        ):

            raise ValueError(
                "Invalid UP probability returned "
                "by production classifier."
            )

        probability_sum = (
            probability_down
            + probability_up
        )

        if abs(
            probability_sum - 1.0
        ) > 1e-6:

            raise ValueError(
                "Classifier probabilities do not "
                "sum to 1.0."
            )

        # -------------------------------------------------
        # Direction and confidence
        # -------------------------------------------------

        if prediction == 1:

            direction = "UP"

            confidence = (
                probability_up
            )

        else:

            direction = "DOWN"

            confidence = (
                probability_down
            )
        # -------------------------------------------------
        # Confidence interpretation
        # -------------------------------------------------

        confidence_percent = (
            confidence * 100
        )

        if confidence_percent < 55:

            confidence_level = "LOW"

        elif confidence_percent < 60:

            confidence_level = "MODERATE"

        elif confidence_percent < 70:

            confidence_level = "STRONG"

        else:

            confidence_level = "VERY_STRONG"

        uncertainty = self.calculate_uncertainty(
            probability_up,
            probability_down
        )
        # -------------------------------------------------
        # Return structured result
        # -------------------------------------------------

        return {
            "prediction": prediction,

            "direction": direction,

            "probability_up": (
                probability_up
            ),

            "probability_down": (
                probability_down
            ),

            "confidence": (
                confidence
            ),

            "confidence_percent": (
                confidence * 100
            ),
            "feature_values": (
                feature_values
            ),
            "uncertainty": uncertainty,
            
            "confidence_level": (
                confidence_level
            )
        }


    # =====================================================
    # BATCH PREDICTION
    # =====================================================


    def predict_dataframe(
        self,
        features,
    ):

        self.validate_features(
            features
        )

        X = features[
            self.feature_columns
        ].copy()

        if X.isnull().any().any():

            raise ValueError(
                "Feature dataset contains "
                "missing values."
            )

        predictions = (
            self.model.predict(X)
        )

        probabilities = (
            self.model.predict_proba(X)
        )

        results = []

        for index in range(
            len(X)
        ):

            prediction = int(
                predictions[index]
            )

            probability_down = float(
                probabilities[index][0]
            )

            probability_up = float(
                probabilities[index][1]
            )

            # -------------------------------------------------
            # Probability validation
            # -------------------------------------------------

            if not (
                0.0
                <= probability_down
                <= 1.0
            ):

                raise ValueError(
                    "Invalid DOWN probability "
                    "returned by production classifier."
                )

            if not (
                0.0
                <= probability_up
                <= 1.0
            ):

                raise ValueError(
                    "Invalid UP probability "
                    "returned by production classifier."
                )

            probability_sum = (
                probability_down
                + probability_up
            )

            if abs(
                probability_sum - 1.0
            ) > 1e-6:

                raise ValueError(
                    "Classifier probabilities do not "
                    "sum to 1.0."
                )

            # -------------------------------------------------
            # Direction and confidence
            # -------------------------------------------------

            if prediction == 1:

                direction = "UP"

                confidence = (
                    probability_up
                )

            else:

                direction = "DOWN"

                confidence = (
                    probability_down
                )

            # -------------------------------------------------
            # Feature values for this row
            # -------------------------------------------------

            row_features = X.iloc[index]

            feature_values = {}

            for feature_name in self.feature_columns:

                value = row_features[
                    feature_name
                ]

                if hasattr(value, "item"):
                    value = value.item()

                feature_values[
                    feature_name
                ] = float(value)

            # -------------------------------------------------
            # Store result
            # -------------------------------------------------

            results.append(
                {
                    "prediction": prediction,

                    "direction": direction,

                    "probability_up": (
                        probability_up
                    ),

                    "probability_down": (
                        probability_down
                    ),

                    "confidence": (
                        confidence
                    ),

                    "confidence_percent": (
                        confidence * 100
                    ),

                    "feature_values": (
                        feature_values
                    ),
                }
            )

        return results
    
    def calculate_uncertainty(self, probability_up, probability_down):
        """
        Calculate model uncertainty based on the distance
        between the predicted probability and a neutral 50/50 split.

        This represents model confidence/uncertainty, not
        financial risk or probability of profit.
        """

        try:
            probability_up = float(probability_up)
            probability_down = float(probability_down)

            # Probability spread from the neutral 50/50 point
            probability_spread = abs(probability_up - probability_down)

            # Convert to percentage points
            probability_spread_percent = probability_spread * 100

            # Determine uncertainty level
            if probability_spread_percent < 10:
                uncertainty_level = "HIGH"
            elif probability_spread_percent < 20:
                uncertainty_level = "MODERATE"
            elif probability_spread_percent < 40:
                uncertainty_level = "LOW"
            else:
                uncertainty_level = "VERY_LOW"

            return {
                "probability_spread": probability_spread,
                "probability_spread_percent": round(
                    probability_spread_percent,
                    2
                ),
                "uncertainty_level": uncertainty_level,
            }

        except (TypeError, ValueError):
            return {
                "probability_spread": None,
                "probability_spread_percent": None,
                "uncertainty_level": "UNKNOWN",
            }