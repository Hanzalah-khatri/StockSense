from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

import json

import pandas as pd
import plotly
import plotly.graph_objs as go

from services.validation_service import ValidationService

from services.market_data import (
    fetch_stock_data
)

from services.prediction_service import (
    PredictionService
)

from services.classifier_service import (
    ClassifierService
)

from services.analyst_service import (
    StockAnalyst
)
from services.explainability_service import ExplainabilityService


# =========================================================
# APP
# =========================================================

app = Flask(__name__)


prediction_service = (
    PredictionService()
)

classifier_service = (
    ClassifierService()
)

analyst = (
    StockAnalyst()
)

validation_service = (
    ValidationService()
)
explainability_service = ExplainabilityService(
    classifier_service
)


# =========================================================
# HOME
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# =========================================================
# PREDICTION API
# =========================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    payload = (
        request.get_json(
            force=True
        )
        or {}
    )

    ticker = (
        payload.get("ticker")
        or "AAPL"
    ).strip().upper()

    try:

        days_ahead = int(
            payload.get(
                "days_ahead"
            )
            or 7
        )

    except (
        TypeError,
        ValueError
    ):

        days_ahead = 7

    days_ahead = max(
        1,
        min(
            days_ahead,
            30
        )
    )

    if not ticker:

        return jsonify({
            "error":
                "Please provide a stock ticker symbol."
        }), 400

    try:

        # =================================================
        # MARKET DATA
        # =================================================

        df = fetch_stock_data(
            ticker,
            period="2y"
        )

        if "Date" not in df.columns:

            df = df.reset_index()

            if "Date" not in df.columns:

                df.rename(
                    columns={
                        df.columns[0]:
                            "Date"
                    },
                    inplace=True
                )

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = df.dropna(
            subset=[
                "Date",
                "Close"
            ]
        ).copy()

        if df.empty:

            return jsonify({
                "error":
                    f"No market data found for {ticker}."
            }), 400

        if len(df) < 200:

            return jsonify({
                "error":
                    "Not enough historical data was returned."
            }), 400

        # =================================================
        # FEATURES
        # =================================================

        features = (
            prediction_service
            .prepare_features(df)
        )

        # =================================================
        # EXPERIMENTAL PRICE FORECAST
        # =================================================

        forecast_result = (
            prediction_service
            .run_prediction(
                df,
                days_ahead=days_ahead
            )
        )

        predicted_prices = [

            round(
                float(price),
                2
            )

            for price
            in forecast_result[
                "predictions"
            ]

        ]

        # =================================================
        # PRIMARY ML DIRECTION MODEL
        # =================================================

        direction_result = (
            classifier_service.predict(
                features
            )
        )

        # =================================================
        # PRODUCTION MODEL INFORMATION
        # =================================================

        model_info = (
            classifier_service
            .get_model_info()
        )

        validation_summary = (
            model_info.get(
                "evaluation_summary",
                {}
            )
        )

        calibrated_validation = (
            validation_summary.get(
                "calibrated_extra_trees",
                {}
            )
        )

        # -------------------------------------------------
        # Historical validation metrics
        # -------------------------------------------------

        validation_accuracy = (
            calibrated_validation.get(
                "accuracy"
            )
        )

        validation_brier = (
            calibrated_validation.get(
                "brier_score"
            )
        )

        validation_log_loss = (
            calibrated_validation.get(
                "log_loss"
            )
        )

        validation_roc_auc = (
            calibrated_validation.get(
                "roc_auc"
            )
        )

        majority_baseline = 0.5364

        validation_improvement = None

        if validation_accuracy is not None:

            validation_improvement = (
                validation_accuracy
                - majority_baseline
            )

        # =================================================
        # AI STOCK ANALYST
        # =================================================

        analyst_result = (
            analyst.analyze(
                features,
                direction_result
            )
        )

        # =================================================
        # CURRENT MARKET DATA
        # =================================================

        last_row = features.iloc[-1]

        last_close = float(
            last_row["Close"]
        )

        previous_close = None

        if len(features) >= 2:

            previous_close = float(
                features.iloc[-2]["Close"]
            )

        daily_change = None
        daily_change_percent = None

        if (
            previous_close is not None
            and previous_close != 0
        ):

            daily_change = (
                last_close
                - previous_close
            )

            daily_change_percent = (
                daily_change
                / previous_close
                * 100
            )

        # =================================================
        # FUTURE DATES
        # =================================================

        last_date = pd.Timestamp(
            df["Date"].iloc[-1]
        )

        future_dates = []

        current_date = last_date

        while len(
            future_dates
        ) < days_ahead:

            current_date += (
                pd.Timedelta(
                    days=1
                )
            )

            if current_date.weekday() >= 5:

                continue

            future_dates.append(
                current_date.strftime(
                    "%Y-%m-%d"
                )
            )

        # =================================================
        # HISTORICAL DATA
        # =================================================

        historical_dates = (
            df["Date"]
            .dt.strftime(
                "%Y-%m-%d"
            )
            .tolist()
        )

        historical_close = [

            round(
                float(value),
                2
            )

            for value in df["Close"]

        ]

        # =================================================
        # FORECAST TABLE DATA
        # =================================================

        forecast_rows = []

        previous_price = last_close

        for index in range(
            len(predicted_prices)
        ):

            predicted_price = (
                predicted_prices[index]
            )

            change_percent = 0.0

            if previous_price != 0:

                change_percent = (
                    (
                        predicted_price
                        - previous_price
                    )
                    / previous_price
                    * 100
                )

            forecast_rows.append({

                "date":
                    future_dates[index],

                "predicted_price":
                    round(
                        predicted_price,
                        2
                    ),

                "change_percent":
                    round(
                        change_percent,
                        2
                    )

            })

            previous_price = (
                predicted_price
            )

        # =================================================
        # CLASSIFIER DIRECTION
        # =================================================

        direction = (
            direction_result[
                "direction"
            ]
        )

        probability_up = (
            direction_result[
                "probability_up"
            ]
        )

        probability_down = (
            direction_result[
                "probability_down"
            ]
        )

        classifier_confidence = (
            direction_result[
                "confidence_percent"
            ]
        )

        # =================================================
        # GRAPH
        # =================================================

        historical_trace = go.Scatter(

            x=historical_dates,

            y=historical_close,

            mode="lines",

            name="Historical Close",

            line=dict(
                color="#2563eb",
                width=2
            )
        )

        forecast_trace = go.Scatter(

            x=[
                historical_dates[-1]
            ] + future_dates,

            y=[
                historical_close[-1]
            ] + predicted_prices,

            mode="lines+markers",

            name="Experimental Forecast",

            line=dict(
                color="#10b981",
                width=3,
                dash="dash"
            ),

            marker=dict(
                size=6
            ),

            hovertemplate=(
                "%{x}<br>"
                "Forecast: $%{y:.2f}"
                "<extra></extra>"
            )
        )

        forecast_start = go.Scatter(

            x=[
                historical_dates[-1]
            ],

            y=[
                historical_close[-1]
            ],

            mode="markers",

            name="Forecast Start",

            marker=dict(
                size=9,
                color="#f59e0b",
                line=dict(
                    width=2,
                    color="#ffffff"
                )
            ),

            hovertemplate=(
                "%{x}<br>"
                "Last Close: $%{y:.2f}"
                "<extra></extra>"
            )
        )

        fig = go.Figure(

            data=[

                historical_trace,

                forecast_trace,

                forecast_start

            ]

        )

        fig.update_layout(

            template="plotly_white",

            margin=dict(
                l=40,
                r=20,
                t=20,
                b=40
            ),

            xaxis_title="Date",

            yaxis_title="Price (USD)",

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),

            hovermode="x unified"
        )

        graph_json = json.loads(

            json.dumps(
                fig,
                cls=plotly.utils.PlotlyJSONEncoder
            )
        )

        # =================================================
        # TECHNICAL INDICATORS
        # =================================================

        technical = (
            analyst_result[
                "technical_indicators"
            ]
        )

        # =================================================
        # REGRESSION EVALUATION
        # =================================================

        evaluation_df = (
            forecast_result[
                "evaluation"
            ]
        )

        evaluation = (
            evaluation_df
            .round(6)
            .to_dict(
                orient="index"
            )
        )

        best_model = (
            forecast_result[
                "best_model"
            ]
        )

        # =================================================
        # FINAL API RESPONSE
        # =================================================

        response = {

            "ticker":
                df.attrs.get(
                    "ticker",
                    ticker
                ),

            "data_source":
                df.attrs.get(
                    "source",
                    "Yahoo Finance"
                ),

            "history_period":
                "2 years",

            "history_rows":
                len(df),

            # ---------------------------------------------
            # MARKET
            # ---------------------------------------------

            "market": {

                "price":
                    round(
                        last_close,
                        2
                    ),

                "previous_close":
                    (
                        round(
                            previous_close,
                            2
                        )
                        if previous_close
                        is not None
                        else None
                    ),

                "daily_change":
                    (
                        round(
                            daily_change,
                            2
                        )
                        if daily_change
                        is not None
                        else None
                    ),

                "daily_change_percent":
                    (
                        round(
                            daily_change_percent,
                            2
                        )
                        if daily_change
                        is not None
                        else None
                    )
            },

            # ---------------------------------------------
            # PRIMARY DIRECTION MODEL
            # ---------------------------------------------

            "direction": {

                "prediction":
                    direction_result[
                        "prediction"
                    ],

                "direction":
                    direction,

                "probability_up":
                    round(
                        probability_up,
                        4
                    ),

                "probability_down":
                    round(
                        probability_down,
                        4
                    ),

                "confidence":
                    round(
                        direction_result[
                            "confidence"
                        ],
                        4
                    ),

                "confidence_percent":
                    round(
                        classifier_confidence,
                        2
                    ),
                "confidence_level":
                    direction_result.get(
                        "confidence_level",
                        "LOW"
                    ),
                "uncertainty": direction_result.get("uncertainty"),
                    
                "feature_values": direction_result.get(
                    "feature_values",
                    {}
                ),

                # -----------------------------------------
                # MODEL INFORMATION
                # -----------------------------------------

                "model": {

                    "name":
                        model_info.get(
                            "model_name"
                        ),

                    "type":
                        model_info.get(
                            "model_type"
                        ),

                    "base_model":
                        model_info.get(
                            "base_model_type"
                        ),

                    "feature_count":
                        model_info.get(
                            "feature_count"
                        ),

                    "features":
                        model_info.get(
                            "features",
                            []
                        ),

                    "training_rows":
                        model_info.get(
                            "training_rows"
                        ),

                    "calibration":
                        model_info.get(
                            "calibration",
                            {}
                        ),

                    "validation": {

                        "accuracy":
                            validation_accuracy,

                        "accuracy_percent":
                            (
                                validation_accuracy
                                * 100
                                if validation_accuracy
                                is not None
                                else None
                            ),

                        "brier_score":
                            validation_brier,

                        "log_loss":
                            validation_log_loss,

                        "roc_auc":
                            validation_roc_auc,

                        "majority_baseline":
                            majority_baseline,

                        "baseline_percent":
                            (
                                majority_baseline
                                * 100
                            ),

                        "improvement_over_baseline":
                            validation_improvement,

                        "improvement_over_baseline_percent":
                            (
                                validation_improvement
                                * 100
                                if validation_improvement
                                is not None
                                else None
                            ),

                        "method":
                            validation_summary.get(
                                "method"
                            ),

                        "confirmation_splits":
                            validation_summary.get(
                                "confirmation_splits"
                            ),

                        "confirmation_min_train_size":
                            validation_summary.get(
                                "confirmation_min_train_size"
                            )
                    },

                    "status":
                        "production_candidate",

                    "probability_note":
                        (
                            "Probability is generated by "
                            "the sigmoid-calibrated classifier. "
                            "It represents model-estimated "
                            "probability and is not a guarantee "
                            "of future market movement."
                        )
                }
            },

            # ---------------------------------------------
            # AI ANALYST
            # ---------------------------------------------

            "analyst":
                analyst_result,

            # ---------------------------------------------
            # TECHNICAL INDICATORS
            # ---------------------------------------------

            "technical":
                technical,

            # ---------------------------------------------
            # HISTORICAL DATA
            # ---------------------------------------------

            "history": {

                "dates":
                    historical_dates,

                "close":
                    historical_close
            },

            # ---------------------------------------------
            # FORECAST
            # ---------------------------------------------

            "forecast": {

                "status":
                    "experimental",

                "model":
                    best_model,

                "dates":
                    future_dates,

                "prices":
                    predicted_prices,

                "rows":
                    forecast_rows,

                "evaluation":
                    evaluation,

                "message":
                    (
                        "Experimental return-based "
                        "forecast. Future prices are "
                        "model estimates and should not "
                        "be treated as guaranteed values."
                    )
            },

            # ---------------------------------------------
            # FUTURE DATES
            # ---------------------------------------------

            "future_dates":
                future_dates,

            # ---------------------------------------------
            # REGRESSION STATUS
            # ---------------------------------------------

            "regression": {

                "status":
                    "experimental",

                "model":
                    best_model,

                "message":
                    (
                        "Experimental regression forecast "
                        "is shown separately from the "
                        "primary direction classifier."
                    )
            },

            # ---------------------------------------------
            # GRAPH
            # ---------------------------------------------

            "graph":
                graph_json
        }

        # =================================================
        # DEBUG
        # =================================================

        print("\n" + "=" * 70)
        print("STOCKSENSE PRODUCTION ANALYSIS")
        print("=" * 70)

        print(
            "Ticker:",
            ticker
        )

        print(
            "Last Close:",
            f"${last_close:.2f}"
        )

        print(
            "Direction:",
            direction
        )

        print(
            "Probability UP:",
            f"{probability_up:.2%}"
        )

        print(
            "Probability DOWN:",
            f"{probability_down:.2%}"
        )

        print(
            "Classifier Confidence:",
            f"{classifier_confidence:.2f}%"
        )

        print(
            "Classifier Model:",
            model_info.get(
                "model_name"
            )
        )

        print(
            "Calibration:",
            model_info.get(
                "calibration",
                {}
            ).get(
                "method",
                "N/A"
            )
        )

        if validation_accuracy is not None:

            print(
                "Validation Accuracy:",
                f"{validation_accuracy:.2%}"
            )

        if validation_improvement is not None:

            print(
                "Improvement vs Baseline:",
                f"{validation_improvement:+.2%}"
            )

        if validation_roc_auc is not None:

            print(
                "Validation ROC-AUC:",
                f"{validation_roc_auc:.4f}"
            )

        print(
            "AI Signal:",
            analyst_result[
                "signal"
            ]
        )

        print(
            "Signal Strength:",
            analyst_result[
                "signal_strength"
            ]
        )

        print(
            "Forecast Model:",
            best_model
        )

        print(
            "Forecast Prices:",
            predicted_prices
        )

        print("=" * 70)

        return jsonify(
            response
        )

    except Exception as exc:

        print("\n" + "=" * 70)
        print("PRODUCTION API ERROR")
        print("=" * 70)

        print(
            type(exc).__name__
        )

        print(
            str(exc)
        )

        print("=" * 70 + "\n")

        return jsonify({
            "error":
                (
                    "Something went wrong: "
                    f"{str(exc)}"
                )
        }), 500

@app.route("/api/explain", methods=["POST"])
def explain_prediction():
    try:
        data = request.get_json(silent=True) or {}

        ticker = data.get(
            "ticker",
            "AAPL"
        ).strip().upper()

        if not ticker:
            return jsonify({
                "success": False,
                "error": "Ticker is required."
            }), 400

        print("\n" + "=" * 70)
        print("STOCKSENSE SHAP EXPLANATION")
        print("=" * 70)
        print("Ticker:", ticker)

        # ---------------------------------------------------------
        # Fetch market data
        # ---------------------------------------------------------

        df = fetch_stock_data(
            ticker,
            period="2y"
        )

        if df is None or df.empty:

            return jsonify({
                "success": False,
                "error": (
                    f"No market data available for {ticker}."
                )
            }), 404

        print(
            "Market rows:",
            len(df)
        )

        # ---------------------------------------------------------
        # Prepare production features
        # ---------------------------------------------------------

        features = prediction_service.prepare_features(
            df
        )

        print(
            "Feature columns:",
            list(features.columns)
        )

        print(
            "Production features:",
            classifier_service.feature_columns
        )

        # ---------------------------------------------------------
        # Generate SHAP explanation
        # ---------------------------------------------------------

        explanation = explainability_service.explain(
            features
        )

        print(
            "SHAP explanation generated successfully."
        )

        print(
            "Explanation method:",
            explanation.get("method")
        )

        print(
            "Feature count:",
            explanation.get("feature_count")
        )

        # ---------------------------------------------------------
        # Get classifier prediction
        # ---------------------------------------------------------

        direction_result = classifier_service.predict(
            features
        )

        print(
            "Prediction:",
            direction_result["direction"]
        )

        print(
            "Confidence:",
            direction_result["confidence_percent"]
        )

        print("=" * 70)

        return jsonify({
            "success": True,

            "ticker": ticker,

            "prediction":
                direction_result["direction"],

            "probability_up":
                direction_result["probability_up"],

            "probability_down":
                direction_result["probability_down"],

            "confidence":
                direction_result["confidence"],

            "confidence_percent":
                direction_result["confidence_percent"],

            "confidence_level":
                direction_result.get(
                    "confidence_level",
                    "LOW"
                ),

            "explanation":
                explanation,

            "note": (
                "SHAP contributions describe the "
                "contribution of each feature to the "
                "underlying Extra Trees classifier. "
                "They do not establish causation in "
                "the market."
            )
        })

    except Exception as e:

        print("\n" + "!" * 70)
        print("SHAP EXPLANATION ERROR")
        print("!" * 70)

        print(
            "Error type:",
            type(e).__name__
        )

        print(
            "Error:",
            str(e)
        )

        print(
            "!" * 70
        )

        return jsonify({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }), 500
    
# =========================================================
# MODEL INFORMATION API
# =========================================================

@app.route("/api/model-info", methods=["GET"])
def model_info():
    try:
        model_info = classifier_service.get_model_info()
        validation_summary = model_info.get("evaluation_summary", {})

        calibrated_validation = validation_summary.get(
            "calibrated_extra_trees",
            {}
        )

        majority_baseline = 0.5364

        validation_accuracy = calibrated_validation.get("accuracy")
        validation_brier = calibrated_validation.get("brier_score")
        validation_log_loss = calibrated_validation.get("log_loss")
        validation_roc_auc = calibrated_validation.get("roc_auc")

        improvement = (
            validation_accuracy - majority_baseline
            if validation_accuracy is not None
            else None
        )

        return jsonify({
            "success": True,

            "model": {
                "name": model_info.get("model_name"),
                "type": model_info.get("model_type"),
                "base_model": model_info.get("base_model_type"),
                "status": "production_candidate",

                "ticker": "AAPL",

                "features": model_info.get("features", []),
                "feature_count": model_info.get("feature_count"),

                "training_rows": model_info.get("training_rows"),

                "calibration": model_info.get(
                    "calibration",
                    {}
                )
            },

            "validation": {
                "accuracy": validation_accuracy,

                "accuracy_percent": (
                    round(validation_accuracy * 100, 2)
                    if validation_accuracy is not None
                    else None
                ),

                "brier_score": validation_brier,

                "log_loss": validation_log_loss,

                "roc_auc": validation_roc_auc,

                "majority_baseline": majority_baseline,

                "baseline_percent": round(
                    majority_baseline * 100,
                    2
                ),

                "improvement_over_baseline": improvement,

                "improvement_over_baseline_percent": (
                    round(improvement * 100, 2)
                    if improvement is not None
                    else None
                ),

                "method": "Expanding walk-forward validation",

                "confirmation_splits": 7,

                "confirmation_min_train_size": 150
            },

            "probability_note": (
                "Probabilities are estimates produced by the calibrated "
                "classifier and are not guarantees of future market direction."
            )
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
# =========================================================
# MODEL VALIDATION API
# =========================================================

@app.route(
    "/api/validate",
    methods=["POST"]
)
def validate_model():

    try:

        payload = (
            request.get_json(
                silent=True
            )
            or {}
        )

        ticker = str(
            payload.get(
                "ticker",
                "AAPL"
            )
        ).strip().upper()

        if not ticker:

            ticker = "AAPL"

        # Keep validation intentionally limited.
        if not ticker.replace(
            ".",
            ""
        ).replace(
            "-",
            ""
        ).isalnum():

            return jsonify({
                "success": False,
                "error":
                    "Invalid ticker symbol."
            }), 400

        print()
        print("=" * 70)
        print("STOCKSENSE — MODEL VALIDATION")
        print("=" * 70)

        print(
            f"Ticker: {ticker}"
        )

        market_data = fetch_stock_data(
            ticker=ticker,
            period="2y"
        )

        if market_data is None:

            raise ValueError(
                "No market data returned."
            )

        if market_data.empty:

            raise ValueError(
                "Market data is empty."
            )

        features = (
            prediction_service
            .prepare_features(
                market_data
            )
        )

        result = (
            validation_service
            .validate(
                features,
                n_splits=5
            )
        )

        print()
        print("VALIDATION RESULTS")
        print("-" * 70)

        metrics = result[
            "metrics"
        ]

        print(
            f"Accuracy:  "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall:    "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1 Score:  "
            f"{metrics['f1']:.4f}"
        )

        if metrics[
            "roc_auc"
        ] is not None:

            print(
                f"ROC-AUC:   "
                f"{metrics['roc_auc']:.4f}"
            )

        print()

        print(
            "Reliability:",
            result[
                "reliability"
            ][
                "level"
            ]
        )

        print(
            "=" * 70
        )

        print()

        return jsonify({

            "success":
                True,

            "ticker":
                ticker,

            "validation":
                result

        })

    except Exception as exc:

        print()
        print(
            "VALIDATION ERROR:",
            str(exc)
        )

        return jsonify({

            "success":
                False,

            "error":
                str(exc)

        }), 500


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
