document.addEventListener("DOMContentLoaded", () => {

    "use strict";

    // =========================================================
    // ELEMENTS
    // =========================================================

    const tickerInput = document.getElementById("ticker");
    const daysInput = document.getElementById("days_ahead");

    const analyzeBtn = document.getElementById("analyzeBtn");
    const buttonText = document.getElementById("buttonText");
    const buttonLoader = document.getElementById("buttonLoader");
    const buttonArrow = document.getElementById("buttonArrow");

    const validateModelBtn = document.getElementById("validateModelBtn");
    const validateButtonText = document.getElementById("validateButtonText");
    const validateButtonLoader = document.getElementById("validateButtonLoader");

    const statusMsg = document.getElementById("statusMsg");
    const resultsSection = document.getElementById("resultsSection");


    // =========================================================
    // SUMMARY ELEMENTS
    // =========================================================

    const sumTicker = document.getElementById("sumTicker");
    const sumLastClose = document.getElementById("sumLastClose");
    const sumDailyChange = document.getElementById("sumDailyChange");
    const sumTrend = document.getElementById("sumTrend");
    const sumProbability = document.getElementById("sumProbability");
    const sumConfidence = document.getElementById("sumConfidence");
    const sumSource = document.getElementById("sumSource");


    // =========================================================
    // CHART
    // =========================================================

    const chartDiv = document.getElementById("chartDiv");


    // =========================================================
    // DIRECTION CLASSIFIER
    // =========================================================

    const directionIcon = document.getElementById("directionIcon");
    const directionValue = document.getElementById("directionValue");

    const probabilityUp = document.getElementById("probabilityUp");
    const probabilityDown = document.getElementById("probabilityDown");

    const probabilityUpBar = document.getElementById("probabilityUpBar");
    const probabilityDownBar = document.getElementById("probabilityDownBar");

    const directionConfidence = document.getElementById("directionConfidence");
    const confidenceBar = document.getElementById("confidenceBar");


    // =========================================================
    // FORECAST
    // =========================================================

    const forecastModel = document.getElementById("forecastModel");
    const forecastHorizon = document.getElementById("forecastHorizon");

    const forecastFirstPrice = document.getElementById("forecastFirstPrice");
    const forecastFinalPrice = document.getElementById("forecastFinalPrice");

    const forecastTableBody = document.getElementById("forecastTableBody");
    const forecastEvaluation = document.getElementById("forecastEvaluation");

    const regressionMessage = document.getElementById("regressionMessage");


    // =========================================================
    // AI ANALYST
    // =========================================================

    const analystBadge = document.getElementById("analystBadge");
    const analystSignal = document.getElementById("analystSignal");
    const analystSummary = document.getElementById("analystSummary");

    const signalStrength = document.getElementById("signalStrength");
    const strengthFill = document.getElementById("strengthFill");

    const techRSI = document.getElementById("techRSI");
    const techMACD = document.getElementById("techMACD");
    const techSMA20 = document.getElementById("techSMA20");
    const techSMA50 = document.getElementById("techSMA50");
    const techSMA200 = document.getElementById("techSMA200");
    const techDailyMove = document.getElementById("techDailyMove");

    const reasonsList = document.getElementById("reasonsList");
    const risksList = document.getElementById("risksList");


    // =========================================================
    // METADATA
    // =========================================================

    const historyInfo = document.getElementById("historyInfo");
    const rowsInfo = document.getElementById("rowsInfo");
    const horizonInfo = document.getElementById("horizonInfo");


    // =========================================================
    // RELIABILITY
    // =========================================================

    const reliabilitySection = document.getElementById("reliabilitySection");

    const reliabilityStatus = document.getElementById("reliabilityStatus");
    const reliabilityStatusDot = document.getElementById("reliabilityStatusDot");
    const reliabilityStatusText = document.getElementById("reliabilityStatusText");

    const reliabilityLevel = document.getElementById("reliabilityLevel");
    const reliabilityDescription = document.getElementById("reliabilityDescription");

    const validationObservations = document.getElementById("validationObservations");
    const validationSplits = document.getElementById("validationSplits");
    const validationFeatures = document.getElementById("validationFeatures");


    // =========================================================
    // STAGE 2B — INTEGRITY
    // =========================================================

    const validationIntegrityIndicator =
        document.getElementById("validationIntegrityIndicator");

    const validationIntegrityValue =
        document.getElementById("validationIntegrityValue");

    const validationIntegrityDescription =
        document.getElementById("validationIntegrityDescription");


    // =========================================================
    // STAGE 2B — CLASS DISTRIBUTION
    // =========================================================

    const validationUpPercent =
        document.getElementById("validationUpPercent");

    const validationDownPercent =
        document.getElementById("validationDownPercent");

    const validationUpBar =
        document.getElementById("validationUpBar");

    const validationDownBar =
        document.getElementById("validationDownBar");

    const validationClassDescription =
        document.getElementById("validationClassDescription");


    // =========================================================
    // STAGE 2B — FOLD STABILITY
    // =========================================================

    const foldAverageAccuracy =
        document.getElementById("foldAverageAccuracy");

    const foldAccuracyRange =
        document.getElementById("foldAccuracyRange");

    const strongestFold =
        document.getElementById("strongestFold");

    const weakestFold =
        document.getElementById("weakestFold");


    // =========================================================
    // STAGE 2B — MODEL ASSESSMENT
    // =========================================================

    const modelAssessmentIndicator =
        document.getElementById("modelAssessmentIndicator");

    const modelAssessment =
        document.getElementById("modelAssessment");

    const modelAssessmentDescription =
        document.getElementById("modelAssessmentDescription");

    const modelAssessmentDelta =
        document.getElementById("modelAssessmentDelta");


    // =========================================================
    // VALIDATION METRICS
    // =========================================================

    const validationAccuracy =
        document.getElementById("validationAccuracy");

    const validationPrecision =
        document.getElementById("validationPrecision");

    const validationRecall =
        document.getElementById("validationRecall");

    const validationF1 =
        document.getElementById("validationF1");

    const validationAuc =
        document.getElementById("validationAuc");


    // =========================================================
    // BASELINE
    // =========================================================

    const baselineResult =
        document.getElementById("baselineResult");

    const modelBaselineAccuracy =
        document.getElementById("modelBaselineAccuracy");

    const baselineStrategy =
        document.getElementById("baselineStrategy");

    const majorityBaselineAccuracy =
        document.getElementById("majorityBaselineAccuracy");

    const baselineImprovement =
        document.getElementById("baselineImprovement");


    // =========================================================
    // FOLD TABLE
    // =========================================================

    const foldTableBody =
        document.getElementById("foldTableBody");

    // =========================================================
    // VALIDATION NOTE
    // =========================================================

    const validationNote =
        document.getElementById("validationNote");


    // =========================================================
    // HELPERS
    // =========================================================

    function getNumber(value, fallback = 0) {

        const number = Number(value);

        return Number.isFinite(number)
            ? number
            : fallback;
    }


    function formatPrice(value) {

        const number = Number(value);

        if (!Number.isFinite(number)) {
            return "—";
        }

        return `$${number.toFixed(2)}`;
    }


    /*
     * Values already expressed as percentages.
     *
     * Example:
     * 53.49 -> 53.49%
     */
    function formatPercent(value, digits = 2) {

        const number = Number(value);

        if (!Number.isFinite(number)) {
            return "—";
        }

        return `${number.toFixed(digits)}%`;
    }


    /*
     * Values expressed as decimals.
     *
     * Example:
     * 0.5349 -> 53.49%
     */
    function formatDecimalPercent(value, digits = 2) {

        const number = Number(value);

        if (!Number.isFinite(number)) {
            return "—";
        }

        return `${(number * 100).toFixed(digits)}%`;
    }


    function formatSignedPercent(value, digits = 2) {

        const number = Number(value);

        if (!Number.isFinite(number)) {
            return "—";
        }

        const sign = number > 0
            ? "+"
            : "";

        return `${sign}${number.toFixed(digits)}%`;
    }


    function formatPercentagePoints(value, digits = 2) {

        const number = Number(value);

        if (!Number.isFinite(number)) {
            return "—";
        }

        const sign = number > 0
            ? "+"
            : "";

        return `${sign}${number.toFixed(digits)} pp`;
    }


    function formatDate(value) {

        if (!value) {
            return "—";
        }

        const date = new Date(value);

        if (Number.isNaN(date.getTime())) {
            return String(value);
        }

        return date.toLocaleDateString(
            undefined,
            {
                year: "numeric",
                month: "short",
                day: "numeric"
            }
        );
    }


    function escapeHtml(value) {

        if (value === null || value === undefined) {
            return "";
        }

        return String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }


    function setElementText(element, value) {

        if (!element) {
            return;
        }

        element.textContent =
            value === null ||
            value === undefined ||
            value === ""
                ? "—"
                : value;
    }


    function setStatus(message, type = "info") {

        if (!statusMsg) {
            return;
        }

        statusMsg.textContent = message || "";

        statusMsg.classList.remove(
            "success",
            "error",
            "info",
            "warning"
        );

        statusMsg.classList.add(type);
    }


    function normalizeDirection(value) {

        if (!value) {
            return "UNKNOWN";
        }

        const direction =
            String(value)
                .trim()
                .toUpperCase();

        if (
            direction === "UP" ||
            direction === "BULLISH" ||
            direction === "BUY"
        ) {
            return "UP";
        }

        if (
            direction === "DOWN" ||
            direction === "BEARISH" ||
            direction === "SELL"
        ) {
            return "DOWN";
        }

        return direction;
    }


    function setBarWidth(element, percentage) {

        if (!element) {
            return;
        }

        let value = Number(percentage);

        if (!Number.isFinite(value)) {
            value = 0;
        }

        value = Math.max(
            0,
            Math.min(
                100,
                value
            )
        );

        element.style.width = `${value}%`;
    }


    /*
     * Converts classifier probability into percentage.
     *
     * Supports BOTH:
     *
     * 0.5349 -> 53.49
     *
     * 53.49 -> 53.49
     */
    function probabilityToPercent(value) {

        const number = Number(value);

        if (!Number.isFinite(number)) {
            return null;
        }

        if (
            number >= 0 &&
            number <= 1
        ) {
            return number * 100;
        }

        return number;
    }


    function metricToPercent(value) {

        const number = Number(value);

        if (!Number.isFinite(number)) {
            return null;
        }

        if (
            number >= 0 &&
            number <= 1
        ) {
            return number * 100;
        }

        return number;
    }


    function clearList(element) {

        if (!element) {
            return;
        }

        element.innerHTML = "";
    }


    function renderList(element, items, emptyText = "No major points reported.") {

        if (!element) {
            return;
        }

        clearList(element);

        if (
            !Array.isArray(items) ||
            items.length === 0
        ) {

            const li =
                document.createElement("li");

            li.textContent = emptyText;

            element.appendChild(li);

            return;
        }

        items.forEach(item => {

            const li =
                document.createElement("li");

            if (
                typeof item === "object" &&
                item !== null
            ) {

                li.textContent =
                    item.text ||
                    item.reason ||
                    item.message ||
                    JSON.stringify(item);

            } else {

                li.textContent =
                    String(item);
            }

            element.appendChild(li);
        });
    }


    // =========================================================
    // INPUT VALIDATION
    // =========================================================

    function getInputs() {

        const ticker =
            (
                tickerInput?.value ||
                "AAPL"
            )
                .trim()
                .toUpperCase();

        let days =
            parseInt(
                daysInput?.value ||
                "7",
                10
            );

        if (!Number.isFinite(days)) {
            days = 7;
        }

        days = Math.max(
            1,
            Math.min(
                30,
                days
            )
        );

        if (daysInput) {
            daysInput.value = days;
        }

        return {
            ticker,
            days
        };
    }


    function validateTicker(ticker) {

        return /^[A-Z0-9.-]{1,10}$/.test(
            ticker
        );
    }


    // =========================================================
    // BUTTON STATES
    // =========================================================

    function setAnalyzeLoading(isLoading) {

        if (!analyzeBtn) {
            return;
        }

        analyzeBtn.disabled = isLoading;

        if (buttonText) {
            buttonText.textContent =
                isLoading
                    ? "Analyzing..."
                    : "Analyze & Predict";
        }

        if (buttonLoader) {
            buttonLoader.style.display =
                isLoading
                    ? "inline-block"
                    : "none";
        }

        if (buttonArrow) {
            buttonArrow.style.display =
                isLoading
                    ? "none"
                    : "inline-block";
        }
    }


    function setValidationLoading(isLoading) {

        if (!validateModelBtn) {
            return;
        }

        validateModelBtn.disabled =
            isLoading;

        if (validateButtonText) {
            validateButtonText.textContent =
                isLoading
                    ? "Validating..."
                    : "Validate Model";
        }

        if (validateButtonLoader) {
            validateButtonLoader.style.display =
                isLoading
                    ? "inline-block"
                    : "none";
        }
    }


    // =========================================================
    // MAIN ANALYSIS
    // =========================================================

    async function analyzeStock() {

        const {
            ticker,
            days
        } = getInputs();

        if (!validateTicker(ticker)) {

            setStatus(
                "Please enter a valid ticker symbol.",
                "error"
            );

            return;
        }

        setAnalyzeLoading(true);

        setStatus(
            `Analyzing ${ticker} market data...`,
            "info"
        );

        try {

            const response =
                await fetch(
                    "/api/predict",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            ticker,
                            days_ahead: days
                        })
                    }
                );

            const data =
                await response.json();

            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "Analysis request failed."
                );
            }

            if (data.error) {

                throw new Error(
                    data.error
                );
            }

            updateDashboard(data);

            setStatus(
                `${ticker} analysis completed successfully.`,
                "success"
            );

        } catch (error) {

            console.error(
                "Prediction error:",
                error
            );

            setStatus(
                error.message ||
                "Unable to analyze this stock.",
                "error"
            );

        } finally {

            setAnalyzeLoading(false);
        }
    }


    // =========================================================
    // MODEL VALIDATION
    // =========================================================

    async function validateModel() {

        const {
            ticker
        } = getInputs();

        if (!validateTicker(ticker)) {

            setStatus(
                "Please enter a valid ticker symbol.",
                "error"
            );

            return;
        }

        setValidationLoading(true);

        setStatus(
            `Running walk-forward validation for ${ticker}...`,
            "info"
        );

        try {

            const response =
                await fetch(
                    "/api/validate",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            ticker
                        })
                    }
                );

            const data =
                await response.json();

            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "Validation request failed."
                );
            }

            if (
                data.success === false ||
                data.error
            ) {

                throw new Error(
                    data.error ||
                    "Validation failed."
                );
            }

            const validation =
                data.validation ||
                data.result ||
                data;

            if (!validation) {

                throw new Error(
                    "Validation response was empty."
                );
            }

            updateReliabilityDashboard(
                validation
            );

            setStatus(
                `${ticker} model validation completed.`,
                "success"
            );

        } catch (error) {

            console.error(
                "Validation error:",
                error
            );

            setStatus(
                error.message ||
                "Unable to validate the model.",
                "error"
            );

        } finally {

            setValidationLoading(false);
        }
    }


    // =========================================================
    // COMPLETE DASHBOARD UPDATE
    // =========================================================

    function updateDashboard(data) {

        updateSummary(data);

        updateDirection(data);

        updateAnalyst(data);

        updateForecastOverview(data);

        updateForecastTable(data);

        updateForecastEvaluation(data);

        updateForecastMessage(data);

        updateMetadata(data);

        renderChart(data);
        updatePredictionTransparency(
            data.direction
        );
        updatePredictionIntegrity(data);
        updatePredictionUncertainty(data.direction);
        const ticker =
            data.market?.ticker ||
            document.getElementById("ticker")?.value ||
            "AAPL";

        loadExplainability(ticker);

        /*
         * /api/predict does not currently perform
         * Stage 2B validation, but this keeps the
         * frontend compatible if validation is later
         * attached to the prediction response.
         */
        if (data.validation) {

            updateReliabilityDashboard(
                data.validation
            );
        }

        if (resultsSection) {

            resultsSection.style.display =
                "block";

            resultsSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });
        }
    }


    // =========================================================
    // SUMMARY
    // =========================================================

    function updateSummary(data) {

        const market =
            data.market ||
            {};

        const direction =
            data.direction ||
            {};

        const ticker =
            data.ticker ||
            getInputs().ticker;

        const lastClose =
            market.price ??
            market.last_close ??
            market.current_price ??
            market.close;

        const dailyChange =
            market.daily_change_percent ??
            market.daily_change;

        const directionValueRaw =
            direction.direction ??
            direction.prediction;

        const normalizedDirection =
            normalizeDirection(
                directionValueRaw
            );

        const confidence =
            direction.confidence_percent ??
            probabilityToPercent(
                direction.confidence
            );

        const probabilityUp =
            probabilityToPercent(
                direction.probability_up
            );

        const source =
            data.data_source ||
            market.data_source ||
            "Yahoo Finance";

        setElementText(
            sumTicker,
            ticker
        );

        setElementText(
            sumLastClose,
            formatPrice(lastClose)
        );

        setElementText(
            sumDailyChange,
            formatSignedPercent(
                dailyChange
            )
        );

        setElementText(
            sumTrend,
            normalizedDirection
        );

        setElementText(
            sumProbability,
            probabilityUp !== null
                ? formatPercent(
                    probabilityUp
                )
                : "—"
        );

        setElementText(
            sumConfidence,
            confidence !== null
                ? formatPercent(
                    confidence
                )
                : "—"
        );

        setElementText(
            sumSource,
            source
        );
    }


    // =========================================================
    // DIRECTION
    // =========================================================

    function updateDirection(data) {

        const direction =
            data.direction ||
            {};

        const directionName =
            normalizeDirection(
                direction.direction ??
                direction.prediction
            );

        const up =
            probabilityToPercent(
                direction.probability_up
            );

        const down =
            probabilityToPercent(
                direction.probability_down
            );

        const confidence =
            direction.confidence_percent ??
            probabilityToPercent(
                direction.confidence
            );

        setElementText(
            directionValue,
            directionName
        );

        setElementText(
            probabilityUp,
            up !== null
                ? formatPercent(up)
                : "—"
        );

        setElementText(
            probabilityDown,
            down !== null
                ? formatPercent(down)
                : "—"
        );

        setBarWidth(
            probabilityUpBar,
            up
        );

        setBarWidth(
            probabilityDownBar,
            down
        );

        setElementText(
            directionConfidence,
            confidence !== null
                ? formatPercent(confidence)
                : "—"
        );

        setBarWidth(
            confidenceBar,
            confidence
        );


        if (directionIcon) {

            directionIcon.textContent =
                directionName === "UP"
                    ? "↑"
                    : directionName === "DOWN"
                        ? "↓"
                        : "•";
        }
    }


    // =========================================================
    // AI ANALYST
    // =========================================================

    function updateAnalyst(data) {

        const analystData =
            data.analyst ||
            {};

        const technical =
            data.technical ||
            analystData.technical_indicators ||
            {};

        const signal =
            analystData.signal ||
            "NEUTRAL";

        const strength =
            analystData.signal_strength ??
            0;

        setElementText(
            analystSignal,
            signal
        );

        setElementText(
            signalStrength,
            typeof strength === "number"
                ? formatPercent(
                    strength <= 1
                        ? strength * 100
                        : strength
                )
                : strength
        );

        setBarWidth(
            strengthFill,
            strength <= 1
                ? strength * 100
                : strength
        );

        setElementText(
            analystSummary,
            analystData.summary ||
            "No analyst summary available."
        );

        setElementText(
            techRSI,
            technical.rsi !== undefined
                ? getNumber(
                    technical.rsi
                ).toFixed(2)
                : "—"
        );

        setElementText(
            techMACD,
            technical.macd !== undefined
                ? getNumber(
                    technical.macd
                ).toFixed(2)
                : "—"
        );

        setElementText(
            techSMA20,
            technical.sma_20 !== undefined
                ? formatPrice(
                    technical.sma_20
                )
                : "—"
        );

        setElementText(
            techSMA50,
            technical.sma_50 !== undefined
                ? formatPrice(
                    technical.sma_50
                )
                : "—"
        );

        setElementText(
            techSMA200,
            technical.sma_200 !== undefined
                ? formatPrice(
                    technical.sma_200
                )
                : "—"
        );

        const dailyMove =
            technical.daily_change_percent ??
            technical.daily_move ??
            data.market?.daily_change_percent;

        setElementText(
            techDailyMove,
            dailyMove !== undefined
                ? formatSignedPercent(
                    dailyMove
                )
                : "—"
        );

        renderList(
            reasonsList,
            analystData.bullish_points ||
            analystData.reasons,
            "No bullish points reported."
        );

        renderList(
            risksList,
            analystData.risks ||
            analystData.bearish_points,
            "No major risks reported."
        );

        setElementText(
            analystBadge,
            signal
        );
    }


    // =========================================================
    // FORECAST OVERVIEW
    // =========================================================

    function updateForecastOverview(data) {

        const forecast =
            data.forecast ||
            {};

        const prices =
            Array.isArray(
                forecast.prices
            )
                ? forecast.prices
                : [];

        const dates =
            Array.isArray(
                forecast.dates
            )
                ? forecast.dates
                : [];

        const model =
            forecast.model ||
            data.regression?.model ||
            "—";

        setElementText(
            forecastModel,
            model
        );

        setElementText(
            forecastHorizon,
            dates.length
                ? `${dates.length} trading days`
                : "—"
        );

        if (prices.length > 0) {

            setElementText(
                forecastFirstPrice,
                formatPrice(
                    prices[0]
                )
            );

            setElementText(
                forecastFinalPrice,
                formatPrice(
                    prices[
                        prices.length - 1
                    ]
                )
            );

        } else {

            setElementText(
                forecastFirstPrice,
                "—"
            );

            setElementText(
                forecastFinalPrice,
                "—"
            );
        }
    }


    // =========================================================
    // FORECAST TABLE
    // =========================================================

    function updateForecastTable(data) {

        if (!forecastTableBody) {
            return;
        }

        const forecast =
            data.forecast ||
            {};

        const rows =
            Array.isArray(
                forecast.rows
            )
                ? forecast.rows
                : [];

        forecastTableBody.innerHTML = "";

        if (rows.length === 0) {

            const row =
                document.createElement("tr");

            row.innerHTML =
                `<td colspan="3">No forecast data available.</td>`;

            forecastTableBody.appendChild(row);

            return;
        }

        rows.forEach(item => {

            const date =
                item.date ??
                item.Date;

            const price =
                item.predicted_price ??
                item.Predicted_Price ??
                item.price;

            const change =
                item.change_percent ??
                item.Change_Percent ??
                item.change;

            const row =
                document.createElement("tr");

            row.innerHTML = `
                <td>${escapeHtml(
                    formatDate(date)
                )}</td>

                <td>${escapeHtml(
                    formatPrice(price)
                )}</td>

                <td>${escapeHtml(
                    formatSignedPercent(change)
                )}</td>
            `;

            forecastTableBody.appendChild(
                row
            );
        });
    }


    // =========================================================
    // FORECAST EVALUATION
    // =========================================================

    function updateForecastEvaluation(data) {

        if (!forecastEvaluation) {
            return;
        }

        const forecast =
            data.forecast ||
            {};

        const evaluation =
            forecast.evaluation ||
            {};

        const models =
            Object.keys(evaluation);

        if (models.length === 0) {

            forecastEvaluation.innerHTML =
                "<p>No evaluation metrics available.</p>";

            return;
        }

        const rows = [];

        models.forEach(model => {

            const metrics =
                evaluation[model] ||
                {};

            const mae =
                metrics.MAE ??
                metrics.mae;

            const rmse =
                metrics.RMSE ??
                metrics.rmse;

            const r2 =
                metrics.R2 ??
                metrics["R²"] ??
                metrics.r2;

            const directional =
                metrics["Directional Accuracy"] ??
                metrics["Directional_Accuracy"] ??
                metrics.directional_accuracy;

            rows.push(`
                <tr>

                    <td>${escapeHtml(
                        model
                    )}</td>

                    <td>${escapeHtml(
                        Number.isFinite(Number(mae))
                            ? Number(mae).toFixed(4)
                            : "—"
                    )}</td>

                    <td>${escapeHtml(
                        Number.isFinite(Number(rmse))
                            ? Number(rmse).toFixed(4)
                            : "—"
                    )}</td>

                    <td>${escapeHtml(
                        Number.isFinite(Number(r2))
                            ? Number(r2).toFixed(4)
                            : "—"
                    )}</td>

                    <td>${escapeHtml(
                        Number.isFinite(Number(directional))
                            ? formatPercent(
                                directional <= 1
                                    ? directional * 100
                                    : directional
                            )
                            : "—"
                    )}</td>

                </tr>
            `);
        });

        forecastEvaluation.innerHTML = `
            <table class="evaluation-table">

                <thead>
                    <tr>
                        <th>Model</th>
                        <th>MAE</th>
                        <th>RMSE</th>
                        <th>R²</th>
                        <th>Direction</th>
                    </tr>
                </thead>

                <tbody>
                    ${rows.join("")}
                </tbody>

            </table>
        `;
    }


    // =========================================================
    // FORECAST MESSAGE
    // =========================================================

    function updateForecastMessage(data) {

        const forecast =
            data.forecast ||
            {};

        const message =
            forecast.message ||
            data.regression?.message ||
            "";

        setElementText(
            regressionMessage,
            message
        );
    }
    async function loadExplainability(ticker) {
        const predictionElement =
            document.getElementById("explainPrediction");

        const confidenceElement =
            document.getElementById("explainConfidence");

        const methodElement =
            document.getElementById("explainMethod");

        const countElement =
            document.getElementById("explainFeatureCount");

        const listElement =
            document.getElementById("explainabilityList");

        if (!listElement) {
            return;
        }

        // ---------------------------------------------------------
        // Loading state
        // ---------------------------------------------------------

        if (predictionElement) {
            predictionElement.textContent = "Loading...";
        }

        if (confidenceElement) {
            confidenceElement.textContent = "—";
        }

        if (methodElement) {
            methodElement.textContent = "—";
        }

        if (countElement) {
            countElement.textContent = "—";
        }

        listElement.innerHTML = `
            <div class="feature-loading">
                Generating SHAP explanation...
            </div>
        `;

        try {
            const response = await fetch(
                "/api/explain",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        ticker: ticker
                    })
                }
            );

            const data = await response.json();

            console.log(
                "SHAP API response:",
                data
            );

            if (
                !response.ok ||
                !data.success
            ) {
                throw new Error(
                    data.error ||
                    "Unable to generate model explanation."
                );
            }

            // -----------------------------------------------------
            // Summary
            // -----------------------------------------------------

            if (predictionElement) {
                predictionElement.textContent =
                    data.prediction || "—";
            }

            if (confidenceElement) {
                const confidence =
                    Number(
                        data.confidence_percent
                    );

                confidenceElement.textContent =
                    Number.isFinite(confidence)
                        ? `${confidence.toFixed(2)}%`
                        : "—";
            }

            if (methodElement) {
                methodElement.textContent =
                    data.explanation?.method ||
                    "SHAP TreeExplainer";
            }

            const explanations =
                data.explanation?.features || [];

            if (countElement) {
                countElement.textContent =
                    explanations.length;
            }

            // -----------------------------------------------------
            // No explanation data
            // -----------------------------------------------------

            if (!explanations.length) {
                listElement.innerHTML = `
                    <div class="feature-loading">
                        No feature contributions were returned.
                    </div>
                `;

                return;
            }

            // -----------------------------------------------------
            // Render feature contributions
            // -----------------------------------------------------

            listElement.innerHTML =
                explanations
                    .map((feature) => {

                        const contribution =
                            Number(
                                feature.contribution
                            );

                        const value =
                            Number(
                                feature.value
                            );

                        const impact =
                            feature.impact ||
                            (
                                contribution > 0
                                    ? "positive"
                                    : contribution < 0
                                        ? "negative"
                                        : "neutral"
                            );

                        let impactClass =
                            "impact-neutral";

                        let arrow = "→";

                        if (
                            impact === "positive"
                        ) {
                            impactClass =
                                "impact-positive";

                            arrow = "↑";

                        } else if (
                            impact === "negative"
                        ) {
                            impactClass =
                                "impact-negative";

                            arrow = "↓";
                        }

                        const contributionText =
                            Number.isFinite(
                                contribution
                            )
                                ? (
                                    contribution >= 0
                                        ? `+${contribution.toFixed(5)}`
                                        : contribution.toFixed(5)
                                )
                                : "—";

                        const valueText =
                            Number.isFinite(value)
                                ? value.toFixed(6)
                                : "—";

                        return `
                            <div
                                class="
                                    explainability-item
                                    ${impactClass}
                                "
                            >

                                <div
                                    class="
                                        explainability-feature
                                    "
                                >
                                    <span
                                        class="
                                            explainability-feature-name
                                        "
                                    >
                                        ${escapeHtml(
                                            feature.name ||
                                            "Unknown feature"
                                        )}
                                    </span>

                                    <span
                                        class="
                                            explainability-feature-value
                                        "
                                    >
                                        ${valueText}
                                    </span>
                                </div>

                                <div
                                    class="
                                        explainability-contribution
                                    "
                                >
                                    <span
                                        class="
                                            explainability-arrow
                                        "
                                    >
                                        ${arrow}
                                    </span>

                                    <span>
                                        ${contributionText}
                                    </span>
                                </div>

                            </div>
                        `;
                    })
                    .join("");

        } catch (error) {

            console.error(
                "SHAP explanation error:",
                error
            );

            if (predictionElement) {
                predictionElement.textContent =
                    "—";
            }

            if (confidenceElement) {
                confidenceElement.textContent =
                    "—";
            }

            if (methodElement) {
                methodElement.textContent =
                    "—";
            }

            if (countElement) {
                countElement.textContent =
                    "—";
            }

            listElement.innerHTML = `
                <div class="feature-loading">
                    Unable to generate model explanation.
                </div>
            `;
        }
    }

    // =========================================================
    // METADATA
    // =========================================================

    function updateMetadata(data) {

        setElementText(
            historyInfo,
            data.history_period ||
            "2 years"
        );

        setElementText(
            rowsInfo,
            data.history_rows ??
            data.history?.close?.length ??
            "—"
        );

        const horizon =
            data.forecast?.dates?.length ||
            data.future_dates?.length ||
            0;

        setElementText(
            horizonInfo,
            horizon
                ? `${horizon} trading days`
                : "—"
        );
    }


    // =========================================================
    // PLOTLY CHART
    // =========================================================

    function renderChart(data) {

        if (
            !chartDiv ||
            typeof Plotly === "undefined"
        ) {
            return;
        }

        const history =
            data.history ||
            {};

        /*
         * Current app.py returns:
         *
         * history: {
         *     dates: [...],
         *     close: [...]
         * }
         */

        const historicalDates =
            Array.isArray(
                history.dates
            )
                ? history.dates
                : [];

        const historicalClose =
            Array.isArray(
                history.close
            )
                ? history.close
                : [];

        const forecast =
            data.forecast ||
            {};

        const forecastDates =
            Array.isArray(
                forecast.dates
            )
                ? forecast.dates
                : [];

        const forecastPrices =
            Array.isArray(
                forecast.prices
            )
                ? forecast.prices
                : [];

        if (
            historicalDates.length === 0 ||
            historicalClose.length === 0
        ) {

            chartDiv.innerHTML =
                "<p>Chart data unavailable.</p>";

            return;
        }


        // -----------------------------------------------------
        // Historical
        // -----------------------------------------------------

        const historicalTrace = {

            x: historicalDates,

            y: historicalClose,

            mode: "lines",

            name: "Historical Close",

            line: {
                width: 2
            }
        };


        // -----------------------------------------------------
        // Forecast
        // -----------------------------------------------------

        const forecastTrace = {

            x: [
                historicalDates[
                    historicalDates.length - 1
                ],
                ...forecastDates
            ],

            y: [
                historicalClose[
                    historicalClose.length - 1
                ],
                ...forecastPrices
            ],

            mode: "lines+markers",

            name: "Experimental Forecast",

            line: {
                width: 3,
                dash: "dash"
            },

            marker: {
                size: 6
            },

            hovertemplate:
                "%{x}<br>" +
                "Forecast: $%{y:.2f}" +
                "<extra></extra>"
        };


        // -----------------------------------------------------
        // Forecast Start
        // -----------------------------------------------------

        const lastIndex =
            historicalDates.length - 1;

        const forecastStart = {

            x: [
                historicalDates[
                    lastIndex
                ]
            ],

            y: [
                historicalClose[
                    lastIndex
                ]
            ],

            mode: "markers",

            name: "Forecast Start",

            marker: {
                size: 9,
                line: {
                    width: 2
                }
            },

            hovertemplate:
                "%{x}<br>" +
                "Last Close: $%{y:.2f}" +
                "<extra></extra>"
        };


        Plotly.react(
            chartDiv,
            [
                historicalTrace,
                forecastTrace,
                forecastStart
            ],
            {
                template: "plotly_white",

                margin: {
                    l: 40,
                    r: 20,
                    t: 20,
                    b: 40
                },

                xaxis: {
                    title: "Date"
                },

                yaxis: {
                    title: "Price (USD)"
                },

                legend: {
                    orientation: "h",
                    yanchor: "bottom",
                    y: 1.02,
                    xanchor: "right",
                    x: 1
                },

                hovermode: "x unified",

                responsive: true
            },

            {
                responsive: true,
                displaylogo: false
            }
        );
    }


    // =========================================================
    // RELIABILITY DASHBOARD
    // =========================================================

    function updateReliabilityDashboard(validation) {

        if (!validation) {
            return;
        }

        if (reliabilitySection) {

            reliabilitySection.style.display =
                "block";
        }

        updateValidationMetrics(
            validation
        );

        updateValidationBaseline(
            validation
        );

        updateValidationReliability(
            validation
        );

        updateValidationIntegrity(
            validation
        );

        updateClassDistribution(
            validation
        );

        updateFoldDiagnostics(
            validation
        );

        updateModelAssessment(
            validation
        );

        updateFoldTable(
            validation
        );

        updateValidationMetadata(
            validation
        );

        updateValidationNote(
            validation
        );
    }


    // =========================================================
    // VALIDATION METRICS
    // =========================================================

    function updateValidationMetrics(validation) {

        const metrics =
            validation.metrics ||
            {};

        setElementText(
            validationAccuracy,
            formatDecimalPercent(
                metrics.accuracy
            )
        );

        setElementText(
            validationPrecision,
            formatDecimalPercent(
                metrics.precision
            )
        );

        setElementText(
            validationRecall,
            formatDecimalPercent(
                metrics.recall
            )
        );

        setElementText(
            validationF1,
            formatDecimalPercent(
                metrics.f1
            )
        );

        setElementText(
            validationAuc,
            metrics.roc_auc !== null &&
            metrics.roc_auc !== undefined
                ? Number(
                    metrics.roc_auc
                ).toFixed(4)
                : "—"
        );
    }


    // =========================================================
    // BASELINE
    // =========================================================

    function updateValidationBaseline(validation) {

        const baseline =
            validation.baseline ||
            {};

        const metrics =
            validation.metrics ||
            {};

        const reliability =
            validation.reliability ||
            {};


        const modelAccuracy =
            metricToPercent(
                metrics.accuracy
            );

        const baselineAccuracy =
            metricToPercent(
                baseline.accuracy
            );


        setElementText(
            modelBaselineAccuracy,
            modelAccuracy !== null
                ? formatPercent(
                    modelAccuracy
                )
                : "—"
        );


        setElementText(
            majorityBaselineAccuracy,
            baselineAccuracy !== null
                ? formatPercent(
                    baselineAccuracy
                )
                : "—"
        );


        setElementText(
            baselineStrategy,
            baseline.strategy ||
            baseline.majority_class ||
            "Majority class"
        );


        /*
        * Prefer the backend's validated improvement.
        *
        * Backend:
        * -0.02649 = -2.649 percentage points
        *
        * If unavailable, calculate it locally.
        */

        let improvement = null;


        if (
            reliability.baseline_improvement !== undefined &&
            reliability.baseline_improvement !== null
        ) {

            const backendImprovement =
                Number(
                    reliability.baseline_improvement
                );

            if (
                Number.isFinite(
                    backendImprovement
                )
            ) {

                improvement =
                    backendImprovement * 100;
            }

        } else if (
            modelAccuracy !== null &&
            baselineAccuracy !== null
        ) {

            improvement =
                modelAccuracy -
                baselineAccuracy;
        }


        setElementText(
            baselineImprovement,
            improvement !== null
                ? formatPercentagePoints(
                    improvement
                )
                : "—"
        );


        if (!baselineResult) {
            return;
        }


        baselineResult.classList.remove(
            "positive",
            "negative"
        );


        if (
            improvement !== null &&
            improvement > 0
        ) {

            baselineResult.textContent =
                "Above baseline";

            baselineResult.classList.add(
                "positive"
            );

        } else if (
            improvement !== null &&
            improvement < 0
        ) {

            baselineResult.textContent =
                "Below baseline";

            baselineResult.classList.add(
                "negative"
            );

        } else {

            baselineResult.textContent =
                "Near baseline";
        }
    }

    // =========================================================
    // LOAD PRODUCTION MODEL INFORMATION
    // =========================================================

    async function loadModelInfo() {

        try {

            const response = await fetch("/api/model-info");

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(
                    data.error || "Unable to load model information."
                );
            }

            const model = data.model || {};
            const validation = data.validation || {};

            console.log("Production Model Information:", data);

            // -------------------------------------------------
            // MODEL NAME
            // -------------------------------------------------

            const modelName = document.getElementById("modelName");

            if (modelName) {
                modelName.textContent =
                    model.name || "Calibrated Extra Trees Classifier";
            }


            // -------------------------------------------------
            // MODEL TYPE
            // -------------------------------------------------

            const modelType = document.getElementById("modelType");

            if (modelType) {
                modelType.textContent =
                    model.type || "CalibratedClassifierCV";
            }


            // -------------------------------------------------
            // BASE MODEL
            // -------------------------------------------------

            const baseModel = document.getElementById("baseModel");

            if (baseModel) {
                baseModel.textContent =
                    model.base_model || "ExtraTreesClassifier";
            }


            // -------------------------------------------------
            // CALIBRATION
            // -------------------------------------------------

            const calibration = document.getElementById("modelCalibration");

            if (calibration) {

                const calibrationData = model.calibration || {};

                const method = calibrationData.method || "sigmoid";
                const cv = calibrationData.cv || 3;

                calibration.textContent =
                    `${method} (CV=${cv})`;
            }


            // -------------------------------------------------
            // FEATURE COUNT
            // -------------------------------------------------

            const featureCount = document.getElementById("modelFeatureCount");

            if (featureCount) {
                featureCount.textContent =
                    model.feature_count ?? "—";
            }


            // -------------------------------------------------
            // TRAINING ROWS
            // -------------------------------------------------

            const trainingRows = document.getElementById("modelTrainingRows");

            if (trainingRows) {
                trainingRows.textContent =
                    model.training_rows ?? "—";
            }

            // PRODUCTION FEATURES
            const productionFeatureList =
                document.getElementById("productionFeatureList");

            const modelFeatureCountBadge =
                document.getElementById("modelFeatureCountBadge");

            const productionFeatures =
                Array.isArray(model.features)
                    ? model.features
                    : [];

            if (modelFeatureCountBadge) {

                modelFeatureCountBadge.textContent =
                    `${productionFeatures.length} features`;

            }

            if (productionFeatureList) {

                productionFeatureList.innerHTML = "";

                if (productionFeatures.length === 0) {

                    productionFeatureList.innerHTML = `
                        <div class="feature-loading">
                            No production features available.
                        </div>
                    `;

                } else {

                    productionFeatures.forEach(
                        (feature, index) => {

                            const featureItem =
                                document.createElement("div");

                            featureItem.className =
                                "production-feature-item";

                            featureItem.innerHTML = `
                                <span class="feature-index">
                                    ${String(index + 1).padStart(2, "0")}
                                </span>

                                <span class="feature-name">
                                    ${escapeHtml(feature)}
                                </span>
                            `;

                            productionFeatureList.appendChild(
                                featureItem
                            );

                        }
                    );

                }

            }


            // -------------------------------------------------
            // VALIDATION ACCURACY
            // -------------------------------------------------

            const validationAccuracy =
                document.getElementById("modelValidationAccuracy");

            if (validationAccuracy) {

                validationAccuracy.textContent =
                    validation.accuracy_percent != null
                        ? `${validation.accuracy_percent.toFixed(2)}%`
                        : "—";
            }


            // -------------------------------------------------
            // ROC-AUC
            // -------------------------------------------------

            const validationRocAuc =
                document.getElementById("modelValidationRocAuc");

            if (validationRocAuc) {

                validationRocAuc.textContent =
                    validation.roc_auc != null
                        ? validation.roc_auc.toFixed(4)
                        : "—";
            }


            // -------------------------------------------------
            // BRIER SCORE
            // -------------------------------------------------

            const validationBrier =
                document.getElementById("modelBrierScore");

            if (validationBrier) {

                validationBrier.textContent =
                    validation.brier_score != null
                        ? validation.brier_score.toFixed(4)
                        : "—";
            }


            // -------------------------------------------------
            // LOG LOSS
            // -------------------------------------------------

            const validationLogLoss =
                document.getElementById("modelLogLoss");

            if (validationLogLoss) {

                validationLogLoss.textContent =
                    validation.log_loss != null
                        ? validation.log_loss.toFixed(4)
                        : "—";
            }


            // -------------------------------------------------
            // BASELINE
            // -------------------------------------------------

            const validationBaseline =
                document.getElementById("modelBaseline");

            if (validationBaseline) {

                validationBaseline.textContent =
                    validation.baseline_percent != null
                        ? `${validation.baseline_percent.toFixed(2)}%`
                        : "—";
            }


            // -------------------------------------------------
            // IMPROVEMENT OVER BASELINE
            // -------------------------------------------------

            const validationImprovement =
                document.getElementById("modelImprovement");

            if (validationImprovement) {

                if (
                    validation.improvement_over_baseline_percent != null
                ) {

                    const improvement =
                        validation.improvement_over_baseline_percent;

                    const sign = improvement >= 0 ? "+" : "";

                    validationImprovement.textContent =
                        `${sign}${improvement.toFixed(2)} pp`;

                } else {

                    validationImprovement.textContent = "—";
                }
            }


            // -------------------------------------------------
            // VALIDATION METHOD
            // -------------------------------------------------

            const validationMethod =
                document.getElementById("modelValidationMethod");

            if (validationMethod) {

                validationMethod.textContent =
                    validation.method ||
                    "Expanding walk-forward validation";
            }


            // -------------------------------------------------
            // CONFIRMATION SPLITS
            // -------------------------------------------------

            const confirmationSplits =
                document.getElementById("modelConfirmationSplits");

            if (confirmationSplits) {

                confirmationSplits.textContent =
                    validation.confirmation_splits ?? "—";
            }


            // -------------------------------------------------
            // MODEL STATUS
            // -------------------------------------------------

            const modelStatus =
                document.getElementById("modelStatus");

            if (modelStatus) {

                modelStatus.textContent =
                    model.status === "production_candidate"
                        ? "Production Candidate"
                        : (model.status || "Unknown");
            }


            // -------------------------------------------------
            // PROBABILITY NOTE
            // -------------------------------------------------

            const probabilityNote =
                document.getElementById("modelProbabilityNote");

            if (probabilityNote) {

                probabilityNote.textContent =
                    data.probability_note ||
                    "Model probabilities are estimates, not guarantees.";
            }


        } catch (error) {

            console.error(
                "Failed to load production model information:",
                error
            );

        }

    }

    // =========================================================
    // RELIABILITY
    // =========================================================

    function updateValidationReliability(validation) {

        const reliability =
            validation.reliability ||
            {};

        const level =
            reliability.level ||
            "Unknown";

        const note =
            reliability.note ||
            reliability.description ||
            "Validation reliability information is available.";

        setElementText(
            reliabilityLevel,
            level
        );

        setElementText(
            reliabilityDescription,
            note
        );

        setElementText(
            reliabilityStatusText,
            level
        );

        if (reliabilityStatusDot) {

            reliabilityStatusDot.classList.remove(
                "success",
                "warning",
                "error"
            );

            const normalized =
                String(level)
                    .toLowerCase();

            if (
                normalized.includes("high") ||
                normalized.includes("good") ||
                normalized.includes("strong")
            ) {

                reliabilityStatusDot.classList.add(
                    "success"
                );

            } else if (
                normalized.includes("low") ||
                normalized.includes("poor") ||
                normalized.includes("weak")
            ) {

                reliabilityStatusDot.classList.add(
                    "error"
                );

            } else {

                reliabilityStatusDot.classList.add(
                    "warning"
                );
            }
        }

        if (reliabilityStatus) {

            reliabilityStatus.textContent =
                "Validation complete";
        }
    }

    // =========================================================
    // UPDATE PREDICTION TRANSPARENCY
    // =========================================================

    function updatePredictionTransparency(direction) {

        if (!direction) {
            return;
        }


        // -----------------------------------------------------
        // PREDICTION
        // -----------------------------------------------------

        const predictionElement =
            document.getElementById(
                "transparencyPrediction"
            );

        if (predictionElement) {

            const prediction =
                normalizeDirection(
                    direction.prediction ||
                    direction.direction
                );

            predictionElement.textContent =
                prediction || "—";

            predictionElement.classList.remove(
                "prediction-up",
                "prediction-down",
                "prediction-neutral"
            );

            if (prediction === "UP") {

                predictionElement.classList.add(
                    "prediction-up"
                );

            } else if (prediction === "DOWN") {

                predictionElement.classList.add(
                    "prediction-down"
                );

            } else {

                predictionElement.classList.add(
                    "prediction-neutral"
                );

            }

        }


        // -----------------------------------------------------
        // UP PROBABILITY
        // -----------------------------------------------------

        const probabilityUpElement =
            document.getElementById(
                "transparencyProbabilityUp"
            );

        if (probabilityUpElement) {

            probabilityUpElement.textContent =
                probabilityToPercent(
                    direction.probability_up
                );

        }


        // -----------------------------------------------------
        // DOWN PROBABILITY
        // -----------------------------------------------------

        const probabilityDownElement =
            document.getElementById(
                "transparencyProbabilityDown"
            );

        if (probabilityDownElement) {

            probabilityDownElement.textContent =
                probabilityToPercent(
                    direction.probability_down
                );

        }
        const confidenceWarning =
            document.getElementById(
                "confidenceWarning"
            );

        if (confidenceWarning) {

            const level =
                direction.confidence_level ||
                "LOW";

            confidenceWarning.hidden =
                level !== "LOW";
        }


        // -----------------------------------------------------
        // CONFIDENCE
        // -----------------------------------------------------

        const confidenceElement =
            document.getElementById(
                "transparencyConfidence"
            );

        if (confidenceElement) {

            confidenceElement.textContent =
                direction.confidence_percent != null
                    ? `${Number(
                        direction.confidence_percent
                    ).toFixed(2)}%`
                    : "—";

        }
        const confidenceLevelElement =
            document.getElementById(
                "transparencyConfidenceLevel"
            );

        if (confidenceLevelElement) {

            const level =
                direction.confidence_level ||
                "LOW";

            confidenceLevelElement.textContent =
                level.replace("_", " ");

            confidenceLevelElement.classList.remove(
                "confidence-low",
                "confidence-moderate",
                "confidence-strong",
                "confidence-very-strong"
            );

            if (level === "LOW") {

                confidenceLevelElement.classList.add(
                    "confidence-low"
                );

            } else if (level === "MODERATE") {

                confidenceLevelElement.classList.add(
                    "confidence-moderate"
                );

            } else if (level === "STRONG") {

                confidenceLevelElement.classList.add(
                    "confidence-strong"
                );

            } else if (level === "VERY_STRONG") {

                confidenceLevelElement.classList.add(
                    "confidence-very-strong"
                );
            }
        }

        // -----------------------------------------------------
        // FEATURE VALUES
        // -----------------------------------------------------

        const featureList =
            document.getElementById(
                "transparencyFeatureList"
            );

        const featureCount =
            document.getElementById(
                "transparencyFeatureCount"
            );

        const featureValues =
            direction.feature_values || {};

        const featureNames =
            Object.keys(featureValues);


        if (featureCount) {

            featureCount.textContent =
                `${featureNames.length} features`;

        }


        if (!featureList) {
            return;
        }


        featureList.innerHTML = "";


        if (featureNames.length === 0) {

            featureList.innerHTML = `
                <div class="feature-loading">
                    No production feature values available.
                </div>
            `;

            return;

        }


        featureNames.forEach(
            (featureName, index) => {

                const value =
                    Number(featureValues[featureName]);

                const item =
                    document.createElement("div");

                item.className =
                    "transparency-feature-item";


                const formattedValue =
                    Number.isFinite(value)
                        ? value.toFixed(6)
                        : "—";


                item.innerHTML = `
                    <span class="feature-index">
                        ${String(index + 1).padStart(2, "0")}
                    </span>

                    <span class="transparency-feature-name">
                        ${escapeHtml(featureName)}
                    </span>

                    <span class="transparency-feature-value">
                        ${formattedValue}
                    </span>
                `;


                featureList.appendChild(item);

            }
        );

    }

    // =========================================================
    // PREDICTION UNCERTAINTY
    // =========================================================

    function updatePredictionUncertainty(direction) {

        const levelElement = document.getElementById("uncertaintyLevel");
        const indicatorElement = document.getElementById("uncertaintyIndicator");
        const spreadElement = document.getElementById("uncertaintySpread");
        const probabilityUpElement = document.getElementById(
            "uncertaintyProbabilityUp"
        );
        const probabilityDownElement = document.getElementById(
            "uncertaintyProbabilityDown"
        );
        const summaryLevelElement = document.getElementById(
            "predictionUncertaintyLevel"
        );

        const summarySpreadElement = document.getElementById(
            "predictionUncertaintySpread"
        );

        if (
            !levelElement ||
            !indicatorElement ||
            !spreadElement ||
            !probabilityUpElement ||
            !probabilityDownElement
        ) {
            return;
        }

        if (!direction) {
            levelElement.textContent = "—";
            indicatorElement.textContent = "—";
            spreadElement.textContent = "—";
            probabilityUpElement.textContent = "—";
            probabilityDownElement.textContent = "—";
            summarySpreadElement.textContent = "—";
            summaryLevelElement.textContent = "—";
            return;
        }

        const uncertainty = direction.uncertainty;

        if (!uncertainty) {
            levelElement.textContent = "—";
            indicatorElement.textContent = "—";
            spreadElement.textContent = "—";
            probabilityUpElement.textContent = "—";
            probabilityDownElement.textContent = "—";
            summaryLevelElement.textContent = "—";
            summarySpreadElement.textContent = "—";
            return;
        }

        const uncertaintyLevel =
            uncertainty.uncertainty_level || "UNKNOWN";

        const spreadPercent =
            Number(uncertainty.probability_spread_percent);

        const probabilityUp =
            Number(direction.probability_up) * 100;

        const probabilityDown =
            Number(direction.probability_down) * 100;


        // ---------------------------------------------------------
        // Uncertainty level
        // ---------------------------------------------------------

        levelElement.textContent =
            uncertaintyLevel.replace("_", " ");

        indicatorElement.textContent =
            uncertaintyLevel.replace("_", " ");


        // ---------------------------------------------------------
        // Probability spread
        // ---------------------------------------------------------

        if (Number.isFinite(spreadPercent)) {
            spreadElement.textContent =
                `${spreadPercent.toFixed(2)} pp`;
        } else {
            spreadElement.textContent = "—";
        }


        // ---------------------------------------------------------
        // UP probability
        // ---------------------------------------------------------

        if (Number.isFinite(probabilityUp)) {
            probabilityUpElement.textContent =
                `${probabilityUp.toFixed(2)}%`;
        } else {
            probabilityUpElement.textContent = "—";
        }


        // ---------------------------------------------------------
        // DOWN probability
        // ---------------------------------------------------------

        if (Number.isFinite(probabilityDown)) {
            probabilityDownElement.textContent =
                `${probabilityDown.toFixed(2)}%`;
        } else {
            probabilityDownElement.textContent = "—";
        }


        // ---------------------------------------------------------
        // Apply uncertainty state class
        // ---------------------------------------------------------

        indicatorElement.classList.remove(
            "uncertainty-high",
            "uncertainty-moderate",
            "uncertainty-low",
            "uncertainty-very-low",
            "uncertainty-unknown"
        );

        const normalizedLevel =
            uncertaintyLevel.toLowerCase();
        
        // ---------------------------------------------------------
        // Main prediction summary
        // ---------------------------------------------------------

        if (summaryLevelElement) {

            summaryLevelElement.textContent =
                `${uncertaintyLevel.replace("_", " ")} UNCERTAINTY`;

            summaryLevelElement.classList.remove(
                "uncertainty-high",
                "uncertainty-moderate",
                "uncertainty-low",
                "uncertainty-very-low",
                "uncertainty-unknown"
            );

            if (normalizedLevel === "high") {

                summaryLevelElement.classList.add(
                    "uncertainty-high"
                );

            } else if (normalizedLevel === "moderate") {

                summaryLevelElement.classList.add(
                    "uncertainty-moderate"
                );

            } else if (normalizedLevel === "low") {

                summaryLevelElement.classList.add(
                    "uncertainty-low"
                );

            } else if (normalizedLevel === "very_low") {

                summaryLevelElement.classList.add(
                    "uncertainty-very-low"
                );

            } else {

                summaryLevelElement.classList.add(
                    "uncertainty-unknown"
                );
            }
        }


        if (summarySpreadElement) {

            if (Number.isFinite(spreadPercent)) {

                summarySpreadElement.textContent =
                    `${spreadPercent.toFixed(2)} pp probability spread`;

            } else {

                summarySpreadElement.textContent = "—";
            }
        }

            // ---------------------------------------------------------
        // Dynamic uncertainty warning
        // ---------------------------------------------------------

        const warningElement = document.getElementById(
            "uncertaintyWarning"
        );

        const warningIconElement = document.getElementById(
            "uncertaintyWarningIcon"
        );

        const warningTitleElement = document.getElementById(
            "uncertaintyWarningTitle"
        );

        const warningMessageElement = document.getElementById(
            "uncertaintyWarningMessage"
        );

        if (
            warningElement &&
            warningIconElement &&
            warningTitleElement &&
            warningMessageElement
        ) {

            warningElement.classList.remove(
                "uncertainty-warning-high",
                "uncertainty-warning-moderate",
                "uncertainty-warning-low",
                "uncertainty-warning-very-low",
                "uncertainty-warning-unknown"
            );


            if (normalizedLevel === "high") {

                warningElement.classList.add(
                    "uncertainty-warning-high"
                );

                warningIconElement.textContent = "!";

                warningTitleElement.textContent =
                    "High model uncertainty";

                warningMessageElement.textContent =
                    "The UP and DOWN probabilities are relatively close. " +
                    "The classifier has only a weak directional edge, " +
                    "so this prediction should be treated with extra caution.";


            } else if (normalizedLevel === "moderate") {

                warningElement.classList.add(
                    "uncertainty-warning-moderate"
                );

                warningIconElement.textContent = "!";

                warningTitleElement.textContent =
                    "Moderate model uncertainty";

                warningMessageElement.textContent =
                    "The classifier has a modest directional edge. " +
                    "The prediction is probabilistic and should not " +
                    "be treated as a guaranteed market outcome.";


            } else if (normalizedLevel === "low") {

                warningElement.classList.add(
                    "uncertainty-warning-low"
                );

                warningIconElement.textContent = "✓";

                warningTitleElement.textContent =
                    "Lower model uncertainty";

                warningMessageElement.textContent =
                    "The classifier shows a clearer separation between " +
                    "the UP and DOWN probabilities. This indicates a " +
                    "stronger model signal, but it still does not " +
                    "guarantee future market performance.";


            } else if (normalizedLevel === "very_low") {

                warningElement.classList.add(
                    "uncertainty-warning-very-low"
                );

                warningIconElement.textContent = "✓";

                warningTitleElement.textContent =
                    "Very low model uncertainty";

                warningMessageElement.textContent =
                    "The classifier shows a strong probability separation. " +
                    "This represents a stronger model signal, not a " +
                    "guarantee of future market direction.";


            } else {

                warningElement.classList.add(
                    "uncertainty-warning-unknown"
                );

                warningIconElement.textContent = "!";

                warningTitleElement.textContent =
                    "Uncertainty information unavailable";

                warningMessageElement.textContent =
                    "The classifier did not provide enough information " +
                    "to determine the current uncertainty level.";
            }
        }

        if (normalizedLevel === "high") {

            indicatorElement.classList.add(
                "uncertainty-high"
            );

        } else if (normalizedLevel === "moderate") {

            indicatorElement.classList.add(
                "uncertainty-moderate"
            );

        } else if (normalizedLevel === "low") {

            indicatorElement.classList.add(
                "uncertainty-low"
            );

        } else if (normalizedLevel === "very_low") {

            indicatorElement.classList.add(
                "uncertainty-very-low"
            );

        } else {

            indicatorElement.classList.add(
                "uncertainty-unknown"
            );
        }
    }

    // =========================================================
    // PREDICTION INTEGRITY
    // =========================================================

    function updatePredictionIntegrity(data) {

        console.log(
            "Prediction integrity data:",
            data
        );

        const validationAccuracy =
            document.getElementById(
                "integrityValidationAccuracy"
            );

        const validationRocAuc =
            document.getElementById(
                "integrityValidationRocAuc"
            );

        const prediction =
            document.getElementById(
                "integrityPrediction"
            );

        const predictionProbability =
            document.getElementById(
                "integrityPredictionProbability"
            );

        const analystSignal =
            document.getElementById(
                "integrityAnalystSignal"
            );

        const analystStrength =
            document.getElementById(
                "integrityAnalystStrength"
            );


        // =========================================================
        // CURRENT CLASSIFIER
        // =========================================================

        const direction =
            data.direction || {};

        if (prediction) {

            prediction.textContent =
                direction.direction ||
                "—";
        }


        if (predictionProbability) {

            const probabilityUp =
                Number(
                    direction.probability_up
                );

            predictionProbability.textContent =
                Number.isFinite(
                    probabilityUp
                )
                    ? `${(
                        probabilityUp * 100
                    ).toFixed(2)}% UP`
                    : "—";
        }


        // =========================================================
        // AI TECHNICAL ANALYST
        // =========================================================

        const analyst =
            data.analyst || {};

        if (analystSignal) {

            analystSignal.textContent =
                analyst.signal ||
                "—";
        }


        if (analystStrength) {

            analystStrength.textContent =
                analyst.strength ||
                analyst.signal_strength ||
                analyst.direction_classifier?.strength ||
                "—";
        }

        // =========================================================
        // VALIDATION
        // =========================================================
        //
        // Validation metrics are loaded separately from
        // /api/model-info.
        //
        // If they have already been stored globally, use them.
        //
        // Otherwise the function will fetch them here.
        // =========================================================

        fetch("/api/model-info")
            .then(response => {

                if (!response.ok) {
                    throw new Error(
                        "Unable to load model information."
                    );
                }

                return response.json();
            })
            .then(modelData => {

                if (
                    !modelData ||
                    !modelData.success
                ) {
                    return;
                }

                const validation =
                    modelData.validation || {};


                // -------------------------------------------------
                // Validation accuracy
                // -------------------------------------------------

                if (validationAccuracy) {

                    const accuracy =
                        Number(
                            validation.accuracy_percent
                        );

                    validationAccuracy.textContent =
                        Number.isFinite(accuracy)
                            ? `${accuracy.toFixed(2)}%`
                            : "—";
                }


                // -------------------------------------------------
                // ROC-AUC
                // -------------------------------------------------

                if (validationRocAuc) {

                    const rocAuc =
                        Number(
                            validation.roc_auc
                        );

                    validationRocAuc.textContent =
                        Number.isFinite(rocAuc)
                            ? rocAuc.toFixed(4)
                            : "—";
                }

            })
            .catch(error => {

                console.error(
                    "Validation information error:",
                    error
                );

                if (validationAccuracy) {
                    validationAccuracy.textContent =
                        "—";
                }

                if (validationRocAuc) {
                    validationRocAuc.textContent =
                        "—";
                }
            });
    }

    // =========================================================
    // STAGE 2B — INTEGRITY
    // =========================================================

    function updateValidationIntegrity(validation) {

        const integrity =
            validation.integrity ||
            {};

        /*
        * Current backend:
        *
        * {
        *     passed: true,
        *     observations: 151,
        *     fold_test_rows: 151,
        *     confusion_matrix_total: 151,
        *     baseline_observations: 151
        * }
        */

        const valid =
            integrity.passed ??
            integrity.valid ??
            integrity.verified ??
            (
                String(
                    integrity.status ||
                    ""
                )
                    .toLowerCase()
                    .includes("valid")
            );

        const observations =
            integrity.observations ??
            validation.observations;

        setElementText(
            validationIntegrityIndicator,
            valid
                ? "✓ VERIFIED"
                : "⚠ FAILED"
        );

        setElementText(
            validationIntegrityValue,
            valid
                ? "PASSED"
                : "FAILED"
        );

        let description;

        if (valid) {

            description =
                "Validation totals are internally consistent.";

        } else {

            description =
                integrity.description ||
                integrity.message ||
                "Validation integrity checks did not pass.";
        }

        /*
        * Keep the observation count available in the
        * description when supplied by the backend.
        */

        if (
            valid &&
            observations !== undefined
        ) {

            description +=
                ` ${observations} observations verified.`;
        }

        setElementText(
            validationIntegrityDescription,
            description
        );

        if (validationIntegrityIndicator) {

            validationIntegrityIndicator.classList.remove(
                "positive",
                "negative",
                "success",
                "error"
            );

            validationIntegrityIndicator.classList.add(
                valid
                    ? "positive"
                    : "negative"
            );
        }
    }

    // =========================================================
    // STAGE 2B — CLASS DISTRIBUTION
    // =========================================================

    function updateClassDistribution(validation) {

        const distribution =
            validation.class_distribution ||
            validation.classDistribution ||
            {};

        const upData =
            distribution.UP ||
            distribution.up ||
            distribution["1"] ||
            {};

        const downData =
            distribution.DOWN ||
            distribution.down ||
            distribution["0"] ||
            {};


        let upCount;
        let downCount;

        let upPercentage;
        let downPercentage;


        /*
        * Current backend structure:
        *
        * UP: {
        *     count: 81,
        *     percentage: 0.5364
        * }
        */

        if (
            typeof upData === "object" &&
            upData !== null
        ) {

            upCount =
                Number(upData.count);

            upPercentage =
                metricToPercent(
                    upData.percentage
                );

        } else {

            upCount =
                Number(upData);

            upPercentage =
                null;
        }


        if (
            typeof downData === "object" &&
            downData !== null
        ) {

            downCount =
                Number(downData.count);

            downPercentage =
                metricToPercent(
                    downData.percentage
                );

        } else {

            downCount =
                Number(downData);

            downPercentage =
                null;
        }


        /*
        * Fallback calculation if percentages were not
        * supplied by the backend.
        */

        if (
            !Number.isFinite(upPercentage) &&
            Number.isFinite(upCount) &&
            Number.isFinite(downCount) &&
            upCount + downCount > 0
        ) {

            upPercentage =
                (
                    upCount /
                    (upCount + downCount)
                ) * 100;
        }


        if (
            !Number.isFinite(downPercentage) &&
            Number.isFinite(upCount) &&
            Number.isFinite(downCount) &&
            upCount + downCount > 0
        ) {

            downPercentage =
                (
                    downCount /
                    (upCount + downCount)
                ) * 100;
        }


        setElementText(
            validationUpPercent,
            Number.isFinite(upPercentage)
                ? formatPercent(
                    upPercentage
                )
                : "—"
        );


        setElementText(
            validationDownPercent,
            Number.isFinite(downPercentage)
                ? formatPercent(
                    downPercentage
                )
                : "—"
        );


        setBarWidth(
            validationUpBar,
            upPercentage
        );


        setBarWidth(
            validationDownBar,
            downPercentage
        );


        const safeUpCount =
            Number.isFinite(upCount)
                ? upCount
                : 0;

        const safeDownCount =
            Number.isFinite(downCount)
                ? downCount
                : 0;


        setElementText(
            validationClassDescription,
            `UP: ${safeUpCount} observations · DOWN: ${safeDownCount} observations`
        );
    }

    // =========================================================
    // STAGE 2B — FOLD DIAGNOSTICS
    // =========================================================

    function updateFoldDiagnostics(validation) {

        const diagnostics =
            validation.fold_diagnostics ||
            validation.foldDiagnostics ||
            {};

        const averageAccuracy =
            diagnostics.average_accuracy;

        const accuracyStd =
            diagnostics.accuracy_std;

        const accuracyRange =
            diagnostics.accuracy_range;


        setElementText(
            foldAverageAccuracy,
            averageAccuracy !== undefined
                ? formatPercent(
                    metricToPercent(
                        averageAccuracy
                    )
                )
                : "—"
        );


        if (
            accuracyRange !== undefined
        ) {

            setElementText(
                foldAccuracyRange,
                formatPercentagePoints(
                    metricToPercent(
                        accuracyRange
                    )
                )
            );

        } else if (
            accuracyStd !== undefined
        ) {

            setElementText(
                foldAccuracyRange,
                formatPercent(
                    metricToPercent(
                        accuracyStd
                    )
                )
            );

        } else {

            setElementText(
                foldAccuracyRange,
                "—"
            );
        }


        const strongest =
            diagnostics.strongest_fold;

        const weakest =
            diagnostics.weakest_fold;


        if (
            strongest &&
            typeof strongest === "object"
        ) {

            setElementText(
                strongestFold,
                `Fold ${strongest.fold} · ${formatPercent(
                    metricToPercent(
                        strongest.accuracy
                    )
                )}`
            );

        } else {

            setElementText(
                strongestFold,
                "—"
            );
        }


        if (
            weakest &&
            typeof weakest === "object"
        ) {

            setElementText(
                weakestFold,
                `Fold ${weakest.fold} · ${formatPercent(
                    metricToPercent(
                        weakest.accuracy
                    )
                )}`
            );

        } else {

            setElementText(
                weakestFold,
                "—"
            );
        }
    }

    // =========================================================
    // STAGE 2B — MODEL ASSESSMENT
    // =========================================================

    function updateModelAssessment(validation) {

        const assessment =
            validation.model_assessment ||
            validation.modelAssessment ||
            {};

        const metrics =
            validation.metrics ||
            {};

        const baseline =
            validation.baseline ||
            {};

        const reliability =
            validation.reliability ||
            {};


        const modelAccuracy =
            metricToPercent(
                metrics.accuracy
            );

        const baselineAccuracy =
            metricToPercent(
                baseline.accuracy
            );


        /*
        * Backend reliability.baseline_improvement
        * is a DECIMAL difference:
        *
        * -0.02649 = -2.649 percentage points
        */

        let delta =
            reliability.baseline_improvement;


        if (
            delta === undefined ||
            delta === null
        ) {

            if (
                modelAccuracy !== null &&
                baselineAccuracy !== null
            ) {

                delta =
                    (
                        modelAccuracy -
                        baselineAccuracy
                    ) / 100;
            }
        }


        const deltaPercentagePoints =
            delta !== null &&
            delta !== undefined
                ? Number(delta) * 100
                : null;


        let assessmentText =
            assessment.result ||
            assessment.label ||
            assessment.status;


        if (!assessmentText) {

            if (
                deltaPercentagePoints !== null &&
                Number.isFinite(
                    deltaPercentagePoints
                )
            ) {

                if (
                    deltaPercentagePoints > 0.5
                ) {

                    assessmentText =
                        "Outperforms baseline";

                } else if (
                    deltaPercentagePoints < -0.5
                ) {

                    assessmentText =
                        "Underperforms baseline";

                } else {

                    assessmentText =
                        "Near baseline";
                }

            } else {

                assessmentText =
                    "Assessment unavailable";
            }
        }


        let description =
            assessment.description ||
            assessment.message;


        if (!description) {

            if (
                deltaPercentagePoints !== null &&
                Number.isFinite(
                    deltaPercentagePoints
                )
            ) {

                if (
                    deltaPercentagePoints < 0
                ) {

                    description =
                        "The classifier performs below the majority-class baseline.";

                } else if (
                    deltaPercentagePoints > 0
                ) {

                    description =
                        "The classifier performs above the majority-class baseline.";

                } else {

                    description =
                        "The classifier performs at the majority-class baseline.";
                }

            } else {

                description =
                    reliability.interpretation ||
                    "Model assessment is based on validation performance relative to the baseline.";
            }
        }


        setElementText(
            modelAssessmentIndicator,
            deltaPercentagePoints !== null
                ? (
                    deltaPercentagePoints > 0
                        ? "✓"
                        : deltaPercentagePoints < 0
                            ? "⚠"
                            : "•"
                )
                : "•"
        );


        setElementText(
            modelAssessment,
            assessmentText
        );


        setElementText(
            modelAssessmentDescription,
            description
        );


        setElementText(
            modelAssessmentDelta,
            deltaPercentagePoints !== null
                ? formatPercentagePoints(
                    deltaPercentagePoints
                )
                : "—"
        );


        if (modelAssessmentIndicator) {

            modelAssessmentIndicator.classList.remove(
                "positive",
                "negative",
                "neutral"
            );


            if (
                deltaPercentagePoints !== null
            ) {

                modelAssessmentIndicator.classList.add(
                    deltaPercentagePoints > 0
                        ? "positive"
                        : deltaPercentagePoints < 0
                            ? "negative"
                            : "neutral"
                );
            }
        }
    }

    // =========================================================
    // FOLD TABLE
    // =========================================================

   function updateFoldTable(validation) {

        if (!foldTableBody) {
            return;
        }

        const folds =
            Array.isArray(
                validation.folds
            )
                ? validation.folds
                : [];

        foldTableBody.innerHTML = "";


        if (folds.length === 0) {

            const row =
                document.createElement("tr");

            row.innerHTML =
                `<td colspan="8">No fold data available.</td>`;

            foldTableBody.appendChild(
                row
            );

            return;
        }


        folds.forEach(
            (fold, index) => {

                const row =
                    document.createElement("tr");

                const foldNumber =
                    fold.fold ??
                    index + 1;

                const trainRows =
                    fold.train_rows ??
                    fold.train_size ??
                    "—";

                const testRows =
                    fold.test_rows ??
                    fold.test_size ??
                    "—";

                const accuracy =
                    metricToPercent(
                        fold.accuracy
                    );

                const precision =
                    metricToPercent(
                        fold.precision
                    );

                const recall =
                    metricToPercent(
                        fold.recall
                    );

                const f1 =
                    metricToPercent(
                        fold.f1
                    );

                const auc =
                    fold.roc_auc !== null &&
                    fold.roc_auc !== undefined
                        ? Number(
                            fold.roc_auc
                        ).toFixed(4)
                        : "—";


                row.innerHTML = `

                    <td>${escapeHtml(
                        foldNumber
                    )}</td>

                    <td>${escapeHtml(
                        trainRows
                    )}</td>

                    <td>${escapeHtml(
                        testRows
                    )}</td>

                    <td>${escapeHtml(
                        accuracy !== null
                            ? formatPercent(
                                accuracy
                            )
                            : "—"
                    )}</td>

                    <td>${escapeHtml(
                        precision !== null
                            ? formatPercent(
                                precision
                            )
                            : "—"
                    )}</td>

                    <td>${escapeHtml(
                        recall !== null
                            ? formatPercent(
                                recall
                            )
                            : "—"
                    )}</td>

                    <td>${escapeHtml(
                        f1 !== null
                            ? formatPercent(
                                f1
                            )
                            : "—"
                    )}</td>

                    <td>${escapeHtml(
                        auc
                    )}</td>

                `;

                foldTableBody.appendChild(
                    row
                );
            }
        );
    }

    // =========================================================
    // VALIDATION METADATA
    // =========================================================

    function updateValidationMetadata(validation) {

        const observations =
            validation.observations ??
            validation.total_observations ??
            validation.rows;

        const splits =
            validation.splits ??
            validation.n_splits ??
            validation.fold_count;

        const features =
            validation.features ??
            validation.selected_features ??
            validation.feature_count;

        setElementText(
            validationObservations,
            observations
        );

        setElementText(
            validationSplits,
            splits
        );

        if (
            Array.isArray(features)
        ) {

            setElementText(
                validationFeatures,
                features.length
            );

        } else {

            setElementText(
                validationFeatures,
                features
            );
        }
    }


    // =========================================================
    // VALIDATION NOTE
    // =========================================================

    function updateValidationNote(validation) {

        const reliability =
            validation.reliability ||
            {};

        const note =
            validation.note ||
            validation.validation_note ||
            reliability.note ||
            "Walk-forward validation evaluates the classifier on sequential unseen data.";

        setElementText(
            validationNote,
            note
        );
    }


    // =========================================================
    // EVENT LISTENERS
    // =========================================================

    if (analyzeBtn) {

        analyzeBtn.addEventListener(
            "click",
            analyzeStock
        );
    }


    if (validateModelBtn) {

        validateModelBtn.addEventListener(
            "click",
            validateModel
        );
    }


    if (tickerInput) {

        tickerInput.addEventListener(
            "input",
            () => {

                tickerInput.value =
                    tickerInput.value
                        .toUpperCase()
                        .replace(
                            /[^A-Z0-9.-]/g,
                            ""
                        )
                        .slice(
                            0,
                            10
                        );
            }
        );
    }


    if (daysInput) {

        daysInput.addEventListener(
            "change",
            () => {

                let value =
                    parseInt(
                        daysInput.value,
                        10
                    );

                if (!Number.isFinite(value)) {
                    value = 7;
                }

                value =
                    Math.max(
                        1,
                        Math.min(
                            30,
                            value
                        )
                    );

                daysInput.value =
                    value;
            }
        );
    }


    // =========================================================
    // ENTER KEY
    // =========================================================

    if (tickerInput) {

        tickerInput.addEventListener(
            "keydown",
            event => {

                if (
                    event.key === "Enter"
                ) {

                    event.preventDefault();

                    analyzeStock();
                }
            }
        );
    }


    // =========================================================
    // INITIAL UI STATE
    // =========================================================

    if (buttonLoader) {
        buttonLoader.style.display = "none";
    }

    if (validateButtonLoader) {
        validateButtonLoader.style.display = "none";
    }

    if (reliabilitySection) {
        reliabilitySection.style.display = "none";
    }


    loadModelInfo();
    // =========================================================
    // DEBUG
    // =========================================================

    console.log(
        "StockSense frontend initialized."
    );

});