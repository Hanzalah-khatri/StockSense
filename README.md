# StockSense — AI Stock Intelligence

StockSense is an AI-powered stock analysis and market-direction prediction system built with Python, Flask, machine learning, technical analysis, and interactive data visualization.

The system combines a **production direction classifier**, technical indicators, deterministic AI-assisted analysis, SHAP-based model explainability, uncertainty measurement, and an experimental price-forecasting module into a single web dashboard.

> **Disclaimer:** StockSense is an educational and portfolio project. Its predictions and probabilities are model estimates, not financial advice, investment recommendations, or guarantees of future market performance.

---

## 🚀 Features

### 📈 Market Analysis

* Real-time market data through Yahoo Finance
* Current price and daily price movement
* Historical market data
* Interactive Plotly price charts
* Automatic date-range handling

### 🤖 Production Direction Classifier

StockSense uses a calibrated Extra Trees classifier to estimate the next market direction:

* **UP**
* **DOWN**

The production classifier uses:

```text
Return_Lag_1
Return_Lag_3
Return_Lag_5
Volatility
Volume_Change
```

The model outputs:

* Predicted direction
* UP probability
* DOWN probability
* Classifier confidence
* Confidence level
* Probability spread
* Model uncertainty level

Probabilities represent estimated model outputs and should not be interpreted as the probability of making a profit.

---

## 🧠 Machine Learning Pipeline

The production direction model is based on:

```text
Historical Market Data
        ↓
Feature Engineering
        ↓
Production Feature Selection
        ↓
Extra Trees Classifier
        ↓
Probability Calibration
        ↓
Prediction
        ↓
Confidence + Uncertainty
        ↓
Dashboard
```

### Production Model

```text
Model: Extra Trees Classifier
Calibration: Sigmoid
Estimators: 300
Maximum Depth: 15
Minimum Samples Split: 10
Class Weight: Balanced
Random State: 42
```

The production artifact is:

```text
models/stocksense_calibrated_classifier.pkl
```

---

## 📊 Model Validation

The production classifier was evaluated using **expanding walk-forward validation** rather than a random train/test split.

### Validation results

| Metric                  |                  Result |
| ----------------------- | ----------------------: |
| Accuracy                |                  57.62% |
| Majority Baseline       |                  53.64% |
| Improvement vs Baseline | +3.98 percentage points |
| ROC-AUC                 |                  0.5723 |
| Brier Score             |                  0.2454 |
| Log Loss                |                  0.6840 |
| Confirmation Splits     |                       7 |

The baseline represents the majority-class prediction rate used for comparison.

The results indicate that the classifier provides a measurable improvement over the majority baseline on the evaluated historical data, but the edge is modest and should not be interpreted as evidence of guaranteed future performance.

---

## 🎯 Probability Calibration

StockSense uses sigmoid probability calibration on the Extra Trees classifier.

Calibration was selected because it improved probability-quality metrics during evaluation:

* Brier score improved from approximately **0.2498 → 0.2454**
* Log loss improved from approximately **0.6949 → 0.6840**
* ROC-AUC improved from approximately **0.5649 → 0.5723**

Accuracy changed from approximately **58.28% → 57.62%**.

The calibrated model was therefore selected because the project prioritizes not only classification accuracy but also more reliable probability estimates.

---

## 🔍 SHAP Explainability

StockSense includes SHAP-based model explainability.

The dashboard can show how the production model's underlying Extra Trees classifier is influenced by each production feature.

Current production features:

```text
Return_Lag_1
Return_Lag_3
Return_Lag_5
Volatility
Volume_Change
```

The explanation pipeline uses:

```text
SHAP TreeExplainer
        ↓
Extra Trees Base Model
        ↓
Feature Contributions
        ↓
Dashboard Explanation
```

### Important limitation

SHAP explains feature contributions from the **Extra Trees base model**.

It does not directly explain the sigmoid calibration layer itself.

---

## ⚠️ Prediction Uncertainty

StockSense separates **model uncertainty** from financial risk.

The uncertainty layer calculates the probability spread:

```text
|UP Probability − DOWN Probability|
```

A smaller spread means the classifier is closer to a 50/50 decision and therefore has greater model uncertainty.

Example:

```text
UP:   55.46%
DOWN: 44.54%

Probability spread: 10.91 percentage points
```

This represents uncertainty in the model's directional classification.

It does **not** represent:

* Probability of profit
* Expected return
* Financial risk
* Investment safety
* Guaranteed prediction accuracy

---

## 🧮 AI Market Analyst

StockSense also contains a deterministic technical-analysis layer that evaluates market conditions using indicators such as:

* RSI
* SMA 20
* SMA 50
* SMA 200
* MACD
* MACD Signal
* MACD Histogram
* Momentum
* Volatility
* Daily price movement

The analyst produces:

* Bullish signals
* Bearish signals
* Technical reasons
* Risk factors
* Overall signal
* Signal strength

The AI Analyst is a **rule-based/technical analysis component**, separate from the machine-learning direction classifier.

---

## 📉 Experimental Price Forecast

StockSense also includes an experimental regression-based price forecasting module.

Current regression models include:

* Ridge
* Random Forest
* Gradient Boosting

The forecasting component is intentionally treated as **experimental** and is not the production direction model.

Historical evaluation showed that the regression models did not consistently outperform a naive baseline. Therefore, their forecasts should not be presented as highly reliable price predictions.

The production decision-making layer is the **direction classifier**, not the experimental price forecast.

---

## 🏗️ Project Architecture

```text
StockSense
│
├── app.py
│
├── services/
│   ├── market_data.py
│   ├── prediction_service.py
│   ├── classifier_service.py
│   ├── analyst_service.py
│   ├── validation_service.py
│   └── explainability_service.py
│
├── models/
│   ├── feature_engineering.py
│   ├── model_evaluation.py
│   ├── model_manager.py
│   ├── stocksense_calibrated_classifier.pkl
│   ├── stocksense_extra_trees_classifier.pkl
│   ├── stocksense_rf_classifier.pkl
│   ├── calibrated_model_metadata.json
│   ├── extra_trees_model_metadata.json
│   └── model_metadata.json
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
│
├── reports/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Technology Stack

### Backend

* Python
* Flask
* pandas
* NumPy
* scikit-learn
* joblib
* yfinance
* SHAP

### Frontend

* HTML5
* CSS3
* JavaScript
* Plotly.js

### Machine Learning

* Extra Trees
* Ridge Regression
* Random Forest
* Gradient Boosting
* Probability calibration
* Walk-forward validation
* SHAP explainability

### Data Source

Stock market data is retrieved through:

```text
Yahoo Finance
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd stock_predictor
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
```

### 3. Activate the environment

```powershell
.\venv\Scripts\activate
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root if your current configuration requires environment variables.

Example:

```env
HF_TOKEN=
```

Never commit the real `.env` file or private API tokens to GitHub.

The repository includes:

```text
.env.example
```

as a safe configuration template.

---

## ▶️ Running StockSense

Activate the virtual environment:

```powershell
.\venv\Scripts\activate
```

Then run:

```powershell
python app.py
```

Open the local Flask application in your browser.

---

## 🔌 API Endpoints

### Home

```text
GET /
```

Loads the StockSense dashboard.

### Prediction

```text
POST /api/predict
```

Generates:

* Market information
* Direction prediction
* Class probabilities
* Confidence
* Uncertainty
* Technical analysis
* Experimental forecast
* Historical chart data

### Validation

```text
GET /api/validate
```

Runs or returns model validation information through the validation service.

### Model Information

```text
GET /api/model-info
```

Returns production model information including:

* Model type
* Features
* Training information
* Calibration
* Validation metrics
* Baseline comparison

### Explainability

```text
POST /api/explain
```

Generates SHAP-based feature explanations for the production classifier.

---

## 📌 Example Prediction

A typical prediction may look like:

```text
Ticker: AAPL

Direction: UP

Probability UP:   55.46%
Probability DOWN: 44.54%

Classifier Confidence: 55.46%

Probability Spread: 10.91 percentage points

Uncertainty: MODERATE
```

This means the model currently favors the UP class, but the directional edge is relatively small.

It does not mean there is a 55.46% chance of making money.

---

## 🧪 Validation Methodology

StockSense uses chronological validation techniques to reduce information leakage from future observations.

The direction classifier is evaluated using expanding walk-forward validation.

Conceptually:

```text
Training Data
████████████

Test
    ██

Training Data
████████████████

Test
                ██

Training Data
████████████████████

Test
                    ██
```

Each validation stage trains on earlier observations and evaluates on later observations.

This better reflects how a model would operate when predicting future market observations.

---

## ⚠️ Current Limitations

StockSense is a portfolio/educational machine-learning project and has several limitations.

### Market limitations

* Financial markets are highly noisy.
* Historical relationships may not persist.
* Market regimes can change.
* News and macroeconomic events are not fully modeled.
* The current production classifier is focused on directional classification rather than expected returns.

### Model limitations

* Validation accuracy is only modestly above the majority baseline.
* ROC-AUC is relatively close to 0.5.
* Probability calibration improves probability metrics but does not guarantee accurate future probabilities.
* Model performance can vary across tickers and market regimes.
* The current production model was evaluated primarily on the project's available historical dataset.

### Forecast limitations

The experimental price forecasting models do not consistently outperform the naive baseline and should therefore not be treated as reliable future-price estimators.

---

## 🔮 Future Improvements

Potential future versions of StockSense could include:

* Multi-ticker model training
* Sector-aware models
* Market-regime detection
* News sentiment analysis
* Financial statement features
* Macroeconomic indicators
* Advanced time-series models
* Transformer-based forecasting
* Better probability calibration
* Calibration curves and reliability diagrams
* Automated model retraining
* Model drift monitoring
* Feature drift monitoring
* Portfolio optimization
* Backtesting
* Risk-adjusted performance metrics
* Docker deployment
* Cloud deployment
* Authentication and user portfolios

---

## 📚 Project Goals

StockSense was developed as a portfolio project to demonstrate practical experience with:

* Machine learning
* Feature engineering
* Financial data analysis
* Model evaluation
* Probability calibration
* Explainable AI
* Flask backend development
* REST-style API endpoints
* Interactive frontend development
* Data visualization
* Software architecture
* Production-oriented ML thinking

---

## 👨‍💻 Author

**Hanzala Khatri**

Software Engineering Student
FAST NUCES Karachi

---

## 📄 License

This project is intended for educational and portfolio purposes.

If this project is published publicly, add an appropriate open-source license such as MIT after deciding the licensing terms.

---

## ⚠️ Financial Disclaimer

StockSense is **not a financial advisor**.

The information, predictions, probabilities, technical signals, and forecasts generated by this application are provided for educational and demonstration purposes only.

Nothing produced by StockSense should be interpreted as financial advice, investment advice, a recommendation to buy or sell securities, or a guarantee of future market performance.

Always perform independent research and consult a qualified financial professional before making investment decisions.
