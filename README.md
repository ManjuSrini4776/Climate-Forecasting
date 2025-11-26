Time-Series Forecasting of Air Temperature (Jena Climate Dataset)
🌤️ Project Overview

This project focuses on building, evaluating, and interpreting time-series forecasting models to predict air temperature (T in °C) using the Jena Climate dataset, recorded at 10-minute intervals from Jan 10, 2009 to Dec 31, 2016.

Your primary objectives are:

✔️ Achieve high forecasting accuracy

✔️ Build robust and generalizable models

✔️ Ensure interpretability of predictions

✔️ Maintain full reproducibility of preprocessing, training, and evaluation

📂 Dataset

Jena Climate Dataset

Time range: 2009-01-10 → 2016-12-31

Sampling frequency: 10 minutes

Total observations: ~420,000

Features include:

Temperature (T in °C)

Pressure

Density

Wind speed

Air humidity

Other meteorological attributes

Target variable:

T (Air Temperature in Celsius)

🎯 Project Goals

Clean and preprocess high-frequency climate data

Explore temporal patterns (daily, seasonal, yearly)

Develop multiple forecasting models

Compare model performance using reliable error metrics

Interpret results and identify key drivers of temperature dynamics

🧪 Models Implemented

This project supports multiple forecasting approaches:

📌 Traditional Statistical Models

ARIMA

SARIMA

SARIMAX

Exponential Smoothing

📌 Machine Learning Models

Random Forest Regressor

XGBoost / LightGBM

Support Vector Regression (SVR)

📌 Deep Learning Models

LSTM

GRU

Bi-LSTM

1D CNN + LSTM Hybrid

Encoder–Decoder LSTM

Attention-based sequence models

Transformer model for long-range forecasting

⚙️ Pipeline Structure

The forecasting workflow includes:

1️⃣ Data Loading & Cleaning

Handling missing values

Converting timestamps

Resampling (10-min → Hourly/Daily if needed)

2️⃣ Feature Engineering

Lag features

Rolling statistics

Seasonal decomposition

Fourier terms for long-period seasonality

3️⃣ Train–Validation–Test Split

Chronological splits

Multi-step forecasting windows

4️⃣ Model Training

Hyperparameter tuning

Walk-forward validation

5️⃣ Evaluation Metrics

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

MAPE

R² Score

6️⃣ Model Interpretability

SHAP values

Attention heatmaps

Feature importance visualizations

📊 Expected Outputs

Forecast graphs

Error metrics comparison table

Model interpretability charts

Final best model selection

Reproducible Jupyter notebook(s)

🖥️ Tech Stack

Python

Pandas, NumPy

Matplotlib, Seaborn, Plotly

Scikit-learn

Statsmodels

TensorFlow / Keras / PyTorch

SHAP / Captum (interpretability)
