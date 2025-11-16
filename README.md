This project builds a machine learning system that predicts the next 5-day price direction of the NASDAQ-100 ETF (QQQ) using:

Daily data from yfinance

Automated data cleaning + feature engineering

An Optuna-tuned XGBoost classifier

A live Streamlit dashboard that updates automatically

A daily GitHub Actions pipeline that fetches new data

The model uses volatility, momentum, trend, and candle-structure features to generate a probability that QQQ will move UP or DOWN over the next 5 days.

The Streamlit app displays:

A real-time UP/DOWN probability

Confidence interpretation

Latest feature values

Recent signal behavior (graph + streaks)

🛠 Installation
1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit app
streamlit run streamlit_app/app.py

🎯 Summary

This project demonstrates an end-to-end ML workflow combining:

Automated data ingestion

Machine learning model training

Feature engineering

Daily pipeline refresh

Real-time prediction dashboard

A simple and effective example of building a live MLOps-style system for financial forecasting.