# 📊 Economic Metrics Dashboard

A **Streamlit web dashboard** for tracking key **macroeconomic indicators** (FRED) and **financial market data** (Yahoo Finance), automatically refreshed **daily** using **GitHub Actions**.

The app displays the latest economic snapshot, historical trends, and pre-generated plots, all accessible through a single URL.

---

## ✨ Features

- 📈 Macroeconomic indicators from **FRED**
- 📉 Market data from **Yahoo Finance**
- 🖼️ Automatically generated charts
- 🕒 Daily scheduled data refresh (cron)
- ☁️ Deployed on **Streamlit Community Cloud**
- 🔐 Secure handling of API keys

---

## 🗂 Project Structure

.
├── app.py                  # Streamlit application
├── economics.py            # Daily data collection & plotting script
├── requirements.txt        # Python dependencies
├── dashboard_data/
│   ├── latest.json         # Latest snapshot used by the app
│   ├── history.csv         # Historical data (optional)
│   ├── metrics_macro.png   # Generated macro plots
│   └── metrics_markets.png # Generated market plots
└── .github/
└── workflows/
└── daily_refresh.yml  # GitHub Actions daily job

---

## 🚀 Live App

**Streamlit App URL:**  
_(add your deployed URL here)_

The app automatically reloads whenever new data is committed to the repository.

---

## 🔄 Daily Update Workflow

1. GitHub Actions runs on a daily schedule
2. `economics.py` fetches fresh data from FRED and Yahoo Finance
3. Output files are written to `dashboard_data/`
4. Updated files are committed back to the repository
5. Streamlit Cloud detects the commit and reloads the app

---

## 🔐 API Keys & Secrets

### FRED API Key (required)

#### GitHub Actions
Add a repository secret:
- **Name:** `FRED_API_KEY`
- **Value:** your FRED API key

#### Streamlit Cloud
Add the same key under **App → Settings → Secrets**:

+++
FRED_API_KEY = "your_key_here"
+++

## Run locally
# Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run data update once
export FRED_API_KEY="your_key_here"
python economics.py --out-dir dashboard_data

# Launch the Streamlit app
streamlit run app.py

## Configuration
- Schedule: configured via cron in .github/workflows/daily_refresh.yml (UTC)
- Data paths: all data is read from dashboard_data/ using relative paths

## Design Notes
- Streamlit app is read-only
- No background jobs run inside Streamlit
- All automation handled externally via GitHub Actions
- Data artifacts are version-controlled
