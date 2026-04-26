# 🌬️ Project Vayu: Air Quality Intelligence Center

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![ML](https://img.shields.io/badge/ML-XGBoost%20|%20LightGBM%20|%20CatBoost-orange.svg)](https://xgboost.readthedocs.io/)

**Project Vayu** is a professional, SaaS-grade Air Quality Intelligence platform designed to provide hyper-local AQI forecasts for Delhi. By fusing satellite-based data with advanced ensemble machine learning models, Vayu offers real-time insights and predictive analytics to help citizens make informed health decisions.

---

## ✨ Key Features

- 🎯 **Hyper-Local Forecasting**: Accurate AQI predictions for 5 key regions: North, South, East, West, and Central Delhi.
- 🤖 **Multi-Model Intelligence**: Comparative analysis using **XGBoost**, **LightGBM**, **CatBoost**, and **Random Forest**.
- 🛰️ **Satellite Data Fusion**: Integrates GEE (Google Earth Engine) satellite imagery with ground-level meteorological data.
- 🌡️ **Weather-Aware Logic**: Dynamic dashboard backgrounds that adapt to current weather conditions and time of day.
- 💬 **AirChat Assistant**: A custom AI-powered assistant for answering air quality queries.
- 📊 **Historical Analytics**: Interactive 12-hour outlooks and trend analysis charts.
- 🏥 **Health Recommendations**: Real-time health assessments and actionable safety recommendations based on current AQI.

---

## 🚀 Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, CatBoost
- **Data Pipeline**: Open-Meteo API, Google Earth Engine (GEE)
- **Frontend**: Vanilla HTML5, CSS3 (Glassmorphism UI), JavaScript (ES6+), Chart.js
- **Scheduling**: APScheduler for real-time data synchronization

---

## 📂 Project Structure

```text
Build2/
├── app.py              # Main Flask application & background scheduler
├── ml/                 # Machine learning pipeline
│   ├── predict.py      # Inference logic & regional model blending
│   ├── preprocess.py   # Feature engineering & scaling
│   └── aqi.py          # Indian AQI calculation & color mapping
├── scripts/            # Data orchestration scripts
│   └── update_data.py  # Live weather & satellite data fetching
├── static/             # Assets (CSS, JS, Images)
├── templates/          # Jinja2 HTML components
└── saved_models/       # (Local only) Trained model binaries
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shikkoustic/vayu-website-ml.git
   cd vayu-website-ml
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file in the root directory and add your API keys (Google Earth Engine, etc.):
   ```text
   GEE_PROJECT=your_project_id
   ```

5. **Run the application**:
   ```bash
   python app.py
   ```

---

## 🛡️ License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Acknowledgments

*   Data provided by Open-Meteo and Google Earth Engine.
*   Designed with ❤️ for a cleaner, breathable future.

---
*Note: Due to file size constraints (4.5GB), pre-trained model binaries in `saved_models/` are not included in the repository. Please ensure you have the models locally or run the training pipeline.*
