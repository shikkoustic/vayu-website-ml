import os
import pickle
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from ml.preprocess import load_dataset, create_features
from ml.aqi import pm25_to_aqi, aqi_category, aqi_transition_message

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_BASE_DIR, "saved_models")

_REGION_COORDS = {
    "North_Delhi":   {"lat": 28.66, "lon": 77.13},
    "South_Delhi":   {"lat": 28.58, "lon": 77.22},
    "East_Delhi":    {"lat": 28.64, "lon": 77.31},
    "West_Delhi":    {"lat": 28.63, "lon": 77.07},
    "Central_Delhi": {"lat": 28.63, "lon": 77.22},
}

_WMO_MAP = {
    0: ("☀️", "Clear"),
    1: ("🌤️", "Mostly Clear"),
    2: ("⛅", "Partly Cloudy"),
    3: ("☁️", "Overcast"),
    45: ("🌫️", "Fog"),
    48: ("🌫️", "Icy Fog"),
    51: ("🌦️", "Drizzle"),
    53: ("🌦️", "Drizzle"),
    55: ("🌧️", "Heavy Drizzle"),
    61: ("🌧️", "Rain"),
    63: ("🌧️", "Moderate Rain"),
    65: ("🌧️", "Heavy Rain"),
    71: ("🌨️", "Snow"),
    80: ("🌦️", "Showers"),
    95: ("⛈️", "Thunderstorm"),
    96: ("⛈️", "Thunderstorm"),
    99: ("⛈️", "Thunderstorm"),
}

def _wmo_to_weather(code, is_night=False):
    emoji, label = _WMO_MAP.get(code, ("🌤️", "Cloudy"))
    if is_night:
        if code == 0:
            emoji, label = "🌙", "Clear Night"
        elif code == 1:
            emoji, label = "🌙", "Mostly Clear Night"
        elif code == 2:
            emoji, label = "🌙☁️", "Partly Cloudy Night"
        elif code == 3:
            emoji, label = "☁️", "Overcast"
    return emoji, label

def fetch_hourly_forecast(region_name, hours=5):
    """Fetch next N hours of weather forecast from Open-Meteo."""
    coords = _REGION_COORDS.get(region_name, _REGION_COORDS["North_Delhi"])
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "hourly": "temperature_2m,weathercode,precipitation_probability,wind_speed_10m",
                "timezone": "Asia/Kolkata",
                "forecast_days": 2,
            },
            timeout=5,
        ).json()

        hourly = resp.get("hourly", {})
        times      = hourly.get("time", [])
        temps      = hourly.get("temperature_2m", [])
        codes      = hourly.get("weathercode", [])
        precip_pct = hourly.get("precipitation_probability", [])
        winds      = hourly.get("wind_speed_10m", [])

        now = datetime.now()
        forecast = []
        for i, t in enumerate(times):
            dt = datetime.fromisoformat(t)
            if dt <= now:
                continue
            is_night = dt.hour >= 20 or dt.hour < 6
            emoji, label = _wmo_to_weather(codes[i] if i < len(codes) else 0, is_night)
            forecast.append({
                "time":      dt.strftime("%I %p").lstrip("0"),
                "hour":      dt.hour,
                "temp":      round(temps[i]) if i < len(temps) else "—",
                "emoji":     emoji,
                "condition": label,
                "precip":    precip_pct[i] if i < len(precip_pct) else 0,
                "wind":      round(winds[i]) if i < len(winds) else 0,
            })
            if len(forecast) >= hours:
                break
        return forecast
    except Exception as e:
        print(f"  ⚠️ Could not fetch forecast: {e}")
        return []

def aqi_color(aqi):
    if aqi <= 50: return "#22c55e"
    elif aqi <= 100: return "#84cc16"
    elif aqi <= 200: return "#eab308"
    elif aqi <= 300: return "#f97316"
    elif aqi <= 400: return "#ef4444"
    else: return "#991b1b"

def load_regional_models(region_name="North_Delhi"):
    """
    Loads XGBoost, LightGBM, CatBoost, GRU, and NeuralProphet models for a specific region.
    """
    def safe_load(path):
        try:
            if not os.path.exists(path):
                return None
            if path.endswith(".h5"):
                from tensorflow.keras.models import load_model
                return load_model(path, compile=False)
            with open(path, "rb") as f: return pickle.load(f)
        except (ImportError, ModuleNotFoundError, Exception) as e:
            print(f"  ⚠️ Could not load {os.path.basename(path)}: {e}")
            return None

    xgb_model = safe_load(os.path.join(_MODELS_DIR, f"{region_name}_xgboost.pkl"))
    xgb_feats = safe_load(os.path.join(_MODELS_DIR, f"{region_name}_xgboost_features.pkl"))
    if xgb_feats is None:
        xgb_feats = safe_load(os.path.join(_MODELS_DIR, f"{region_name}_features.pkl"))
    xgb_metrics = safe_load(os.path.join(_MODELS_DIR, f"{region_name}_xgboost_metrics.pkl"))
    if xgb_metrics is None:
        xgb_metrics = safe_load(os.path.join(_MODELS_DIR, f"{region_name}_metrics.pkl"))
    xgb = (xgb_model, xgb_feats, xgb_metrics)

    lgb = (safe_load(os.path.join(_MODELS_DIR, f"{region_name}_lgbm.pkl")),
           safe_load(os.path.join(_MODELS_DIR, f"{region_name}_lgbm_features.pkl")),
           safe_load(os.path.join(_MODELS_DIR, f"{region_name}_lgbm_metrics.pkl")))

    cat = (safe_load(os.path.join(_MODELS_DIR, f"{region_name}_catboost.pkl")),
           safe_load(os.path.join(_MODELS_DIR, f"{region_name}_catboost_features.pkl")),
           safe_load(os.path.join(_MODELS_DIR, f"{region_name}_catboost_metrics.pkl")))

    gru = (safe_load(os.path.join(_MODELS_DIR, f"{region_name}_gru.h5")),
           safe_load(os.path.join(_MODELS_DIR, f"{region_name}_gru_features.pkl")),
           safe_load(os.path.join(_MODELS_DIR, f"{region_name}_gru_scaler.pkl")),
           safe_load(os.path.join(_MODELS_DIR, f"{region_name}_gru_metrics.pkl")))

    rf = (safe_load(os.path.join(_MODELS_DIR, f"{region_name}_rf.pkl")),
           safe_load(os.path.join(_MODELS_DIR, f"{region_name}_rf_features.pkl")),
           safe_load(os.path.join(_MODELS_DIR, f"{region_name}_rf_metrics.pkl")))

    return xgb, lgb, cat, gru, rf

def predict_next_day(region_name="North_Delhi"):
    """
    Generates AQI prediction for the next time step (2-hourly or daily).
    """
    xgb, lgb, cat, gru, rf = load_regional_models(region_name)
    xgb_model, xgb_features, xgb_metrics = xgb
    lgb_model, lgb_features, lgb_metrics = lgb
    cat_model, cat_features, cat_metrics = cat
    gru_model, gru_features, gru_scaler, gru_metrics = gru
    rf_model, rf_features, rf_metrics = rf

    if not xgb_model:
        return {"error": f"Models for {region_name} not found."}

    raw_df = load_dataset(region_name)
    last_row_date = pd.to_datetime(raw_df["date"].iloc[-1])
    
    time_diff = raw_df["date"].diff().median()
    prediction_date = last_row_date + time_diff

    df = create_features(raw_df)
    df = df.reset_index(drop=True)
    
    if df.empty:
        return {"error": "Insufficient data for prediction."}

    last_row = df.iloc[-1]

    col_mapping = {
        "temp_2m_C": "temp_2m", "temp_2m_K": "temp_2m", "temp": "temp_2m",
        "dewpoint_C": "dewpoint", "dewpoint_K": "dewpoint",
        "precipitation_mm": "precipitation", "precipitation_m": "precipitation", "precip": "precipitation",
        "solar_radiation_W": "solar_radiation", "solar_radiation_Jm2": "solar_radiation", "solar": "solar_radiation",
        "surface_pressure_hPa": "pressure", "surface_pressure_Pa": "pressure",
        "wind_speed_10m_kmh": "wind_speed", "wind_speed_10m": "wind_speed", "wind": "wind_speed"
    }
    df_pred = df.copy()
    df_pred = df_pred.rename(columns={k: v for k, v in col_mapping.items() if k in df_pred.columns})
    
    if len(df_pred) < 24:
        pad = [df_pred.iloc[0:1]] * (24 - len(df_pred))
        df_pred = pd.concat(pad + [df_pred])
    
    last_24_rows = df_pred.tail(24)
    row_feat = df_pred.iloc[-1]

    results = {}

    def resolve_model(mod, row_series):
        """Handle blended seasonal models (dicts with 'winter'/'summer' keys)."""
        if isinstance(mod, dict) and mod.get("blended"):
            current_month = prediction_date.month
            is_winter_month = current_month in [10, 11, 12, 1, 2, 3]
            return mod["winter"] if is_winter_month else mod["summer"]
        return mod

    for mod_name, mod, feats, metrics in [
        ("xgb", xgb_model, xgb_features, xgb_metrics),
        ("lgb", lgb_model, lgb_features, lgb_metrics),
        ("cat", cat_model, cat_features, cat_metrics),
        ("gru", gru_model, gru_features, gru_metrics),
        ("rf", rf_model, rf_features, rf_metrics)
    ]:
        if not mod:
            continue
        try:
            active_mod = resolve_model(mod, row_feat)

            X_input = []
            for f in feats:
                if f in row_feat:
                    X_input.append(row_feat[f])
                else:
                    reverse_map = {"temp": "temp_2m", "wind": "wind_speed", "precip": "precipitation", "solar": "solar_radiation"}
                    alt_f = reverse_map.get(f)
                    if alt_f and alt_f in row_feat:
                        X_input.append(row_feat[alt_f])
                    else:
                        if "pm25" in f: X_input.append(row_feat.get("PM2.5", 0))
                        else: X_input.append(0) 
            
            X_input = np.array(X_input)

            if mod_name == "gru":
                X_seq = []
                for idx, row in last_24_rows.iterrows():
                    r_feat = []
                    for f in feats:
                        if f in row: r_feat.append(row[f])
                        else:
                            alt_f = {"temp": "temp_2m", "wind": "wind_speed", "precip": "precipitation", "solar": "solar_radiation"}.get(f)
                            r_feat.append(row.get(alt_f, 0))
                    X_seq.append(r_feat)
                
                X_seq = np.array(X_seq)
                X_scaled = gru_scaler.transform(X_seq) if gru_scaler else X_seq
                X = np.expand_dims(X_scaled, axis=0) 
                log_pred = active_mod.predict(X)
            else:
                X = X_input.reshape(1, -1)
                log_pred = active_mod.predict(X)
                if isinstance(log_pred, np.ndarray):
                    log_pred = log_pred.flatten()[0]

            pm25 = float(np.expm1(log_pred))
            mae = metrics["mae"] if metrics else 10.0
            
            aqi = round(pm25_to_aqi(pm25))
            results[mod_name] = {
                "pm25": round(pm25, 2),
                "aqi": aqi,
                "low": round(max(pm25 - mae, 0), 2),
                "high": round(pm25 + mae, 2),
                "category": aqi_category(aqi),
                "color": aqi_color(aqi),
                "message": aqi_transition_message(aqi)
            }
        except Exception as e:
            results[mod_name] = {
                "error": str(e),
                "aqi": "Error",
                "color": "#666",
                "category": "N/A",
                "pm25": 0,
                "low": 0,
                "high": 0
            }

    temp = float(row_feat.get("temp_2m", 0))
    dewpoint = float(row_feat.get("dewpoint", 0))
    humidity = max(0, min(100, 100 - 5 * (temp - dewpoint)))

    hour = last_row_date.hour
    precip = float(row_feat.get("precipitation", 0))
    wind = float(row_feat.get("wind_speed", 0))
    solar = float(row_feat.get("solar_radiation", 0))
    raw_pm25_for_vis = max(0, float(row_feat.get("PM2.5", 0)))

    if 5 <= hour < 7:
        time_of_day = "morning"
        time_period = "sunrise"
    elif 7 <= hour < 12:
        time_of_day = "day"
        time_period = "morning"
    elif 12 <= hour < 18:
        time_of_day = "day"
        time_period = "afternoon"
    elif 18 <= hour < 20:
        time_of_day = "evening"
        time_period = "evening"
    else:
        time_of_day = "night"
        time_period = "night"

    if precip > 5 and wind > 20:
        cond, w_class, bg_image = "Thunderstorm", "thunderstorm", "thunderstorm.png"
    elif precip > 1:
        cond, w_class, bg_image = "Rain", "rain", "rain.png"
    elif time_period == "sunrise" or (18 <= hour <= 19 and humidity < 70):
        cond, w_class, bg_image = "Evening", "sunset", "sunset.png"
    elif humidity > 85 and temp < 28:
        cond, w_class, bg_image = "Mist", "mist", "mist.png"
    elif humidity > 70 and solar < 200:
        if time_of_day == "night":
            cond, w_class, bg_image = "Cloudy Night", "cloudy_night", "cloudy.png"
        else:
            cond, w_class, bg_image = "Cloudy", "cloudy", "cloudy.png"
    elif time_of_day == "night":
        cond, w_class, bg_image = "Clear Night", "clear_night", "clear_night.png"
    elif time_of_day == "evening":
        cond, w_class, bg_image = "Evening", "sunset", "sunset.png"
    else:
        cond, w_class, bg_image = "Clear", "clear_day", "clear_day.png"

    if humidity > 90:
        visibility = 1.0
    elif raw_pm25_for_vis > 300:
        visibility = 1.0
    elif humidity > 80 or raw_pm25_for_vis > 200:
        visibility = 2.0
    elif humidity > 70 or raw_pm25_for_vis > 150:
        visibility = 4.0
    elif humidity > 50 or raw_pm25_for_vis > 100:
        visibility = 6.0
    else:
        visibility = 10.0

    results["weather"] = {
        "temp": round(temp, 1),
        "wind_speed": round(wind, 1),
        "humidity": round(humidity, 1),
        "pressure": round(row_feat.get("pressure", 1000), 1),
        "precipitation": round(precip, 1),
        "visibility": round(visibility, 1),
        "condition": cond,
        "class": w_class,
        "bg_image": bg_image,
        "time_of_day": time_of_day,
        "time_period": time_period,
        "forecast_hours": fetch_hourly_forecast(region_name, hours=6)
    }

    c_pm25 = max(0, float(row_feat.get("PM2.5", 0)))
    c_aqi = round(pm25_to_aqi(c_pm25))

    prev_aqi = None
    try:
        raw_df_dated = raw_df.copy()
        raw_df_dated["date"] = pd.to_datetime(raw_df_dated["date"])
        yesterday = last_row_date - pd.Timedelta(hours=24)
        prev_rows = raw_df_dated[raw_df_dated["date"] <= yesterday]
        if not prev_rows.empty:
            prev_pm25 = max(0, float(prev_rows.iloc[-1].get("PM2.5", 0)))
            prev_aqi = round(pm25_to_aqi(prev_pm25))
    except Exception:
        prev_aqi = None
    
    results["current_pollutants"] = {
        "pm25": round(c_pm25, 1),
        "aqi": c_aqi,
        "category": aqi_category(c_aqi),
        "color": aqi_color(c_aqi),
        "no2": max(0, round(row_feat.get("NO2_Density", 0) * 10000, 1)), 
        "co": max(0, round(row_feat.get("CO_Density", 0) * 100, 1)),
        "so2": max(0, round(row_feat.get("SO2_Density", 0) * 100000, 1)),
        "prev_aqi": prev_aqi,
        "trend": (c_aqi - prev_aqi) if prev_aqi is not None else None
    }

    model_maes = {
        "xgb": (xgb_metrics or {}).get("mae", float("inf")),
        "lgb": (lgb_metrics or {}).get("mae", float("inf")),
        "cat": (cat_metrics or {}).get("mae", float("inf")),
        "gru": (gru_metrics or {}).get("mae", float("inf")),
        "rf":  (rf_metrics  or {}).get("mae", float("inf")),
    }
    valid_models = {k: v for k, v in model_maes.items()
                    if k in results and isinstance(results[k].get("aqi"), (int, float))}
    if valid_models:
        best_key = min(valid_models, key=lambda k: valid_models[k])
    else:
        best_key = next((k for k in ["xgb", "lgb", "cat", "rf", "gru"] if k in results), None)

    results["best_model"] = best_key
    results["best"] = results.get(best_key, {})

    results["prediction_date"] = prediction_date.strftime("%Y-%m-%d %I:%M %p")
    results["last_updated"] = last_row_date.strftime("%d %b %Y, %I:%M %p")
    results["region"] = region_name
    return results


def fetch_weather_forecast_12h(region_name):
    """Fetch 12 hours of hourly weather from Open-Meteo for recursive forecasting."""
    coords = _REGION_COORDS.get(region_name, _REGION_COORDS["North_Delhi"])
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "hourly": "temperature_2m,dew_point_2m,precipitation,surface_pressure,wind_speed_10m",
                "timezone": "Asia/Kolkata",
                "forecast_days": 2,
            },
            timeout=8,
        ).json()
        hourly = resp.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        dews = hourly.get("dew_point_2m", [])
        precips = hourly.get("precipitation", [])
        pressures = hourly.get("surface_pressure", [])
        winds = hourly.get("wind_speed_10m", [])

        now = datetime.now()
        forecast = []
        for i, t in enumerate(times):
            dt = datetime.fromisoformat(t)
            if dt <= now:
                continue
            forecast.append({
                "dt": dt,
                "temp_2m": temps[i] if i < len(temps) else 30.0,
                "dewpoint": dews[i] if i < len(dews) else 20.0,
                "precipitation": precips[i] if i < len(precips) else 0.0,
                "pressure": pressures[i] if i < len(pressures) else 1013.0,
                "wind_speed": winds[i] if i < len(winds) else 5.0,
            })
            if len(forecast) >= 14:
                break
        return forecast
    except Exception as e:
        print(f"  ⚠️ Could not fetch 12h weather forecast: {e}")
        return []


def forecast_extended_aqi(region_name="North_Delhi"):
    """
    Recursive 12-hour AQI forecast using the best model.
    Returns predictions at +4h, +6h, +8h, +10h, +12h (5 bars).
    The +2h prediction is already shown in the model cards.
    """
    from ml.preprocess import load_dataset, create_features
    from ml.aqi import pm25_to_aqi, aqi_category

    xgb, lgb, cat, gru, rf = load_regional_models(region_name)
    model_info = {
        "xgb": (xgb[0], xgb[1], xgb[2]),
        "lgb": (lgb[0], lgb[1], lgb[2]),
        "cat": (cat[0], cat[1], cat[2]),
        "rf":  (rf[0],  rf[1],  rf[2]),
    }

    all_forecasts = {}
    
    raw_df = load_dataset(region_name)
    last_row_date = pd.to_datetime(raw_df["date"].iloc[-1])
    df = create_features(raw_df)
    df = df.reset_index(drop=True)

    if df.empty:
        return {}

    col_mapping = {
        "temp_2m_C": "temp_2m", "temp_2m_K": "temp_2m", "temp": "temp_2m",
        "dewpoint_C": "dewpoint", "dewpoint_K": "dewpoint",
        "precipitation_mm": "precipitation", "precipitation_m": "precipitation",
        "surface_pressure_hPa": "pressure", "surface_pressure_Pa": "pressure",
        "wind_speed_10m_kmh": "wind_speed", "wind_speed_10m": "wind_speed",
    }
    df_pred = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
    row_feat_base = df_pred.iloc[-1].copy()

    weather_forecast = fetch_weather_forecast_12h(region_name)
    if not weather_forecast:
        return {}

    now = datetime.now()
    is_winter = now.month in [10, 11, 12, 1, 2, 3]

    for model_key, (mod, feats, metrics) in model_info.items():
        if not mod or not feats:
            continue
            
        if isinstance(mod, dict) and mod.get("blended"):
            active_model = mod["winter"] if is_winter else mod["summer"]
        else:
            active_model = mod

        row_feat = row_feat_base.copy()
        
        current_pm25 = float(row_feat.get("PM2.5", row_feat.get("pm25_lag1", 50)))
        prev_pm25 = float(row_feat.get("pm25_lag1", current_pm25))

        forecast_results = []

        for step in range(6):
            hour_offset = (step + 1) * 2
            forecast_time = last_row_date + pd.Timedelta(hours=hour_offset)

            wf = None
            for w in weather_forecast:
                if w["dt"] >= forecast_time - pd.Timedelta(minutes=90):
                    wf = w
                    break
            if wf is None and weather_forecast:
                wf = weather_forecast[-1]

            step_features = row_feat.copy()

            if wf:
                step_features["temp_2m"] = wf["temp_2m"]
                step_features["dewpoint"] = wf["dewpoint"]
                step_features["precipitation"] = wf["precipitation"]
                step_features["pressure"] = wf["pressure"]
                step_features["wind_speed"] = wf["wind_speed"]

            step_features["hour"] = forecast_time.hour
            step_features["month"] = forecast_time.month
            step_features["day_of_year"] = forecast_time.timetuple().tm_yday
            step_features["day_of_week"] = forecast_time.weekday()

            step_features["pm25_lag1"] = current_pm25
            if "pm25_lag12" in step_features:
                step_features["pm25_lag12"] = row_feat.get("pm25_lag12", current_pm25)
            if "pm25_lag24" in step_features:
                step_features["pm25_lag24"] = row_feat.get("pm25_lag24", current_pm25)
            if "pm25_roll_short" in step_features or "pm25_roll12" in step_features:
                roll_key = "pm25_roll_short" if "pm25_roll_short" in step_features else "pm25_roll12"
                old_roll = float(step_features.get(roll_key, current_pm25))
                step_features[roll_key] = (old_roll * 0.8) + (current_pm25 * 0.2)

            X_input = []
            reverse_map = {"temp": "temp_2m", "wind": "wind_speed",
                           "precip": "precipitation", "solar": "solar_radiation"}
            for f in feats:
                if f in step_features:
                    X_input.append(float(step_features[f]))
                else:
                    alt = reverse_map.get(f)
                    if alt and alt in step_features:
                        X_input.append(float(step_features[alt]))
                    elif "pm25" in f:
                        X_input.append(current_pm25)
                    else:
                        X_input.append(0)

            X = np.array(X_input).reshape(1, -1)

            try:
                log_pred = active_model.predict(X)
                if isinstance(log_pred, np.ndarray):
                    log_pred = log_pred.flatten()[0]
                predicted_pm25 = float(np.expm1(log_pred))
                predicted_pm25 = max(0, predicted_pm25)
            except Exception as e:
                print(f"  ⚠️ Extended forecast step {step} error: {e}")
                predicted_pm25 = current_pm25

            aqi = round(pm25_to_aqi(predicted_pm25))

            if hour_offset >= 4:
                forecast_results.append({
                    "hour_offset": hour_offset,
                    "time_label": forecast_time.strftime("%I %p").lstrip("0"),
                    "hour": forecast_time.hour,
                    "aqi": aqi,
                    "pm25": round(predicted_pm25, 1),
                    "category": aqi_category(aqi),
                    "color": aqi_color(aqi),
                })

            prev_pm25 = current_pm25
            current_pm25 = predicted_pm25

        all_forecasts[model_key] = forecast_results

    return all_forecasts


if __name__ == "__main__":
    print(predict_next_day("North_Delhi"))