from flask import Flask, request, render_template, redirect, url_for, session, jsonify
import random
import smtplib
from email.message import EmailMessage
import mysql.connector
import os
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import google.generativeai as genai
from ml.predict import predict_next_day, load_regional_models, aqi_color, forecast_extended_aqi
from ml.preprocess import load_dataset, create_features
from ml.aqi import pm25_to_aqi, aqi_category, aqi_transition_message, get_color
import pandas as pd
import numpy as np
import requests
import uuid
import subprocess
from datetime import datetime as dt_now
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv()

CASHFREE_APP_ID = os.getenv("CASHFREE_APP_ID")
CASHFREE_SECRET_KEY = os.getenv("CASHFREE_SECRET_KEY")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
DB_PASSWORD = os.getenv("DB_PASSWORD")
FLASK_KEY = os.getenv("FLASK_KEY")
EMAIL_ADDRESS = "shiragh.4@gmail.com"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

app = Flask(__name__)
app.secret_key = FLASK_KEY

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password=DB_PASSWORD,
    database="authSystem"
)
cursor = db.cursor(dictionary=True)
otp_store = {}
reset_otp_store = {}

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        query = "SELECT * FROM users WHERE email=%s"
        cursor.execute(query, (email,))
        user = cursor.fetchone()
        if user and check_password_hash(user['password'], password):
            otp = random.randint(100000, 999999)
            otp_store[email] = otp
            msg = EmailMessage()
            msg.set_content(f"Your login OTP is: {otp}")
            msg['Subject'] = 'OTP Verification'
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = email
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                smtp.send_message(msg)
            return render_template('login.html', show_otp=True, email=email)
        return render_template('login.html', error="Either Email or Password is Wrong")
    return render_template('login.html')

@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    email = request.form['email']
    user_otp = request.form['otp']
    if email in otp_store and str(otp_store[email]) == user_otp:
        otp_store.pop(email)
        session['user_email'] = email
        query = "SELECT name FROM users WHERE email=%s"
        cursor.execute(query, (email,))
        user = cursor.fetchone()
        session['user_name'] = user['name'] if user else 'User'
        return redirect(url_for("dashboard"))
    return render_template('login.html', show_otp=True, email=email, error="Wrong OTP, Try Again")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email'].lower()
        password = request.form['password']
        if len(password) < 6:
            return render_template('register.html', error='Password must be at least 6 characters long')
        if not any(char.isdigit() for char in password):
            return render_template('register.html', error='Password must contain at least one number')
        check_query = "SELECT * FROM users WHERE email=%s"
        cursor.execute(check_query, (email,))
        existing_user = cursor.fetchone()
        if existing_user:
            return render_template('register.html', error='Email is already registered. Please use a different email or login.')
        hash_pass = generate_password_hash(password)
        query = "INSERT INTO users(name, email, password) VALUES (%s,%s,%s)"
        cursor.execute(query, (name, email, hash_pass))
        db.commit()
        return render_template('login.html', success="Registration successful! You can now login.")
    return render_template('register.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email'].lower()
        query = "SELECT * FROM users WHERE email=%s"
        cursor.execute(query, (email,))
        user = cursor.fetchone()
        if not user:
            return render_template('forgot-password.html', error='Email not found. Please register first.')
        otp = random.randint(100000, 999999)
        reset_otp_store[email] = otp
        msg = EmailMessage()
        msg.set_content(f"Your password reset OTP is: {otp}\n\nIf you didn't request this, please ignore this email.")
        msg['Subject'] = 'Password Reset OTP'
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = email
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                smtp.send_message(msg)
            return render_template('forgot-password.html', show_otp=True, email=email, success='OTP sent to your email')
        except:
            return render_template('forgot-password.html', error='Failed to send OTP. Please try again.')
    return render_template('forgot-password.html')

@app.route('/verify-reset-otp', methods=['POST'])
def verify_reset_otp():
    email = request.form['email']
    user_otp = request.form['otp']
    if email in reset_otp_store and str(reset_otp_store[email]) == user_otp:
        return render_template('forgot-password.html', show_reset=True, email=email)
    return render_template('forgot-password.html', show_otp=True, email=email, error='Invalid OTP. Please try again.')

@app.route('/reset-password', methods=['POST'])
def reset_password():
    email = request.form['email']
    new_password = request.form['new_password']
    confirm_password = request.form['confirm_password']
    if len(new_password) < 6:
        return render_template('forgot-password.html', show_reset=True, email=email, error='Password must be at least 6 characters long')
    if not any(char.isdigit() for char in new_password):
        return render_template('forgot-password.html', show_reset=True, email=email, error='Password must contain at least one number')
    if new_password != confirm_password:
        return render_template('forgot-password.html', show_reset=True, email=email, error='Passwords do not match')
    hash_pass = generate_password_hash(new_password)
    query = "UPDATE users SET password=%s WHERE email=%s"
    cursor.execute(query, (hash_pass, email))
    db.commit()
    if email in reset_otp_store:
        reset_otp_store.pop(email)
    return render_template('login.html', success='Password reset successful! Please login with your new password.')

@app.route('/dashboard')
def dashboard():
    if 'user_email' not in session:
        return redirect(url_for("login"))
    region = request.args.get("region", "North_Delhi")
    try:
        prediction_data = predict_next_day(region)
    except Exception as e:
        print(f"Prediction error for dashboard ({region}):", e)
        prediction_data = None
    return render_template('dashboard.html', 
                         user_email=session.get('user_email'),
                         user_name=session.get('user_name', 'User'),
                         selected_region=region,
                         prediction=prediction_data)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route('/pollutants/pm25')
def pm25():
    return render_template('pollutant-info/pm25.html', user_email=session.get('user_email'))

@app.route('/pollutants/pm10')
def pm10():
    return render_template('pollutant-info/pm10.html', user_email=session.get('user_email'))

@app.route('/pollutants/no2')
def no2():
    return render_template('pollutant-info/no2.html', user_email=session.get('user_email'))

@app.route('/pollutants/so2')
def so2():
    return render_template('pollutant-info/so2.html', user_email=session.get('user_email'))

@app.route('/pollutants/co')
def co():
    return render_template('pollutant-info/co.html', user_email=session.get('user_email'))

@app.route('/pollutants/o3')
def o3():
    return render_template('pollutant-info/o3.html', user_email=session.get('user_email'))

@app.route('/airchat')
def airchat():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    embed = request.args.get('embed', 'false') == 'true'
    return render_template('air-quality/airchat.html',user_email=session.get('user_email'), embed=embed)

@app.route('/airchat/send', methods=['POST'])
def airchat_send():
    data = request.get_json()
    user_message = data.get('message', '')
    region = data.get('region', 'North_Delhi') 
    try:
        from ml.predict import predict_next_day, forecast_extended_aqi
        forecast_data = predict_next_day(region)
        extended_forecast = forecast_extended_aqi(region)
        cp = forecast_data.get('current_pollutants', {})
        best_model = forecast_data.get('best_model', 'XGBoost')
        ext_text = ""
        for m_key, m_data in extended_forecast.items():
            ext_text += f"  - {m_key.upper()}: " + ", ".join([f"{f['time_label']} ({f['aqi']} AQI)" for f in m_data]) + "\n"
        system_context = f"""
You are AirChat, an intelligent AI assistant for the 'Vayu' Air Quality Intelligence Center. 
Your primary job is to answer questions about air quality, pollution, and explain the current forecasts.
Keep your answers concise, friendly, and easy to read (1-3 short paragraphs). Do NOT use heavy markdown formatting.

Current Context for {region.replace('_', ' ')}:
- Current AQI: {cp.get('aqi', 'N/A')} ({cp.get('category', 'N/A')})
- PM2.5 Level: {cp.get('pm25', 'N/A')} µg/m³
- 24h Trend: {cp.get('trend', 'N/A')} (positive means worsening, negative means improving)

Models' Prediction for NEXT 2 HOURS:
- XGBoost: {forecast_data.get('xgb', {}).get('aqi')}
- LightGBM: {forecast_data.get('lgb', {}).get('aqi')}
- CatBoost: {forecast_data.get('cat', {}).get('aqi')}
- Best Performing Model Right Now: {best_model}

12-HOUR EXTENDED OUTLOOK (Predicted AQI for upcoming hours):
{ext_text}

Instructions:
If the user asks why the air quality is worsening or improving, act as an expert data scientist. 
Explain the relationship between weather, time of day (like evening inversions or traffic), and the model's predictions. 
If they ask for the forecast, give them the numbers from the context above.
If the query is NOT related to air quality, politely decline.

User Query: {user_message}
"""
        response = model.generate_content(system_context)
        return {"reply": response.text}
    except Exception as e:
        print("AirChat LLM Error:", e)
        return {"reply": "Sorry, my AI brain is experiencing a temporary glitch. Please try again in a moment!"}

@app.route('/about')
def about():
    from ml.predict import predict_next_day
    try:
        next_day = predict_next_day("North_Delhi")
        weather = (next_day or {}).get("weather", {})
    except:
        weather = {}
    return render_template(
        'about.html',
        bg_image=weather.get("bg_image", "clear_day.png"),
        time_period=weather.get("time_period", "afternoon")
    )

@app.route('/pay')
def pay():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    return render_template('pay.html')

@app.route('/create-payment-order', methods=['POST'])
def create_payment_order():
    if 'user_email' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    amount = request.json.get('amount')
    if not amount or float(amount) <= 0:
        return jsonify({"error": "Invalid amount"}), 400
    order_id = "vayu_" + str(uuid.uuid4().hex)[:16]
    url = "https://sandbox.cashfree.com/pg/orders"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-version": "2023-08-01",
        "x-client-id": CASHFREE_APP_ID,
        "x-client-secret": CASHFREE_SECRET_KEY
    }
    customer_email = session.get('user_email', 'anonymous@vayu.com')
    customer_phone = "9999999999"  
    payload = {
        "customer_details": {
            "customer_id": "cust_" + str(uuid.uuid4().hex)[:8],
            "customer_email": customer_email,
            "customer_phone": customer_phone 
        },
        "order_amount": float(amount),
        "order_currency": "INR",
        "order_id": order_id,
        "order_meta": {
            "return_url": request.host_url.rstrip('/') + url_for('payment_return') + "?order_id={order_id}"
        }
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        res_data = response.json()
        if response.status_code == 200:
            return jsonify({"payment_session_id": res_data.get("payment_session_id")})
        else:
            return jsonify({"error": res_data.get("message", "Payment creation failed")}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/payment-return')
def payment_return():
    order_id = request.args.get('order_id')
    return render_template('payment_success.html', order_id=order_id)

def predict_for_date(date, region="North_Delhi"):
    from ml.predict import load_regional_models, aqi_color
    from ml.preprocess import load_dataset, create_features
    try:
        df = load_dataset(region)
    except:
        return None
    df['date'] = pd.to_datetime(df['date'])
    target_date = pd.Timestamp(date).date()
    day_rows = df[df['date'].dt.date == target_date]
    if day_rows.empty:
        return None
    row = day_rows.iloc[-1:]
    pm25_actual = row.iloc[0]['PM2.5']
    actual_aqi = round(pm25_to_aqi(pm25_actual))
    try:
        xgb, lgb, cat, gru, rf = load_regional_models(region)
        xgb_model, xgb_features, _ = xgb
        lgb_model, lgb_features, _ = lgb
        cat_model, cat_features, _ = cat
        gru_model, gru_features, gru_scaler, _ = gru
        rf_model, rf_features, _ = rf
        df_feat = create_features(df.copy())
        df_feat = df_feat.dropna().reset_index(drop=True)
        col_mapping = {
            "temp_2m_C": "temp_2m", "temp_2m_K": "temp_2m",
            "dewpoint_C": "dewpoint", "dewpoint_K": "dewpoint",
            "precipitation_mm": "precipitation", "precipitation_m": "precipitation",
            "solar_radiation_W": "solar_radiation", "solar_radiation_Jm2": "solar_radiation",
            "surface_pressure_hPa": "pressure", "surface_pressure_Pa": "pressure",
            "wind_speed_10m_kmh": "wind_speed", "wind_speed_10m": "wind_speed"
        }
        df_feat = df_feat.rename(columns={k: v for k, v in col_mapping.items() if k in df_feat.columns})
        target_dt = row.iloc[0]['date']
        feat_row = df_feat[df_feat['date'] == target_dt]
        if not feat_row.empty:
            r = feat_row.iloc[0]
            def pred(mod_name, mod, feats):
                if not mod or feats is None: return actual_aqi
                if mod_name == "rf":
                    X = r[feats].values.reshape(1, -1)
                    pred_val = mod.predict(X)
                    if isinstance(pred_val, np.ndarray): pred_val = pred_val.flatten()[0]
                else:
                    if mod_name == "gru": 
                        idx = df_feat.index[df_feat['date'] == target_dt][0]
                        last_6 = df_feat.iloc[max(0, idx-5):idx+1]
                        if len(last_6) < 6:
                            pad = [last_6.iloc[0:1]] * (6 - len(last_6))
                            last_6 = pd.concat(pad + [last_6])
                        X_scaled = gru_scaler.transform(last_6[feats])
                        X = np.expand_dims(X_scaled, axis=0) # shape (1, 6, hidden)
                    else: 
                        X = r[feats].values.reshape(1, -1)
                    pred_val = mod.predict(X)
                    if isinstance(pred_val, np.ndarray): pred_val = pred_val.flatten()[0]
                return round(pm25_to_aqi(float(np.expm1(pred_val))))
            xgb_aqi = pred("xgb", xgb_model, xgb_features)
            lgb_aqi = pred("lgb", lgb_model, lgb_features)
            cat_aqi = pred("cat", cat_model, cat_features)
            gru_aqi = pred("gru", gru_model, gru_features)
            rf_aqi = pred("rf", rf_model, rf_features)
        else:
            xgb_aqi = lgb_aqi = cat_aqi = gru_aqi = rf_aqi = actual_aqi
    except Exception as e:
        print(f"Model prediction error for {region}:", e)
        xgb_aqi = lgb_aqi = cat_aqi = gru_aqi = rf_aqi = actual_aqi
    return {
        "actual_aqi":      actual_aqi,
        "xgb_aqi":         xgb_aqi,
        "lgb_aqi":         lgb_aqi,
        "cat_aqi":         cat_aqi,
        "gru_aqi":         gru_aqi,
        "rf_aqi":          rf_aqi,
        "actual_category": aqi_category(actual_aqi),
        "xgb_category":    aqi_category(xgb_aqi),
        "lgb_category":    aqi_category(lgb_aqi),
        "cat_category":    aqi_category(cat_aqi),
        "gru_category":    aqi_category(gru_aqi),
        "rf_category":     aqi_category(rf_aqi),
        "actual_color":    get_color(actual_aqi),
        "xgb_color":       get_color(xgb_aqi),
        "lgb_color":       get_color(lgb_aqi),
        "cat_color":       get_color(cat_aqi),
        "gru_color":       get_color(gru_aqi),
        "rf_color":        get_color(rf_aqi),
        "transition_message": aqi_transition_message(actual_aqi)
    }

@app.route("/aqi")
def aqi_show():
    region = request.args.get("region", "North_Delhi")
    try:
        next_day = predict_next_day(region)
    except Exception as e:
        print(f"Next-day prediction error for {region}:", e)
        next_day = None
    date = request.args.get("date")
    if date:
        prediction = predict_for_date(pd.Timestamp(date), region)
        if not prediction:
            return "No data found for that date. Please try a different date."
        return render_template(
            "air-quality/aqi_show.html",
            selected_date=date,
            selected_region=region,
            actual_aqi=prediction["actual_aqi"],
            xgb_aqi=prediction["xgb_aqi"],
            lgb_aqi=prediction["lgb_aqi"],
            cat_aqi=prediction["cat_aqi"],
            gru_aqi=prediction["gru_aqi"],
            rf_aqi=prediction["rf_aqi"],
            actual_category=prediction["actual_category"],
            xgb_category=prediction["xgb_category"],
            lgb_category=prediction["lgb_category"],
            cat_category=prediction["cat_category"],
            gru_category=prediction["gru_category"],
            rf_category=prediction["rf_category"],
            actual_color=prediction["actual_color"],
            xgb_color=prediction["xgb_color"],
            lgb_color=prediction["lgb_color"],
            cat_color=prediction["cat_color"],
            gru_color=prediction["gru_color"],
            rf_color=prediction["rf_color"],
            transition_message=prediction.get("transition_message"),
            show_result=True,
            next_day=next_day
        )
    try:
        extended_forecast = forecast_extended_aqi(region)
    except Exception as e:
        print(f"Extended forecast error: {e}")
        extended_forecast = {}
    all_regions = ["North_Delhi", "South_Delhi", "East_Delhi", "West_Delhi", "Central_Delhi"]
    map_data = {}
    for r in all_regions:
        try:
            raw_df = load_dataset(r)
            last_pm25 = max(0, float(raw_df["PM2.5"].iloc[-1]))
            r_aqi = round(pm25_to_aqi(last_pm25))
            map_data[r] = {
                "aqi": r_aqi,
                "category": aqi_category(r_aqi),
                "color": aqi_color(r_aqi),
                "pm25": round(last_pm25, 1),
            }
        except Exception:
            map_data[r] = {"aqi": 0, "category": "N/A", "color": "#666", "pm25": 0}
    weather = (next_day or {}).get("weather", {})
    return render_template(
        "air-quality/aqi_show.html",
        selected_region=region,
        show_result=False,
        next_day=next_day,
        extended_forecast=extended_forecast,
        map_data=map_data,
        bg_image=weather.get("bg_image", "clear_day.png"),
        time_period=weather.get("time_period", "afternoon"),
    )

@app.route("/aqi-range", methods=["POST"])
def aqi_range():
    try:
        data = request.get_json()
        region = data.get("region", "North_Delhi")
        range_type = data.get("range_type", "custom")
        raw_df = load_dataset(region)
        raw_df['date'] = pd.to_datetime(raw_df['date'])
        max_date = raw_df['date'].max()
        if range_type == "24h":
            start = max_date - pd.Timedelta(hours=24)
            end = max_date
        elif range_type == "7d":
            start = max_date - pd.Timedelta(days=7)
            end = max_date
        elif range_type == "15d":
            start = max_date - pd.Timedelta(days=15)
            end = max_date
        elif range_type == "30d":
            start = max_date - pd.Timedelta(days=30)
            end = max_date
        else:
            start = pd.to_datetime(data["start_date"])
            end = pd.to_datetime(data["end_date"]) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        df = create_features(raw_df)
        df = df.dropna().reset_index(drop=True)
        col_mapping = {
            "temp_2m_C": "temp_2m", "temp_2m_K": "temp_2m",
            "dewpoint_C": "dewpoint", "dewpoint_K": "dewpoint",
            "precipitation_mm": "precipitation", "precipitation_m": "precipitation",
            "solar_radiation_W": "solar_radiation", "solar_radiation_Jm2": "solar_radiation",
            "surface_pressure_hPa": "pressure", "surface_pressure_Pa": "pressure",
            "wind_speed_10m_kmh": "wind_speed", "wind_speed_10m": "wind_speed"
        }
        df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
        from ml.predict import load_regional_models
        xgb, lgb, cat, gru, rf = load_regional_models(region)
        xgb_model, xgb_features, _ = xgb
        lgb_model, lgb_features, _ = lgb
        cat_model, cat_features, _ = cat
        gru_model, gru_features, gru_scaler, _ = gru
        rf_model, rf_features, _ = rf
        filtered = df[(df['date'] >= start) & (df['date'] <= end)].sort_values("date")
        res_dicts = []
        for idx, row in filtered.iterrows():
            actual_aqi = pm25_to_aqi(row["PM2.5"])
            def pred(mod_name, mod, feats):
                if not mod or feats is None: return actual_aqi
                if mod_name == "rf":
                    X = row[feats].values.reshape(1, -1)
                    pred_val = mod.predict(X)
                    if isinstance(pred_val, np.ndarray): pred_val = pred_val.flatten()[0]
                else:
                    if mod_name == "gru": 
                        idx = df.index[df['date'] == row["date"]][0]
                        last_6 = df.iloc[max(0, idx-5):idx+1]
                        if len(last_6) < 6:
                            pad = [last_6.iloc[0:1]] * (6 - len(last_6))
                            last_6 = pd.concat(pad + [last_6])
                        X_scaled = gru_scaler.transform(last_6[feats])
                        X = np.expand_dims(X_scaled, axis=0) # shape (1, 6, hidden)
                    else: 
                        X = row[feats].values.reshape(1, -1)
                    pred_val = mod.predict(X)
                    if isinstance(pred_val, np.ndarray): pred_val = pred_val.flatten()[0]
                return pm25_to_aqi(float(np.expm1(pred_val)))
            res_dicts.append({
                "date": row["date"],
                "actual": actual_aqi,
                "xgb": pred("xgb", xgb_model, xgb_features),
                "lgb": pred("lgb", lgb_model, lgb_features),
                "cat": pred("cat", cat_model, cat_features),
                "gru": pred("gru", gru_model, gru_features),
                "rf": pred("rf", rf_model, rf_features)
            })
        res_df = pd.DataFrame(res_dicts)
        dates, actual_values, xgb_values, lgb_values, cat_values, gru_values, rf_values = [], [], [], [], [], [], []
        if not res_df.empty:
            unique_days = res_df['date'].dt.date.nunique()
            show_hourly = (range_type == "24h") or (unique_days <= 1)
            if not show_hourly:
                res_df['day'] = res_df['date'].dt.date
                daily = res_df.groupby('day').mean(numeric_only=True).reset_index()
                for _, r in daily.iterrows():
                    dates.append(r['day'].strftime('%Y-%m-%d'))
                    actual_values.append(round(r['actual']))
                    xgb_values.append(round(r['xgb']))
                    lgb_values.append(round(r['lgb']))
                    cat_values.append(round(r['cat']))
                    gru_values.append(round(r['gru']))
                    rf_values.append(round(r['rf']))
            else:
                for _, r in res_df.iterrows():
                    dates.append(r['date'].strftime('%I:%M %p').lstrip('0'))
                    actual_values.append(round(r['actual']))
                    xgb_values.append(round(r['xgb']))
                    lgb_values.append(round(r['lgb']))
                    cat_values.append(round(r['cat']))
                    gru_values.append(round(r['gru']))
                    rf_values.append(round(r['rf']))
        return jsonify({
            "dates": dates,
            "actual": actual_values,
            "xgb": xgb_values,
            "lgb": lgb_values,
            "cat": cat_values,
            "gru": gru_values,
            "rf": rf_values
        })
    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    _BASE = os.path.dirname(os.path.abspath(__file__))
    _DATA_DIR = os.path.join(_BASE, "data")
    _SCRIPT = os.path.join(_BASE, "scripts", "update_data.py")
    _VENV_PYTHON = os.path.join(_BASE, "venv", "bin", "python")

    def run_data_update():
        print(f"\n🔄 [{dt_now.now().strftime('%I:%M %p')}] Auto-refreshing data...")
        try:
            result = subprocess.run(
                [_VENV_PYTHON, _SCRIPT, "--days", "2"],
                cwd=_BASE, capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                print(f"✅ [{dt_now.now().strftime('%I:%M %p')}] Data refresh complete.")
            else:
                print(f"⚠️ Data refresh failed: {result.stderr[-200:] if result.stderr else 'Unknown'}")
        except subprocess.TimeoutExpired:
            print("⚠️ Data refresh timed out (>120s). Will retry next cycle.")
        except Exception as e:
            print(f"⚠️ Data refresh error: {e}")

    def check_and_fetch_on_startup():
        for region in ["North_Delhi", "South_Delhi", "East_Delhi", "West_Delhi", "Central_Delhi"]:
            csv_path = os.path.join(_DATA_DIR, f"{region}_Historical.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    last_time = pd.to_datetime(df["time"].iloc[-1])
                    age_hours = (pd.Timestamp.now() - last_time).total_seconds() / 3600
                    if age_hours > 1.5:
                        print(f"📡 {region}: Data is {age_hours:.1f}h old — refreshing...")
                        run_data_update()
                        return  
                    else:
                        print(f"✅ {region}: Data is fresh ({age_hours:.1f}h old)")
                except Exception as e:
                    print(f"⚠️ Could not check {region}: {e}")
            else:
                print(f"📡 {region}: No data file found — fetching...")
                run_data_update()
                return

    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
        check_and_fetch_on_startup()
        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(
            run_data_update,
            'cron',
            hour='0,2,4,6,8,10,12,14,16,18,20,22',
            minute=5,
            id='auto_data_refresh',
            name='AQI Data Auto-Refresh',
            misfire_grace_time=3600,  
        )
        scheduler.start()
        print("\n⏰ Scheduler active — data will auto-refresh every 2 hours at :05")