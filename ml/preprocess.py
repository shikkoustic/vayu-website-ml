import pandas as pd
import numpy as np
import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_BASE_DIR, "data")

def load_dataset(region_name="North_Delhi"):
    """
    Loads the historical dataset for a specific region.
    """
    file_path = os.path.join(DATA_DIR, f"{region_name}_Historical.csv")
    if not os.path.exists(file_path):
        global_path = os.path.join(DATA_DIR, "Delhi_Daily_Final_Clean.csv")
        if os.path.exists(global_path):
            df = pd.read_csv(global_path)
            df["date"] = pd.to_datetime(df["date"], format="mixed")
            return df.sort_values("date").reset_index(drop=True)
        raise FileNotFoundError(f"No dataset found for region: {region_name}")

    df = pd.read_csv(file_path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.rename(columns={"time": "date"})
    else:
        df["date"] = pd.to_datetime(df["date"], format="mixed")
    
    return df.sort_values("date").reset_index(drop=True)

def create_features(df):
    """
    Generates lag, rolling, and time-based features.
    Handles both daily and 2-hourly data.
    """
    time_diff = df["date"].diff().median()
    is_2hourly = time_diff < pd.Timedelta(hours=3)

    if is_2hourly:
        df["pm25_lag1"] = df["PM2.5"].shift(1)   
        df["pm25_lag6"] = df["PM2.5"].shift(6)   
        df["pm25_lag12"] = df["PM2.5"].shift(12) 
        df["pm25_lag24"] = df["PM2.5"].shift(24) 
        df["pm25_lag84"] = df["PM2.5"].shift(84) 
        
        df["pm25_delta1"] = df["PM2.5"].diff(1)
        df["pm25_delta6"] = df["PM2.5"].diff(6)
        df["pm25_accel"] = df["pm25_delta1"].diff(1)
    else:
        df["pm25_lag1"] = df["PM2.5"].shift(1)
        df["pm25_lag2"] = df["PM2.5"].shift(2)
        df["pm25_lag3"] = df["PM2.5"].shift(3)
        df["pm25_lag7"] = df["PM2.5"].shift(7)
        df["pm25_delta1"] = df["PM2.5"].diff(1)
        df["pm25_accel"] = df["pm25_delta1"].diff(1)

    if is_2hourly:
        df["pm25_roll12"] = df["PM2.5"].rolling(12).mean()
        df["pm25_roll36"] = df["PM2.5"].rolling(36).mean()
        df["pm25_roll_std"] = df["PM2.5"].rolling(12).std()
        
        if "AOD_055" in df.columns: df["aod_roll12"] = df["AOD_055"].rolling(12).mean()
        if "NO2_Density" in df.columns: df["no2_roll12"] = df["NO2_Density"].rolling(12).mean()
        if "CO_Density" in df.columns: df["co_roll12"] = df["CO_Density"].rolling(12).mean()
        if "SO2_Density" in df.columns: df["so2_roll12"] = df["SO2_Density"].rolling(12).mean()
    
    df["pm25_roll_short"] = df["PM2.5"].rolling(12 if is_2hourly else 3).mean()
    df["pm25_roll_long"] = df["PM2.5"].rolling(36 if is_2hourly else 9).mean()

    for col in ["AOD_055", "NO2_Density", "CO_Density", "SO2_Density"]:
        if col in df.columns:
            df[f"{col.split('_')[0].lower()}_lag1"] = df[col].shift(1)

    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["day_of_week"] = df["date"].dt.dayofweek
    df["hour"] = df["date"].dt.hour

    df["is_winter"] = df["month"].isin([11, 12, 1, 2]).astype(int)
    df["is_monsoon"] = df["month"].isin([7, 8, 9]).astype(int)
    df["is_summer"] = df["month"].isin([4, 5, 6]).astype(int)
    df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df["high_risk_month"] = df["month"].isin([10, 11, 12, 1]).astype(int)

    temp_col = "temp_2m_C" if "temp_2m_C" in df.columns else "temp_2m"
    wind_col = "wind_speed_10m_kmh" if "wind_speed_10m_kmh" in df.columns else "wind_speed"
    
    if temp_col in df.columns and wind_col in df.columns:
        df["calm_wind"] = (df[wind_col] < 5).astype(int)
        df["hot_dry"] = ((df[temp_col] > 35) & (df[wind_col] > 10)).astype(int)
        
    df["in_spike_event"] = (df["PM2.5"].diff(1) > 20).astype(int)

    df["target"] = np.log1p(df["PM2.5"].shift(-1))

    return df

def get_training_data(region_name="North_Delhi"):
    """
    Returns X, y, and feature column list for a specific region.
    Dynamically adapts to available columns in the regional CSV.
    """
    df = load_dataset(region_name)
    df = create_features(df)
    df = df.dropna().reset_index(drop=True)

    col_mapping = {
        "temp_2m_C": "temp_2m",
        "temp_2m_K": "temp_2m",
        "dewpoint_C": "dewpoint",
        "dewpoint_K": "dewpoint",
        "precipitation_mm": "precipitation",
        "precipitation_m": "precipitation",
        "solar_radiation_W": "solar_radiation",
        "solar_radiation_Jm2": "solar_radiation",
        "surface_pressure_hPa": "pressure",
        "surface_pressure_Pa": "pressure",
        "wind_speed_10m_kmh": "wind_speed",
        "wind_speed_10m": "wind_speed"
    }
    
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})

    base_features = [
        "temp_2m", "dewpoint", "precipitation", "pressure", "wind_speed",
        "month", "day_of_year", "day_of_week", "hour", "is_winter", "is_monsoon"
    ]
    
    lag_features = [c for c in df.columns if "lag" in c]
    roll_features = [c for c in df.columns if "roll" in c]
    sat_features = []
    if "AOD_055" in df.columns: sat_features.append("AOD_055")
    if "NO2_Density" in df.columns: sat_features.append("NO2_Density")
    if "CO_Density" in df.columns: sat_features.append("CO_Density")
    if "SO2_Density" in df.columns: sat_features.append("SO2_Density")

    feature_columns = base_features + lag_features + roll_features + sat_features
    feature_columns = [c for c in feature_columns if c in df.columns]

    X = df[feature_columns]
    y = df["target"]

    return X, y, feature_columns