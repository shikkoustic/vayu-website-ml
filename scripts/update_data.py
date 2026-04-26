import pandas as pd
from datetime import datetime, timedelta
import requests
import ee
import os
import argparse

parser = argparse.ArgumentParser(description="Update Regional Delhi AQI and weather data.")
parser.add_argument("--days", type=int, default=7, help="Number of days of data to fetch (default: 7)")
args, unknown = parser.parse_known_args()

ee.Initialize(project="aqi-shikkoustic")

REGIONS = {
    "North_Delhi":   {"lat": 28.66, "lon": 77.13},
    "South_Delhi":   {"lat": 28.58, "lon": 77.22},
    "East_Delhi":    {"lat": 28.64, "lon": 77.31},
    "West_Delhi":    {"lat": 28.63, "lon": 77.07},
    "Central_Delhi": {"lat": 28.63, "lon": 77.22},
}

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_BASE_DIR, "data")

def get_gee_geometry(lat, lon, buffer_km=4):
    pt = ee.Geometry.Point([lon, lat])
    return pt.buffer(buffer_km * 1000).bounds()

def fetch_weather_and_aq(lat, lon, start_date, end_date):
    """Fetches 2-hourly weather and PM2.5 from Open-Meteo."""
    print(f"Fetching Open-Meteo data for {lat}, {lon}...")
    
    w_url = "https://api.open-meteo.com/v1/forecast"
    w_params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": "temperature_2m,dewpoint_2m,precipitation,surface_pressure,wind_speed_10m,shortwave_radiation",
        "timezone": "Asia/Kolkata"
    }
    w_res = requests.get(w_url, params=w_params).json()
    if "hourly" not in w_res:
        raise ValueError(f"Weather API Error: {w_res.get('reason', 'Unknown error')}")
    df_w = pd.DataFrame(w_res["hourly"])
    df_w["time"] = pd.to_datetime(df_w["time"])
    
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": "pm2_5",
        "timezone": "Asia/Kolkata"
    }
    aq_res = requests.get(aq_url, params=aq_params).json()
    if "hourly" not in aq_res:
        raise ValueError(f"Air Quality API Error: {aq_res.get('reason', 'Unknown error')}")
    df_aq = pd.DataFrame(aq_res["hourly"])
    df_aq["time"] = pd.to_datetime(df_aq["time"])
    
    df = pd.merge(df_w, df_aq, on="time")
    
    df.rename(columns={
        "temperature_2m": "temp_2m_C",
        "dewpoint_2m": "dewpoint_C",
        "precipitation": "precipitation_mm",
        "shortwave_radiation": "solar_radiation_W",
        "surface_pressure": "surface_pressure_hPa",
        "wind_speed_10m": "wind_speed_10m_kmh",
        "pm2_5": "PM2.5"
    }, inplace=True)
    
    df = df[df['time'].dt.hour % 2 == 0].reset_index(drop=True)
    
    current_time = pd.Timestamp.now()
    df = df[df['time'] <= current_time].reset_index(drop=True)
    
    return df

def fetch_satellite(geom, start_date, end_date):
    """Fetches AOD, NO2, CO, and SO2 from GEE."""
    print("Fetching GEE Satellite Data...")
    
    modis = ee.ImageCollection("MODIS/061/MCD19A2_GRANULES") \
        .filterBounds(geom).filterDate(start_date, end_date).select("Optical_Depth_055")
    
    s5p_no2 = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2") \
        .filterBounds(geom).filterDate(start_date, end_date).select("NO2_column_number_density")
    
    s5p_co = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CO") \
        .filterBounds(geom).filterDate(start_date, end_date).select("CO_column_number_density")
    
    s5p_so2 = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_SO2") \
        .filterBounds(geom).filterDate(start_date, end_date).select("SO2_column_number_density")

    def reduce_mean(col, band, scale):
        def wrap(img):
            mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=scale)
            return ee.Feature(None, {
                'date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
                band: mean.get(band)
            })
        return col.map(wrap).getInfo()['features']

    aod_feats = reduce_mean(modis, "Optical_Depth_055", 1000)
    no2_feats = reduce_mean(s5p_no2, "NO2_column_number_density", 3500)
    co_feats  = reduce_mean(s5p_co,  "CO_column_number_density", 3500)
    so2_feats = reduce_mean(s5p_so2, "SO2_column_number_density", 3500)

    data = {}
    for f in aod_feats:
        d = f['properties']['date']
        v = f['properties'].get('Optical_Depth_055')
        if v: data.setdefault(d, {})['AOD_055'] = v
    for f in no2_feats:
        d = f['properties']['date']
        v = f['properties'].get('NO2_column_number_density')
        if v: data.setdefault(d, {})['NO2_Density'] = v
    for f in co_feats:
        d = f['properties']['date']
        v = f['properties'].get('CO_column_number_density')
        if v: data.setdefault(d, {})['CO_Density'] = v
    for f in so2_feats:
        d = f['properties']['date']
        v = f['properties'].get('SO2_column_number_density')
        if v: data.setdefault(d, {})['SO2_Density'] = v

    df_sat = pd.DataFrame([{'date': k, **v} for k, v in data.items()])
    if not df_sat.empty:
        df_sat['date'] = pd.to_datetime(df_sat['date'])
    return df_sat

def update_region(name, info, days):
    print(f"\n--- Updating {name} ---")
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    df_new = fetch_weather_and_aq(info['lat'], info['lon'], start_str, end_str)
    
    geom = get_gee_geometry(info['lat'], info['lon'])
    df_sat = fetch_satellite(geom, start_str, (end_dt + timedelta(days=1)).strftime("%Y-%m-%d"))

    if not df_sat.empty:
        df_new['date_only'] = df_new['time'].dt.normalize()
        df_new = pd.merge(df_new, df_sat, left_on='date_only', right_on='date', how='left').drop(columns=['date_only', 'date'])
    
    file_path = os.path.join(DATA_DIR, f"{name}_Historical.csv")
    if os.path.exists(file_path):
        df_old = pd.read_csv(file_path)
        df_old['time'] = pd.to_datetime(df_old['time'])
        df_final = pd.concat([df_old, df_new]).drop_duplicates(subset=['time'], keep='last').sort_values('time')
    else:
        df_final = df_new.sort_values('time')

    df_final = df_final.ffill().bfill()
    df_final.to_csv(file_path, index=False)
    print(f"✅ Successfully updated {name} dataset. Total rows: {len(df_final)}")

if __name__ == "__main__":
    for region_name, info in REGIONS.items():
        try:
            update_region(region_name, info, args.days)
        except Exception as e:
            print(f"Failed to update {region_name}: {e}")