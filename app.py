import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# =========================
# CONFIG
# =========================
BERAS_FILE = "dataset/beras.xlsx"
MINYAK_FILE = "dataset/minyak.xlsx"


# =========================
# AMBIL LIST KOTA (SHEET)
# =========================
def get_cities():
    return pd.ExcelFile(BERAS_FILE).sheet_names


# =========================
# AMBIL LIST KOMODITAS
# =========================
def get_commodities(file, city):
    df = pd.read_excel(file, sheet_name=city)
    return df.iloc[:, 1].dropna().astype(str).unique().tolist()


# =========================
# AMBIL TIME SERIES DATA
# =========================
def get_series(file, city, commodity):
    df = pd.read_excel(file, sheet_name=city)

    row = df[df.iloc[:, 1].astype(str) == commodity]

    if row.empty:
        return []

    values = row.iloc[:, 2:].values.flatten()

    # cleaning
    cleaned = []
    for v in values:
        v = str(v).replace(",", "").replace("-", "")
        if v == "" or v.lower() == "nan":
            continue
        try:
            cleaned.append(float(v))
        except:
            cleaned.append(0)

    return cleaned


# =========================
# BUILD MODEL
# =========================
def build_model(series):
    if len(series) < 2:
        return None, None

    X = np.arange(1, len(series) + 1).reshape(-1, 1)
    y = np.array(series)

    model = LinearRegression()
    model.fit(X, y)

    pred = model.predict(X)

    metrics = {
        "mae": round(mean_absolute_error(y, pred), 2),
        "mse": round(mean_squared_error(y, pred), 2),
        "rmse": round(math.sqrt(mean_squared_error(y, pred)), 2),
        "r2": round(r2_score(y, pred), 4),
    }

    return model, metrics


# =========================
# UI
# =========================
st.set_page_config(page_title="PriceWise Jabar", layout="centered")

st.title("📊 PriceWise - Prediksi Harga & Kualitas Pangan Jabar")

# pilih kota
cities = get_cities()
city = st.selectbox("Pilih Kota", cities)

# pilih jenis data
dataset_type = st.selectbox("Pilih Komoditas", ["Beras", "Minyak"])

file = BERAS_FILE if dataset_type == "Beras" else MINYAK_FILE

# ambil kualitas dari dataset
commodities = get_commodities(file, city)
commodity = st.selectbox("Pilih Kualitas / Jenis", commodities)

# input waktu
tahun = st.number_input("Tahun", 2024, 2035, 2026)
bulan = st.number_input("Bulan", 1, 12, 1)

# =========================
# PROCESS DATA
# =========================
series = get_series(file, city, commodity)

model, metrics = build_model(series)

if model is None:
    st.error("Data tidak cukup untuk membuat model")
    st.stop()

# =========================
# PREDICTION
# =========================
if st.button("🔮 Prediksi"):

    periode = (tahun - 2024) * 12 + bulan
    pred = round(model.predict([[periode]])[0], 2)

    st.success("📌 HASIL PREDIKSI")
    st.write(f"🏷️ Komoditas : {commodity}")
    st.write(f"📍 Kota : {city}")
    st.write(f"💰 Prediksi Harga : Rp {pred:,.0f}")

    st.write("---")
    st.write("📊 Model Quality")

    st.json(metrics)
