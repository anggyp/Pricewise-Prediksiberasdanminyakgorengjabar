import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math


# =========================
# LIST KOTA (DARI SHEET)
# =========================
def get_city_list():
    file = pd.ExcelFile("dataset/beras.xlsx")
    return file.sheet_names


# =========================
# AMBIL DATA PER KOMODITAS
# =========================
def get_series(file_name, city, keyword):

    df = pd.read_excel(file_name, sheet_name=city)

    # filter baris sesuai komoditas (beras / minyak)
    rows = df[df.iloc[:, 1].astype(str).str.contains(keyword, case=False, na=False)]

    if rows.empty:
        return []

    # ambil semua baris lalu rata-rata
    data = rows.iloc[:, 2:]

    # ubah format "19,600" → 19600
    data = data.replace("-", 0)
    data = data.applymap(lambda x: float(str(x).replace(",", "")) if str(x) not in ["nan", ""] else 0)

    return data.mean().tolist()


# =========================
# BUILD MODEL
# =========================
def build(city):

    beras = get_series("dataset/beras.xlsx", city, "Beras")
    minyak = get_series("dataset/minyak.xlsx", city, "Minyak")

    if len(beras) == 0 or len(minyak) == 0:
        return None, None, {}

    X = np.arange(1, len(beras) + 1).reshape(-1, 1)

    model_beras = LinearRegression().fit(X, beras)
    model_minyak = LinearRegression().fit(X, minyak)

    pred_beras = model_beras.predict(X)
    pred_minyak = model_minyak.predict(X)

    metrics = {
        "mae_beras": round(mean_absolute_error(beras, pred_beras), 2),
        "mse_beras": round(mean_squared_error(beras, pred_beras), 2),
        "rmse_beras": round(math.sqrt(mean_squared_error(beras, pred_beras)), 2),
        "r2_beras": round(r2_score(beras, pred_beras), 4),

        "mae_minyak": round(mean_absolute_error(minyak, pred_minyak), 2),
        "mse_minyak": round(mean_squared_error(minyak, pred_minyak), 2),
        "rmse_minyak": round(math.sqrt(mean_squared_error(minyak, pred_minyak)), 2),
        "r2_minyak": round(r2_score(minyak, pred_minyak), 4),
    }

    return model_beras, model_minyak, metrics


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="PriceWise Jabar", layout="centered")

st.title("📊 PriceWise - Prediksi Harga Beras & Minyak Goreng Jabar")

# kota dari sheet
cities = get_city_list()
city = st.selectbox("Pilih Kota", cities)

# komoditas fix (biar simpel & stabil)
komoditas = st.selectbox("Komoditas", ["Beras", "Minyak"])

tahun = st.number_input("Tahun", 2024, 2035, 2026)
bulan = st.number_input("Bulan", 1, 12, 1)

model_beras, model_minyak, metrics = build(city)

if model_beras is None:
    st.error("Data tidak ditemukan di dataset")
    st.stop()

if st.button("🔮 Prediksi"):

    periode = (tahun - 2024) * 12 + bulan

    if komoditas == "Beras":
        pred = round(model_beras.predict([[periode]])[0], 2)
        st.success(f"🍚 Prediksi Beras di {city}: {pred}")
    else:
        pred = round(model_minyak.predict([[periode]])[0], 2)
        st.success(f"🛢️ Prediksi Minyak di {city}: {pred}")

    st.write("### 📊 Metrics Model")
    st.json(metrics)
