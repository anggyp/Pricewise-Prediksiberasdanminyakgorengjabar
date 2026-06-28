
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math


# =========================
# LOAD DATA
# =========================
def get_series(file_name, sheet_name, row_name):

    df = pd.read_excel(file_name, sheet_name=sheet_name)

    row = df[df.iloc[:, 1].astype(str).str.strip() == row_name].iloc[0]

    vals = []

    for v in row.iloc[2:]:
        nilai = str(v).strip()

        if nilai == "-" or nilai == "" or nilai.lower() == "nan":
            vals.append(0)
        else:
            vals.append(float(nilai.replace(",", "")))

    return vals


# =========================
# BUILD MODEL
# =========================
def build(wilayah):

    beras = get_series("dataset/beras.xlsx", wilayah, "Beras")
    minyak = get_series("dataset/minyak.xlsx", wilayah, "Minyak Goreng")

    X = np.arange(1, len(beras) + 1).reshape(-1, 1)

    model_beras = LinearRegression().fit(X, beras)
    model_minyak = LinearRegression().fit(X, minyak)

    pred_beras_train = model_beras.predict(X)
    pred_minyak_train = model_minyak.predict(X)

    metrics = {
        "mae_beras": round(mean_absolute_error(beras, pred_beras_train), 2),
        "mse_beras": round(mean_squared_error(beras, pred_beras_train), 2),
        "rmse_beras": round(math.sqrt(mean_squared_error(beras, pred_beras_train)), 2),
        "r2_beras": round(r2_score(beras, pred_beras_train), 4),

        "mae_minyak": round(mean_absolute_error(minyak, pred_minyak_train), 2),
        "mse_minyak": round(mean_squared_error(minyak, pred_minyak_train), 2),
        "rmse_minyak": round(math.sqrt(mean_squared_error(minyak, pred_minyak_train)), 2),
        "r2_minyak": round(r2_score(minyak, pred_minyak_train), 4),
    }

    return model_beras, model_minyak, metrics


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="PriceWise Jabar", layout="centered")

st.title("📊 Prediksi Harga Beras & Minyak Goreng Jabar")
st.write("Aplikasi prediksi menggunakan Linear Regression")

# input user
wilayah = st.selectbox("Pilih Wilayah", ["Kota Bandung"])

tahun = st.number_input("Tahun", min_value=2024, max_value=2035, value=2026)
bulan = st.number_input("Bulan", min_value=1, max_value=12, value=1)

# build model
model_beras, model_minyak, metrics = build(wilayah)

# tombol prediksi
if st.button("🔮 Prediksi"):

    periode = (tahun - 2024) * 12 + bulan

    pred_b = round(model_beras.predict([[periode]])[0], 2)
    pred_m = round(model_minyak.predict([[periode]])[0], 2)

    st.success(f"🍚 Prediksi Harga Beras: {pred_b}")
    st.success(f"🛢️ Prediksi Harga Minyak Goreng: {pred_m}")

    st.write("### 📊 Metrics Model")
    st.json(metrics)
