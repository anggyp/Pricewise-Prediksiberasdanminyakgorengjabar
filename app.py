import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math


# =========================
# LOAD DATA
# =========================
def get_series(file_name, wilayah, row_name):

    df = pd.read_excel(file_name)

    # filter berdasarkan wilayah + komoditas
    row = df[
        (df.iloc[:, 0].astype(str).str.strip() == wilayah) &
        (df.iloc[:, 1].astype(str).str.strip() == row_name)
    ]

    if row.empty:
        return []

    row = row.iloc[0]

    vals = []

    for v in row.iloc[2:]:
        nilai = str(v).strip()

        if nilai in ["-", "", "nan", "NaN"]:
            vals.append(0)
        else:
            vals.append(float(str(nilai).replace(",", "")))

    return vals


# =========================
# AMBIL WILAYAH (FIX ANGKA)
# =========================
def get_wilayah_list():
    df = pd.read_excel("dataset/beras.xlsx")

    wilayah = df.iloc[:, 0].astype(str)

    # ambil hanya yang ada huruf (nama kota)
    wilayah = wilayah[wilayah.str.contains("[A-Za-z]", regex=True)]

    return wilayah.dropna().unique().tolist()


# =========================
# BUILD MODEL
# =========================
def build(wilayah):

    beras = get_series("dataset/beras.xlsx", wilayah, "Beras")
    minyak = get_series("dataset/minyak.xlsx", wilayah, "Minyak Goreng")

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
st.write("Model Linear Regression")

# dropdown wilayah (SUDAH FIX ANGKA)
wilayah_list = get_wilayah_list()
wilayah = st.selectbox("Pilih Wilayah", wilayah_list)

tahun = st.number_input("Tahun", min_value=2024, max_value=2035, value=2026)
bulan = st.number_input("Bulan", min_value=1, max_value=12, value=1)

model_beras, model_minyak, metrics = build(wilayah)

if model_beras is None:
    st.error("Data tidak ditemukan untuk wilayah ini di dataset.")
    st.stop()

if st.button("🔮 Prediksi"):

    periode = (tahun - 2024) * 12 + bulan

    pred_beras = round(model_beras.predict([[periode]])[0], 2)
    pred_minyak = round(model_minyak.predict([[periode]])[0], 2)

    st.success(f"🍚 Beras ({wilayah}): {pred_beras}")
    st.success(f"🛢️ Minyak ({wilayah}): {pred_minyak}")

    st.write("### 📊 Evaluation Metrics")
    st.json(metrics)
