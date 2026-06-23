
from flask import Flask,render_template,request
import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import math

app=Flask(__name__)

def get_series(file_name, sheet_name, row_name):

    df = pd.read_excel(
        file_name,
        sheet_name=sheet_name
    )

    row = df[
        df.iloc[:,1].astype(str).str.strip() == row_name
    ].iloc[0]

    vals = []

    for v in row.iloc[2:]:

        nilai = str(v).strip()

        if nilai == "-" or nilai == "" or nilai.lower() == "nan":
            vals.append(0)

        else:
            vals.append(
                float(nilai.replace(",", ""))
            )
    return vals


def build(wilayah):

    beras = get_series(
        "dataset/beras.xlsx",
        wilayah,
        "Beras"
    )

    minyak = get_series(
        "dataset/minyak.xlsx",
        wilayah,
        "Minyak Goreng"

    )

    X = np.arange(1, len(beras)+1).reshape(-1,1)

    mb = LinearRegression().fit(X, beras)
    mm = LinearRegression().fit(X, minyak)

    pb = mb.predict(X)
    pm = mm.predict(X)

    metrics = {
        "mae_b": round(mean_absolute_error(beras,pb),2),
        "mse_b": round(mean_squared_error(beras,pb),2),
        "rmse_b": round(math.sqrt(mean_squared_error(beras,pb)),2),
        "r2_b": round(r2_score(beras,pb),4),

        "mae_m": round(mean_absolute_error(minyak,pm),2),
        "mse_m": round(mean_squared_error(minyak,pm),2),
        "rmse_m": round(math.sqrt(mean_squared_error(minyak,pm)),2),
        "r2_m": round(r2_score(minyak,pm),4),
    }

    return mb, mm, metrics, beras, minyak

@app.route("/", methods=["GET", "POST"])
def index():
    pred_b = None
    pred_m = None
    bulan_hasil = None
    tahun_hasil = None

    tahun = 2026
    bulan = 1

    wilayah = "Kota Bandung"

    if request.method == "POST":
        wilayah = request.form["wilayah"]
    mb, mm, metrics, beras, minyak = build(wilayah)

    if request.method == "POST":

        tahun = int(request.form["tahun"])
        bulan = int(request.form["bulan"])

        print("TAHUN =", tahun)
        print("BULAN =", bulan)

        periode = (tahun - 2024) * 12 + bulan

        pred_b = round(
            mb.predict([[periode]])[0],
            2
        )

        pred_m = round(
            mm.predict([[periode]])[0],
            2
        )

        nama_bulan = [
            "Januari", "Februari", "Maret", "April",
            "Mei", "Juni", "Juli", "Agustus",
            "September", "Oktober", "November", "Desember"
        ]

        bulan_hasil = nama_bulan[bulan - 1]
        tahun_hasil = tahun

    return render_template(
    "index.html",
    metrics=metrics,
    pred_b=pred_b,
    pred_m=pred_m,
    bulan_hasil=bulan_hasil,
    tahun_hasil=tahun_hasil,
    wilayah=wilayah,
    tahun=tahun,
    bulan=bulan,
    data_beras=beras,
    data_minyak=minyak
)

if __name__=="__main__":
    app.run(debug=True)
