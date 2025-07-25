import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Konfigurasi dasar aplikasi ---
st.set_page_config(page_title="Prediksi Daya Listrik pada PLTGU", layout="centered")

# --- Load Model dan Data ---
# --- Load Model dan Data ---
@st.cache_resource
def load_model():
    return joblib.load("model_gradient_boosting.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

model = load_model()
df = load_data()

# Evaluasi model pada test set (nilai tetap dari IPYNB)
r2_val = 0.9669
mae_val = 2.10
rmse_val = 3.00

# --- Navigasi Halaman ---
page = st.sidebar.selectbox("📁 Pilih Halaman", ["🔍 Prediksi", "ℹ️ Tentang Aplikasi"])

# === Halaman Prediksi ===
if page == "🔍 Prediksi":
    st.title("🔌 Prediksi Daya Listrik pada PLTGU")
    st.markdown("Masukkan data kondisi lingkungan untuk memprediksi **daya listrik**. Model ini menggunakan algoritma **Gradient Boosting Regression**.")

    # Sidebar input
    st.sidebar.header("Input Parameter Lingkungan")
    at = st.sidebar.number_input("Suhu Udara Sekitar (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    v = st.sidebar.number_input("Tekanan Vakum Buangan (cm Hg)", min_value=20.0, max_value=100.0, value=40.0, step=0.1)
    ap = st.sidebar.number_input("Tekanan Udara Lingkungan (mbar)", min_value=900.0, max_value=1100.0, value=1013.0, step=0.1)
    rh = st.sidebar.number_input("Kelembapan Relatif (%)", min_value=10.0, max_value=100.0, value=60.0, step=0.1)


    # Tombol prediksi
    if st.button("🔎 Prediksi"):
        X_new = np.array([[at, v, ap, rh]])
        pred_pe = model.predict(X_new)[0]
        avg_pe = df['PE'].mean()

        st.subheader("💡 Hasil Prediksi")
        st.metric(label="Prediksi Daya Listrik:", value=f"{pred_pe:.2f} MW")

        # Rekomendasi berdasarkan hasil prediksi
        if pred_pe < avg_pe - 10:
            st.info("⚠️ Prediksi daya lebih rendah dari rata-rata historis. Mungkin kondisi lingkungan tidak optimal.")
        elif pred_pe > avg_pe + 10:
            st.success("✅ Prediksi daya di atas rata-rata. Kondisi lingkungan kemungkinan mendukung kinerja optimal.")
        else:
            st.warning("ℹ️ Prediksi daya berada dalam kisaran rata-rata. Kinerja stabil, tapi tidak maksimum.")

        # Garis pemisah
        st.markdown("---")

        # Evaluasi model
        st.markdown("#### ⚙️ Evaluasi Model")
        st.write("""
Evaluasi model dilakukan untuk mengetahui seberapa akurat model dalam memprediksi daya listrik berdasarkan data uji (test set).

**Penjelasan metrik evaluasi:**
- **R² Score:** Seberapa baik model menjelaskan variansi data (semakin mendekati 1, semakin baik)
- **MAE (Mean Absolute Error):** Rata-rata selisih absolut antara prediksi dan data aktual)
- **RMSE (Root Mean Squared Error):** Akar dari rata-rata kesalahan kuadrat (lebih sensitif terhadap outlier)
""")

        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{r2_val:.4f}")
        col2.metric("MAE", f"{mae_val:.2f} MW")
        col3.metric("RMSE", f"{rmse_val:.2f} MW")

        # Garis pemisah
        st.markdown("---")

        # Cek apakah input ada di data asli
        df_match = df[
            (df['AT'].round(2) == round(at, 2)) &
            (df['V'].round(2) == round(v, 2)) &
            (df['AP'].round(2) == round(ap, 2)) &
            (df['RH'].round(2) == round(rh, 2))
        ]

        if not df_match.empty:
            actual_pe = df_match['PE'].values[0]
            error = abs(actual_pe - pred_pe)
            st.success(f"🎯 Nilai aktual Daya listrik dari dataset: **{actual_pe:.2f} MW**")
            st.info(f"Selisih absolut prediksi vs aktual: **{error:.2f} MW**")
        else:
            st.warning("⚠️ Data input ini tidak ditemukan dalam dataset asli, nilai aktual tidak tersedia.")

        # Garis pemisah
        st.markdown("---")

        # Visualisasi distribusi data PE
        st.subheader("📊 Distribusi Data Daya Listrik")
        st.markdown("""
Visualisasi berikut menunjukkan sebaran daya listrik dari data historis.  
Garis **merah** menunjukkan posisi prediksi Anda pada distribusi ini, dan garis **hijau** menunjukkan rata-rata seluruh data.

Ini membantu Anda melihat apakah prediksi termasuk nilai umum, rendah, atau sangat tinggi.
""")

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df['PE'], bins=50, kde=True, ax=ax, color='skyblue')
        ax.axvline(pred_pe, color='red', linestyle='--', label='Prediksi Anda')
        ax.axvline(avg_pe, color='green', linestyle='--', label='Rata-rata Data')
        ax.set_title("Distribusi Daya Listrik")
        ax.set_xlabel("Daya Listrik (MW)")
        ax.legend()
        st.pyplot(fig)

# === Halaman Tentang Aplikasi ===
elif page == "ℹ️ Tentang Aplikasi":
    st.title("ℹ️ Tentang Aplikasi Prediksi Daya Listrik pada PLTGU")
    st.markdown("""
Aplikasi ini bertujuan untuk **memprediksi daya listrik** dari pembangkit listrik tenaga gas dan uap (PLTGU) menggunakan **machine learning**.

### 🧠 Model yang Digunakan
- **Gradient Boosting Regressor**
- Dilatih menggunakan dataset dari [UCI CCPP Dataset](https://archive.ics.uci.edu/ml/datasets/combined+cycle+power+plant) dan [CCPP Environmental Dataset 2021–2024 (GitHub)](https://github.com/rizkifirmansyahil/Combined-Cycle-Power-Plant-CCPP-Environmental-and-Power-Output-Dataset-2021-2024-)

## 📥 Input yang Dibutuhkan
- **AT** (*ambient temperature*): Suhu udara sekitar (°C)  
- **V** (*exhaust vacuum*): Tekanan vakum buangan (cm Hg)  
- **AP** (*ambient pressure*): Tekanan udara lingkungan (mbar)  
- **RH** (*relative humidity*): Kelembapan relatif (%)

### 📊 Output
- Prediksi **daya listrik keluaran bersih** (*net electrical power output*, PE) dalam satuan megawatt (MW)
- Evaluasi model: R², MAE, dan RMSE
- Rekomendasi hasil dan distribusi data

### 📌 Catatan
- Prediksi berbasis input user dan bisa dibandingkan dengan data asli jika tersedia
- Cocok untuk simulasi efisiensi dan studi performa PLTGU

---

Developed by **Faizah Rizki Auliawati**  
Fakultas Teknologi Industri, Jurusan Informatika  
Universitas Gunadarma  
📧 frauliawati@gmail.com  

*This application is part of an academic project. Not intended for commercial or operational use.*
""")
