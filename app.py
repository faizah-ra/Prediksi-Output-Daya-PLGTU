import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --- Konfigurasi Aplikasi ---
st.set_page_config(page_title="Prediksi Daya Listrik pada PLTGU", layout="centered")

# --- Load Model dan Data ---
@st.cache_resource
def load_model():
    return joblib.load("model_gradient_boosting.pkl")  # pastikan model ini dilatih dari data baru

@st.cache_data
def load_data():
    return pd.read_excel("ccpp_env_2021_2024.xlsx")  # dataset baru

@st.cache_data
def load_test_data():
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv")
    return X_test, y_test

model = load_model()
df = load_data()
X_test, y_test = load_test_data()

# Evaluasi Model
y_pred_test = model.predict(X_test)
r2_val = r2_score(y_test, y_pred_test)
mae_val = mean_absolute_error(y_test, y_pred_test)
rmse_val = np.sqrt(mean_squared_error(y_test, y_pred_test))

# --- Navigasi Halaman ---
page = st.sidebar.selectbox("📁 Pilih Halaman", ["🔍 Prediksi", "ℹ️ Tentang Aplikasi"])

# === Halaman Prediksi ===
if page == "🔍 Prediksi":
    st.title("🔌 Prediksi Daya Listrik pada PLTGU")
    st.markdown("Masukkan data kondisi lingkungan untuk memprediksi **daya listrik**. Model ini menggunakan algoritma **Gradient Boosting Regression**.")

    # Sidebar Input
    st.sidebar.header("Input Parameter Lingkungan")
    at = st.sidebar.number_input("Suhu Udara Sekitar (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    v = st.sidebar.number_input("Tekanan Vakum Buangan (cm Hg)", min_value=20.0, max_value=100.0, value=40.0, step=0.1)
    ap = st.sidebar.number_input("Tekanan Udara Lingkungan (mbar)", min_value=900.0, max_value=1100.0, value=1013.0, step=0.1)
    rh = st.sidebar.number_input("Kelembapan Relatif (%)", min_value=10.0, max_value=100.0, value=60.0, step=0.1)

    # Tombol Prediksi
    if st.button("🔎 Prediksi"):
        X_new = np.array([[at, v, ap, rh]])
        pred_pe = model.predict(X_new)[0]
        avg_pe = df['PE'].mean()

        st.subheader("💡 Hasil Prediksi")
        st.metric(label="Prediksi Daya Listrik:", value=f"{pred_pe:.2f} MW")

        # Rekomendasi
        if pred_pe < avg_pe - 10:
            st.info("⚠️ Prediksi daya lebih rendah dari rata-rata historis. Mungkin kondisi lingkungan tidak optimal.")
        elif pred_pe > avg_pe + 10:
            st.success("✅ Prediksi daya di atas rata-rata. Kondisi lingkungan kemungkinan mendukung kinerja optimal.")
        else:
            st.warning("ℹ️ Prediksi daya berada dalam kisaran rata-rata. Kinerja stabil, tapi tidak maksimum.")

        st.markdown("---")

        # Evaluasi Model
        st.markdown("#### ⚙️ Evaluasi Model")
        st.write("""
**Penjelasan metrik evaluasi model:**

- **R² Score:** Seberapa baik model menjelaskan variansi data
- **MAE:** Rata-rata selisih absolut prediksi vs aktual
- **RMSE:** Akar dari rata-rata kuadrat kesalahan
""")
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{r2_val:.4f}")
        col2.metric("MAE", f"{mae_val:.2f} MW")
        col3.metric("RMSE", f"{rmse_val:.2f} MW")

        st.markdown("---")

        # Cek Nilai Aktual Jika Ada
        df_match = df[
            (df['AT'].round(2) == round(at, 2)) &
            (df['V'].round(2) == round(v, 2)) &
            (df['AP'].round(2) == round(ap, 2)) &
            (df['RH'].round(2) == round(rh, 2))
        ]

        if not df_match.empty:
            actual_pe = df_match['PE'].values[0]
            error = abs(actual_pe - pred_pe)
            st.success(f"🎯 Nilai aktual dari dataset: **{actual_pe:.2f} MW**")
            st.info(f"Selisih prediksi vs aktual: **{error:.2f} MW**")
        else:
            st.warning("⚠️ Kombinasi input tidak ditemukan dalam dataset. Nilai aktual tidak tersedia.")

        st.markdown("---")

        # Visualisasi Histogram
        st.subheader("📊 Distribusi Data Daya Listrik")
        st.markdown("""
Distribusi historis daya listrik (PE) ditampilkan di bawah.  
- Garis **merah**: nilai prediksi Anda  
- Garis **hijau**: rata-rata keseluruhan data
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
Aplikasi ini bertujuan untuk **memprediksi daya listrik** dari pembangkit listrik tenaga gas dan uap (PLTGU) menggunakan algoritma **Gradient Boosting Regression**.

### 📚 Dataset
- Dataset: *CCPP Environmental Data 2021–2024*  
- Sumber: data lingkungan dan output daya sintetis untuk studi PLTGU

### 🧠 Model
- Algoritma: Gradient Boosting Regressor
- Input: AT (suhu), V (vakum buangan), AP (tekanan udara), RH (kelembapan)
- Output: PE (daya listrik bersih dalam MW)

### 📈 Evaluasi
- R² Score, MAE, dan RMSE digunakan untuk menilai kinerja model
- Visualisasi distribusi PE

---

**Dikembangkan oleh:** Faizah Rizki Auliawati  
Fakultas Teknologi Industri, Jurusan Informatika  
Universitas Gunadarma  
📧 frauliawati@gmail.com  

> *Aplikasi ini dibuat sebagai bagian dari proyek ilmiah dan tidak untuk penggunaan operasional.*
""")
