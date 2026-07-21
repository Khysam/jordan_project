import streamlit as st
import pandas as pd
import joblib
 
MODEL_PATH = 'best_sneaker_resale_model.pkl'
FEATURES_PATH = 'model_features.pkl'
UNIQUE_CAT_VALUES_PATH = 'unique_categorical_values.pkl'
 
# st.set_page_config HARUS jadi perintah Streamlit paling pertama di script
st.set_page_config(layout="wide", page_title="Jordan Resale Price Predictor", page_icon="👟")
 
 
@st.cache_resource
def load_artifacts():
    """Load model dan artefak pendukung sekali saja, lalu di-cache."""
    model = joblib.load(MODEL_PATH)
    model_features = joblib.load(FEATURES_PATH)
    unique_cat_values = joblib.load(UNIQUE_CAT_VALUES_PATH)
    return model, model_features, unique_cat_values
 
 
try:
    model, model_features, unique_cat_values = load_artifacts()
except FileNotFoundError:
    st.error(
        "Error: File model tidak ditemukan. Pastikan 'ml_app.py' telah dijalankan "
        "terlebih dahulu untuk melatih dan menyimpan model serta artefaknya, dan "
        "ketiga file .pkl berada di folder yang sama dengan app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()
 
 
# ----------------------------------------------------------------------
# Navigasi (dropdown Menu di sidebar)
# ----------------------------------------------------------------------
MENU_OPTIONS = ["Home", "Panduan Pemakaian", "Prediksi Harga"]
 
if "menu" not in st.session_state:
    st.session_state.menu = "Home"
 
with st.sidebar:
    st.markdown("**Menu**")
    selected_menu = st.selectbox(
        "Menu",
        MENU_OPTIONS,
        index=MENU_OPTIONS.index(st.session_state.menu),
        label_visibility="collapsed",
    )
    st.session_state.menu = selected_menu
 
 
def go_to(page_name: str):
    st.session_state.menu = page_name
    st.rerun()
 
 
# ----------------------------------------------------------------------
# Halaman: Home
# ----------------------------------------------------------------------
def show_home():
    st.title("👟 Jordan Resale Price Prediction App")
    st.markdown(
        "Aplikasi ini dibuat untuk memprediksi estimasi harga jual kembali "
        "(resale) sepatu Air Jordan dalam **USD**, berdasarkan model sepatu, "
        "colorway, kondisi, dan harga ritelnya, menggunakan model Machine "
        "Learning yang sudah dilatih sebelumnya."
    )
 
    st.markdown("### 📊 Dataset")
    st.markdown(
        "Dataset yang digunakan adalah **Jordan Market Dataset 2026**, berisi "
        "data transaksi jual-beli sneaker Air Jordan — seperti model sepatu, "
        "colorway, kondisi, harga ritel, harga jual kembali, kanal penjualan, "
        "hingga lama waktu sepatu berada di inventaris (`jordan_market_dataset_2026.csv`)."
    )
 
    st.markdown("### 🧪 Data Preprocessing & Training")
    st.markdown(
        "Seluruh proses preprocessing data (one-hot encoding untuk fitur "
        "kategorikal), pemilihan model, hingga evaluasi dilakukan pada "
        "`ml_app.py`. Tiga model dibandingkan — **RandomForestRegressor**, "
        "**LinearRegression**, dan **GradientBoostingRegressor** — dan model "
        "dengan skor R² terbaik dipilih serta disimpan untuk digunakan di "
        "aplikasi ini."
    )
 
    st.markdown("### 📂 Isi Aplikasi")
    st.markdown(
        "Aplikasi ini terdiri dari 3 halaman, bisa dipilih lewat dropdown "
        "**Menu** di sidebar kiri:"
    )
    st.markdown(
        "- 🏠 **Home** *(halaman ini)* — penjelasan umum aplikasi, dataset, "
        "dan sumber preprocessing\n"
        "- 📘 **Panduan Pemakaian** — panduan cara mengisi form dan contoh "
        "penggunaan\n"
        "- 👟 **Prediksi Harga** — form untuk memprediksi harga jual kembali sneaker"
    )
 
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📘 Buka Panduan Pemakaian", use_container_width=True):
            go_to("Panduan Pemakaian")
    with col2:
        if st.button("👟 Coba Prediksi Harga", use_container_width=True):
            go_to("Prediksi Harga")
 
    st.markdown("---")
    st.caption(
        "⚠️ Catatan: Model ini dilatih pada dataset `jordan_market_dataset_2026.csv` "
        "dan hanya untuk keperluan demonstrasi. Hasil prediksi adalah estimasi "
        "statistik, bukan harga pasti."
    )
 
 
# ----------------------------------------------------------------------
# Halaman: Panduan Pemakaian
# ----------------------------------------------------------------------
def show_guide():
    st.title("📘 Panduan Pemakaian")
    st.markdown("Berikut cara menggunakan halaman **Prediksi Harga**:")
    st.markdown(
        "1. Buka menu **Prediksi Harga** dari dropdown di sidebar.\n"
        "2. Pilih **Model Sepatu** Air Jordan yang ingin diprediksi.\n"
        "3. Pilih **Colorway** (warna) sepatu.\n"
        "4. Pilih **Kondisi** sepatu — *Deadstock (Brand New)*, *VNDS*, atau *Used*.\n"
        "5. Masukkan **Harga Ritel (USD)** asli sepatu tersebut.\n"
        "6. Klik tombol **Prediksi Harga Jual Kembali**.\n"
        "7. Estimasi harga jual kembali akan muncul di bawah form."
    )
 
    st.markdown("### 💡 Contoh Penggunaan")
    st.markdown(
        "Misalnya kamu punya Air Jordan 1 Low colorway Bred, kondisi VNDS, "
        "dengan harga ritel $180. Pilih ketiga opsi tersebut, isi harga ritel "
        "`180`, lalu klik tombol prediksi untuk melihat estimasi harga jual "
        "kembalinya."
    )
 
    if st.button("👟 Coba Prediksi Harga", use_container_width=True):
        go_to("Prediksi Harga")
 
 
# ----------------------------------------------------------------------
# Halaman: Prediksi Harga
# ----------------------------------------------------------------------
def show_prediction():
    st.title("👟 Prediktor Harga Jual Kembali Sneaker")
    st.markdown(
        "Aplikasi ini memprediksi harga jual kembali sepatu sneaker berdasarkan "
        "model, colorway, kondisi, dan harga ritelnya."
    )
    st.markdown("---")
 
    st.subheader("Masukkan Detail Sneaker")
 
    shoe_models_options = unique_cat_values['Shoe_Model']
    colorway_options = unique_cat_values['Colorway']
    condition_options = unique_cat_values['Condition']
 
    col1, col2 = st.columns(2)
    with col1:
        selected_shoe_model = st.selectbox("Model Sepatu", shoe_models_options)
        selected_colorway = st.selectbox("Colorway", colorway_options)
    with col2:
        selected_condition = st.selectbox("Kondisi", condition_options)
        retail_price = st.number_input("Harga Ritel (USD)", min_value=0, value=200, step=5)
 
    st.markdown("---")
 
    if st.button("Prediksi Harga Jual Kembali"):
        if retail_price <= 0:
            st.warning("Harga Ritel harus lebih besar dari 0 untuk prediksi yang akurat.")
        else:
            input_df = pd.DataFrame(0, index=[0], columns=model_features)
            input_df['Retail_Price_USD'] = retail_price
 
            for feature_col in model_features:
                if feature_col.startswith('Shoe_Model_') and feature_col.replace('Shoe_Model_', '') == selected_shoe_model:
                    input_df[feature_col] = 1
                elif feature_col.startswith('Colorway_') and feature_col.replace('Colorway_', '') == selected_colorway:
                    input_df[feature_col] = 1
                elif feature_col.startswith('Condition_') and feature_col.replace('Condition_', '') == selected_condition:
                    input_df[feature_col] = 1
 
            try:
                final_input_data = input_df[model_features]
                prediction = model.predict(final_input_data)[0]
 
                st.success(f"### Harga Jual Kembali Prediksi: ${prediction:.2f} USD")
                st.info("Prediksi ini didasarkan pada model Machine Learning yang dilatih dengan data historis.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
 
 
# ----------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------
if st.session_state.menu == "Home":
    show_home()
elif st.session_state.menu == "Panduan Pemakaian":
    show_guide()
else:
    show_prediction()
 
