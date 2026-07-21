import streamlit as st
import pandas as pd
import joblib

# Define paths for artifacts
MODEL_PATH = 'best_sneaker_resale_model.pkl'
FEATURES_PATH = 'model_features.pkl'
UNIQUE_CAT_VALUES_PATH = 'unique_categorical_values.pkl'

st.set_page_config(layout="wide", page_title="Sneaker Resale Price Predictor")


@st.cache_resource
def load_artifacts():
    """Load model and supporting artifacts once and cache them across reruns."""
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

st.title("👟 Prediktor Harga Jual Kembali Sneaker")
st.markdown(
    "Aplikasi ini memprediksi harga jual kembali sepatu sneaker berdasarkan "
    "model, colorway, kondisi, dan harga ritelnya."
)
st.markdown("---")

# --- Input Fields ---
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

# --- Prediction Button ---
if st.button("Prediksi Harga Jual Kembali"):
    if retail_price <= 0:
        st.warning("Harga Ritel harus lebih besar dari 0 untuk prediksi yang akurat.")
    else:
        # Create an empty DataFrame with all possible feature columns
        # and initialize with zeros
        input_df = pd.DataFrame(0, index=[0], columns=model_features)

        # Fill in the user's input for the numerical feature
        input_df['Retail_Price_USD'] = retail_price

        # Fill in one-hot encoded categorical features
        for feature_col in model_features:
            if feature_col.startswith('Shoe_Model_') and feature_col.replace('Shoe_Model_', '') == selected_shoe_model:
                input_df[feature_col] = 1
            elif feature_col.startswith('Colorway_') and feature_col.replace('Colorway_', '') == selected_colorway:
                input_df[feature_col] = 1
            elif feature_col.startswith('Condition_') and feature_col.replace('Condition_', '') == selected_condition:
                input_df[feature_col] = 1

        try:
            # Ensure the input DataFrame has columns in the exact order as model_features
            final_input_data = input_df[model_features]
            prediction = model.predict(final_input_data)[0]

            st.success(f"### Harga Jual Kembali Prediksi: ${prediction:.2f} USD")
            st.info("Prediksi ini didasarkan pada model Machine Learning yang dilatih dengan data historis.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

st.markdown("---")
st.markdown("Dibuat dengan ❤️")
