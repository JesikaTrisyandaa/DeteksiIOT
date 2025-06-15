import streamlit as st
import numpy as np
import pickle  
from modelsvm import SVM

st.set_page_config(page_title="IoTect", layout="wide")

st.markdown("""
    <style>
    .header-container {
        background-color: #0B2C54;
        padding: 40px 0 30px 0;
        text-align: center;
        color: white;
        width: 100%;
        margin-top: -3rem;
    }
    </style>
    <div class="header-container">
        <h1>IoTect</h1>
        <h3>BotNet Attack Detection on IoT Networks</h3>
        <p>Masukkan parameter lalu lintas jaringan IoT Anda untuk mendeteksi potensi serangan BotNet secara otomatis</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Daftar nama fitur
feature_names = [
    "MI_dir_L0.1_weight", "MI_dir_L0.1_mean", "MI_dir_L0.1_variance",
    "H_L0.1_weight", "H_L0.1_mean", "H_L0.1_variance",
    "HH_L0.1_weight", "HH_L0.1_mean", "HH_L0.1_std",
    "HH_L0.1_magnitude", "HH_L0.1_radius", "HH_L0.1_covariance", "HH_L0.1_pcc",
    "HH_jit_L0.1_weight", "HH_jit_L0.1_mean", "HH_jit_L0.1_variance",
    "HpHp_L0.1_weight", "HpHp_L0.1_mean", "HpHp_L0.1_std",
    "HpHp_L0.1_magnitude", "HpHp_L0.1_radius", "HpHp_L0.1_covariance", "HpHp_L0.1_pcc"
]

# Input user
col1, col2, col3 = st.columns(3)
user_inputs = []

with col1:
    for i in range(6):
        val = st.text_input(feature_names[i], key=f'col1_{i}')
        user_inputs.append(val)

with col2:
    for i in range(6, 12):
        val = st.text_input(feature_names[i], key=f'col2_{i}')
        user_inputs.append(val)

with col3:
    for i in range(12, 18):
        val = st.text_input(feature_names[i], key=f'col3_{i}')
        user_inputs.append(val)

# Tombol prediksi
if st.button("PREDICT"):
    try:
        if "" in user_inputs:
            st.warning("⚠️ Semua input harus diisi.")
        else:
            input_data = np.array([float(x) for x in user_inputs]).reshape(1, -1)

            # Load model menggunakan pickle
            with open("SVM_MODEL_MANUAL.pkl", "rb") as file:
                model = pickle.load(file)

            prediction = model.predict(input_data)

            if prediction[0] == 1:
                st.error("🚨 BotNet Attack Detected!")
            else:
                st.success("✅ Normal Traffic - No Attack Detected.")
    except ValueError as e:
        st.warning(f"⚠️ Pastikan semua input diisi dengan angka. Error: {e}")
    except FileNotFoundError:
        st.error("❌ Model file tidak ditemukan.")
