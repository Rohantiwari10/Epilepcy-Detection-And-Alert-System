# app.py — Epilepsy Monitoring & Alert System (DEPLOYMENT READY)

import streamlit as st
import pandas as pd
import numpy as np
import pywt
from hurst import compute_Hc
import joblib
import base64
import matplotlib.pyplot as plt
import os
from datetime import datetime
import requests
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

# ------------------------------------------------
# HYBRID KEY MANAGEMENT
# ------------------------------------------------
load_dotenv()

def get_secret(key):
    value = os.getenv(key)
    if value:
        return value
    try:
        if key in st.secrets:
            return st.secrets[key]
    except:
        return None
    return None

TELEGRAM_TOKEN = get_secret("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = get_secret("TELEGRAM_CHAT_ID")
EMAIL_SENDER = get_secret("EMAIL_SENDER")
EMAIL_PASSWORD = get_secret("EMAIL_PASSWORD")
EMAIL_RECEIVER = get_secret("EMAIL_RECEIVER")

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="Epileptic Seizure Detection", page_icon="🧠", layout="wide")

# ------------------------------------------------
# UI THEME
# ------------------------------------------------
def medical_ui_theme():
    st.markdown("""
        <style>
        .main { background: linear-gradient(to bottom right, #eef2f3, #dfe9f3); }
        .title { font-size:36px; font-weight:700; text-align:center; color:#1f3c88; padding:10px 0; }
        .card { background:white; padding:18px; border-radius:12px; box-shadow:0 6px 20px rgba(31,60,136,0.08); margin-bottom:14px; }
        .alert-box { animation: blink 0.7s infinite; background:#ff4b4b; color:white; padding:16px; border-radius:12px; 
        text-align:center; font-size:20px; font-weight:700; margin-bottom:12px; }
        @keyframes blink { 50% { opacity:0; } }
        .stButton>button { background:#1f3c88; color:white; border-radius:10px; padding:8px 12px; }
        .stButton>button:hover { background:#162a60; }
        </style>
    """, unsafe_allow_html=True)

medical_ui_theme()
st.markdown("<div class='title'>🧠 Epileptic Seizure Detection System</div>", unsafe_allow_html=True)

# ------------------------------------------------
# FEATURE EXTRACTION
# ------------------------------------------------
def getHurst(df_eeg):
    return [compute_Hc(df_eeg.iloc[i], kind="change", simplified=True)[0] for i in range(len(df_eeg))]

def statisticsForWavelet(coefs):
    n5, n25, n75, n95 = np.nanpercentile(coefs, [5, 25, 75, 95])
    median = np.nanpercentile(coefs, 50)
    mean = np.nanmean(coefs)
    std = np.nanstd(coefs)
    var = np.nanvar(coefs)
    rms = np.nanmean(np.sqrt(coefs ** 2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]

def getWaveletFeatures(df_eeg, hurst_values):
    features = []
    for i in range(len(df_eeg)):
        coeffs = pywt.wavedec(df_eeg.iloc[i], "db4")
        row = [hurst_values[i]]
        for c in coeffs:
            row += statisticsForWavelet(c)
        features.append(row)
    return pd.DataFrame(features)

# ------------------------------------------------
# SOUND ALERT
# ------------------------------------------------
def play_alarm_sound():
    if not os.path.exists("alert.wav"):
        st.warning("⚠️ alert.wav missing!")
        return
    try:
        audio = open("alert.wav", "rb").read()
        b64 = base64.b64encode(audio).decode()
        st.components.v1.html(
            f"""<audio autoplay="true">
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>""",
            height=0
        )
    except:
        pass

# ------------------------------------------------
# TELEGRAM ALERT
# ------------------------------------------------
def send_telegram_alert(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    chat_ids = [i.strip() for i in TELEGRAM_CHAT_ID.split(",")]
    for cid in chat_ids:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": cid, "text": message, "parse_mode": "HTML"})
        except:
            pass

# ------------------------------------------------
# EMAIL ALERT
# ------------------------------------------------
def send_email_alert(subject, body):
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        return

    receivers = [i.strip() for i in EMAIL_RECEIVER.split(",")]
    msg = MIMEText(body)
    msg["From"] = EMAIL_SENDER
    msg["To"] = ", ".join(receivers)
    msg["Subject"] = subject

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, receivers, msg.as_string())
        server.quit()
    except:
        pass

# ------------------------------------------------
# MASTER ALERT
# ------------------------------------------------
def trigger_alert(row_index=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"""
🚨 <b>SEIZURE DETECTED</b>
🕒 {timestamp}
📊 Source: {row_index + 1 if row_index is not None else "Manual Input"}
"""

    if enable_sound:
        play_alarm_sound()
    if enable_mobile_alert:
        send_telegram_alert(message)
    if enable_email_alert:
        send_email_alert("URGENT: Seizure Detected", message)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
try:
    model = joblib.load("rf_model.joblib")
    scaler = joblib.load("scaler.joblib")
except:
    st.error("❌ Model files missing! Upload rf_model.joblib and scaler.joblib.")
    st.stop()

# ------------------------------------------------
# SIDEBAR CLEAN VERSION
# ------------------------------------------------
with st.sidebar:

    st.markdown("### 🧪 Input Method")
    input_choice = st.radio("", ["Upload CSV", "Manual Input"])

    st.markdown("---")

    st.markdown("### 📥 Sample Testing Dataset")
    sample_df = pd.read_csv("test_chunk_1.csv")
    st.download_button(
        label="Download Sample Dataset",
        data=sample_df.to_csv(index=False),
        file_name="sample_test_data.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("---")

    st.markdown("### ⚙️ Alert Controls")
    enable_sound = st.checkbox("🔊 Sound Alert", True)
    enable_mobile_alert = st.checkbox("📲 Telegram Alert", True)
    enable_email_alert = st.checkbox("📧 Email Alert", False)

# ------------------------------------------------
# MAIN UI CARD
# ------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.write("Upload your EEG CSV file or enter manual EEG values.")
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------
# CSV INPUT
# ------------------------------------------------
if input_choice == "Upload CSV":

    st.markdown("### 📤 Upload EEG CSV File")
    file = st.file_uploader("Upload CSV (178 numeric columns required)")

    if file:
        df = pd.read_csv(file)

        st.write("### Data Preview")
        st.dataframe(df.head())

        # ----- SLIDER MOVED HERE (NEAR PREVIEW) -----
        st.markdown("### Select Row for Prediction")
        row_index = st.slider(
            "Pick a row",
            min_value=0,
            max_value=len(df) - 1,
            value=0
        )

        st.write("### Selected Row")
        st.dataframe(df.iloc[[row_index]])

        if st.button("Predict Selected Row"):
            eeg = df.select_dtypes(include=np.number).iloc[[row_index]]
            raw = eeg.values.flatten().tolist()
            hurst = getHurst(eeg)
            feats = getWaveletFeatures(eeg, hurst)
            scaled = scaler.transform(feats)
            pred = model.predict(scaled)
            prob = model.predict_proba(scaled)

            if pred[0] == 1:
                st.error("🚨 Seizure Detected")
                st.markdown("<div class='alert-box'>SEIZURE DETECTED!</div>", unsafe_allow_html=True)
                trigger_alert(row_index)
            else:
                st.success("🟢 Normal Activity")

            st.metric("Normal", f"{prob[0][0] * 100:.2f}%")
            st.metric("Seizure", f"{prob[0][1] * 100:.2f}%")

# ------------------------------------------------
# MANUAL INPUT
# ------------------------------------------------
else:
    st.info("Enter exactly 178 comma-separated EEG values:")
    txt = st.text_area("EEG Input (178 values)", height=150)

    if st.button("Predict Manually"):
        try:
            vals = [float(i.strip()) for i in txt.split(",") if i.strip()]
            if len(vals) != 178:
                st.error(f"Expected 178 values, got {len(vals)}")
            else:
                df = pd.DataFrame([vals], columns=[f"X{i}" for i in range(1, 179)])
                hurst = getHurst(df)
                feats = getWaveletFeatures(df, hurst)
                scaled = scaler.transform(feats)
                pred = model.predict(scaled)

                if pred[0] == 1:
                    st.error("🚨 Seizure Detected")
                    trigger_alert()
                else:
                    st.success("🟢 Normal Activity")
        except:
            st.error("Invalid input numbers.")

st.markdown("<div style='text-align:center; color:gray;'>Made with ❤️ — Epilepsy Detection System</div>", unsafe_allow_html=True)
