# app.py — Epilepsy Monitoring & Alert System (TELEGRAM ENABLED)

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
import requests  # Required for Telegram
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# TELEGRAM CONFIGURATION
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ------------------------------------------------
# Streamlit Page Config
# ------------------------------------------------
st.set_page_config(page_title="Epileptic Seizure Detection", page_icon="🧠", layout="wide")

# ------------------------------------------------
# UI Theme
# ------------------------------------------------
def medical_ui_theme():
    st.markdown("""
        <style>
        .main { background: linear-gradient(to bottom right, #eef2f3, #dfe9f3); }
        .title { font-size:36px; font-weight:700; text-align:center; color:#1f3c88; padding:10px 0; }
        .card { background:white; padding:18px; border-radius:12px; box-shadow:0 6px 20px rgba(31,60,136,0.08); margin-bottom:14px; }
        .alert-box { animation: blink 0.7s infinite; background:#ff4b4b; color:white; padding:16px; border-radius:12px; text-align:center; font-size:20px; font-weight:700; margin-bottom:12px; }
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
        feat_row = [hurst_values[i]]
        for c in coeffs:
            feat_row += statisticsForWavelet(c)
        features.append(feat_row)
    return pd.DataFrame(features)

# ------------------------------------------------
# SOUND ALERT
# ------------------------------------------------
def play_alarm_sound():
    if not os.path.exists("alert.wav"):
        st.warning("⚠️ alert.wav missing! Place an alert sound file in the folder.")
        return
    try:
        data = open("alert.wav", "rb").read()
        b64 = base64.b64encode(data).decode()
        st.components.v1.html(
            f"""<audio autoplay="true"><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>""",
            height=0
        )
    except Exception as e:
        st.error(f"Alarm error: {e}")

# ------------------------------------------------
# TELEGRAM ALERT FUNCTION
# ------------------------------------------------
def send_telegram_alert(message):
    """
    Sends notification via Telegram Bot (Free & Fast)
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        st.error("⚠️ Telegram keys missing in .env file!")
        return

    # 'parse_mode': 'HTML' allows us to use <b>bold</b> text
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML" 
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            st.success("📲 Telegram Alert Sent Successfully!")
        else:
            st.error(f"Telegram Error: {response.text}")
    except Exception as e:
        st.error(f"Connection Failed: {e}")

# ------------------------------------------------
# MASTER ALERT FUNCTION
# ------------------------------------------------
def trigger_alert(row_index=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Construct the message with HTML formatting
    alert_message = f"🚨 <b>SEIZURE DETECTED</b>\n"
    alert_message += f"🕒 Time: {timestamp}\n"
    
    if row_index is not None:
        # Add +1 so "Row 0" becomes "Row 1" (Human readable)
        alert_message += f"📊 Data Source: <b>Row {row_index + 1}</b>"
    else:
        alert_message += "📊 Data Source: <b>Manual Input</b>"

    alert_message += "\n\n❗ <b>Patient requires immediate attention.</b>"

    # Alarm
    if enable_sound:
        play_alarm_sound()

    # Telegram Alert
    if enable_mobile_alert:
        send_telegram_alert(alert_message)

# ------------------------------------------------
# LOAD MODEL + SCALER
# ------------------------------------------------
try:
    model = joblib.load("rf_model.joblib")
    scaler = joblib.load("scaler.joblib")
except:
    st.error("Model or scaler file missing! Ensure rf_model.joblib and scaler.joblib are in the folder.")
    st.stop()

# ------------------------------------------------
# SIDEBAR SETTINGS
# ------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Alert Controls")

    enable_sound = st.checkbox("Play Sound Alarm", True)
    enable_mobile_alert = st.checkbox("Send Telegram Alert", True)

    if enable_mobile_alert:
        st.info(f"Alerts enabled for Chat ID: {TELEGRAM_CHAT_ID}")
        
    st.divider()
    st.markdown("### Status")
    if TELEGRAM_TOKEN:
        st.success("Telegram Bot: Connected")
    else:
        st.error("Telegram Bot: Not Configured")

# ------------------------------------------------
# PREDICTION FUNCTION
# ------------------------------------------------
def make_prediction(df, raw=None, row_index=None):

    hurst = getHurst(df)
    feats = getWaveletFeatures(df, hurst)
    scaled = scaler.transform(feats)

    pred = model.predict(scaled)
    prob = model.predict_proba(scaled)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Prediction Result")

    if pred[0] == 1:
        st.error("🚨 Epileptic Seizure Detected")
        st.markdown("<div class='alert-box'> SEIZURE DETECTED — ATTENTION REQUIRED </div>", unsafe_allow_html=True)

        # Pass row_index to the alert system
        trigger_alert(row_index)

    else:
        st.success("🟢 Normal Brain Activity")

    col1, col2 = st.columns(2)
    col1.metric("Normal", f"{prob[0][0] * 100:.2f}%")
    col2.metric("Seizure", f"{prob[0][1] * 100:.2f}%")

    if raw:
        st.write("### EEG Preview")
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(raw)
        ax.grid(alpha=0.2)
        st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------
# MAIN UI
# ------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.write("Upload EEG CSV or select row to detect seizures.")
st.markdown("</div>", unsafe_allow_html=True)

input_choice = st.sidebar.radio("Input Method:", ["Upload CSV", "Manual Input"])

# CSV INPUT
if input_choice == "Upload CSV":
    file = st.sidebar.file_uploader("Upload CSV (X1..X178 required)")

    if file:
        df = pd.read_csv(file)

        st.write("### Data Preview:")
        st.dataframe(df.head())

        row_index = st.sidebar.slider(
            "Select Row for Prediction",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            step=1
        )

        st.sidebar.write(f"Selected Row: **{row_index}**")
        st.write("### Selected EEG Row")
        st.dataframe(df.iloc[[row_index]])

        if st.sidebar.button("Predict Selected Row"):
            eeg = df.select_dtypes(include=np.number).iloc[[row_index]]
            raw = eeg.values.flatten().tolist()
            # Pass row_index so the Telegram message says "Row X"
            make_prediction(eeg, raw, row_index=row_index)

# MANUAL INPUT
else:
    # Example data for quick testing
    example = "386,382,356" 
    txt = st.text_area("Enter 178 comma-separated values:", height=150)

    if st.button("Predict Manually"):
        try:
            vals = [float(v.strip()) for v in txt.split(",") if v.strip()]
            if len(vals) != 178:
                st.error(f"Expected 178 values, got {len(vals)}.")
            else:
                df = pd.DataFrame([vals], columns=[f"X{i}" for i in range(1, 179)])
                # No row index for manual input
                make_prediction(df, vals, row_index=None)
        except:
            st.error("Invalid input")

st.markdown("<div style='text-align:center; color:gray;'>Made with ❤️ — Epilepsy Detection System</div>", unsafe_allow_html=True)