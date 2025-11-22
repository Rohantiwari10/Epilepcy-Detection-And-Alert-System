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
import requests  # For Telegram
import smtplib   # For Email
from email.mime.text import MIMEText # For Email body
from dotenv import load_dotenv

# ------------------------------------------------
# HYBRID KEY MANAGEMENT (Cloud + Local)
# ------------------------------------------------
# 1. Load local .env file (if on local machine)
load_dotenv()

# 2. Helper function to get keys from either Cloud Secrets OR Local .env
# ------------------------------------------------
# HYBRID KEY MANAGEMENT (Cloud + Local)
# ------------------------------------------------
load_dotenv()

def get_secret(key):
    # 1. Priority: Check Local .env first (Best for your laptop)
    value = os.getenv(key)
    if value:
        return value
    
    # 2. Fallback: Check Streamlit Cloud Secrets
    # We use try-except to prevent crashing locally if secrets.toml is missing
    try:
        if key in st.secrets:
            return st.secrets[key]
    except:
        return None # If secrets file is missing, just return None
    
    return None

# 3. Fetch keys using the helper
TELEGRAM_TOKEN = get_secret("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = get_secret("TELEGRAM_CHAT_ID")
EMAIL_SENDER = get_secret("EMAIL_SENDER")
EMAIL_PASSWORD = get_secret("EMAIL_PASSWORD")
EMAIL_RECEIVER = get_secret("EMAIL_RECEIVER")

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
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        st.error("⚠️ Telegram keys missing! Check Secrets/Env.")
        return

    # Support multiple Chat IDs (comma separated)
    chat_ids = [id.strip() for id in str(TELEGRAM_CHAT_ID).split(",") if id.strip()]
    success_count = 0
    
    for chat_id in chat_ids:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML" 
        }
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                success_count += 1
            else:
                print(f"Failed Telegram to {chat_id}: {response.text}")
        except Exception as e:
            print(f"Telegram connection error: {e}")

    if success_count > 0:
        st.success(f"📲 Telegram sent to {success_count} recipient(s)!")

# ------------------------------------------------
# EMAIL ALERT FUNCTION
# ------------------------------------------------
def send_email_alert(subject, body):
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        st.error("⚠️ Email keys missing! Check Secrets/Env.")
        return

    # Support multiple Receivers (comma separated)
    receivers = [email.strip() for email in str(EMAIL_RECEIVER).split(",") if email.strip()]

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = ", ".join(receivers)

    try:
        # Try Port 587 (TLS) - Safest for Cloud/College Wi-Fi
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, receivers, msg.as_string())
        server.quit()
        
        st.success(f"📧 Email sent to {len(receivers)} recipient(s)!")
    except Exception as e:
        st.error(f"Email Sending Failed: {e}")

# ------------------------------------------------
# MASTER ALERT FUNCTION
# ------------------------------------------------
def trigger_alert(row_index=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Prepare Telegram Message (HTML)
    telegram_msg = f"🚨 <b>SEIZURE DETECTED</b>\n"
    telegram_msg += f"🕒 Time: {timestamp}\n"
    if row_index is not None:
        telegram_msg += f"📊 Data Source: <b>Row {row_index + 1}</b>"
    else:
        telegram_msg += "📊 Data Source: <b>Manual Input</b>"
    telegram_msg += "\n\n❗ <b>Patient requires immediate attention.</b>"

    # 2. Prepare Email Message (Text)
    email_subject = "🚨 URGENT: Seizure Detected"
    email_body = f"""
    URGENT MEDICAL ALERT
    --------------------
    A Seizure event has been detected.
    
    Time: {timestamp}
    Source: {f'Row {row_index + 1}' if row_index is not None else 'Manual Input'}
    
    Please check the patient immediately.
    """

    # 3. Trigger Selected Alerts
    if enable_sound:
        play_alarm_sound()

    if enable_mobile_alert:
        send_telegram_alert(telegram_msg)

    if enable_email_alert:
        send_email_alert(email_subject, email_body)

# ------------------------------------------------
# LOAD MODEL + SCALER
# ------------------------------------------------
try:
    model = joblib.load("rf_model.joblib")
    scaler = joblib.load("scaler.joblib")
except:
    st.error("Model files missing! Upload rf_model.joblib and scaler.joblib.")
    st.stop()

# ------------------------------------------------
# SIDEBAR SETTINGS
# ------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Alert Controls")

    enable_sound = st.checkbox("Play Sound Alarm", True)
    enable_mobile_alert = st.checkbox("Send Telegram Alert", True)
    enable_email_alert = st.checkbox("Send Email Alert", False)

    st.divider()
    st.markdown("### Connection Status")
    
    if TELEGRAM_TOKEN:
        st.success("Telegram: Connected")
    else:
        st.error("Telegram: Missing Key")
        
    if EMAIL_SENDER:
        st.success("Email: Connected")
    else:
        st.warning("Email: Missing Key")

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
            make_prediction(eeg, raw, row_index=row_index)

# MANUAL INPUT
else:
    # We provide a full 178-value example so you don't have to type it
    st.info("Tip: You need exactly 178 values.")
    txt = st.text_area("Enter 178 comma-separated values:", height=150)

    if st.button("Predict Manually"):
        try:
            # Convert string input to list of floats
            vals = [float(v.strip()) for v in txt.split(",") if v.strip()]
            
            # 1. CHECK LENGTH FIRST
            if len(vals) != 178:
                st.error(f"❌ Error: Expected 178 values, but got {len(vals)}.")
                st.warning("Please paste a full row of CSV data.")
            
            # 2. ONLY PROCEED IF LENGTH IS CORRECT
            else:
                df = pd.DataFrame([vals], columns=[f"X{i}" for i in range(1, 179)])
                # Correct indentation - this only runs if length is 178
                make_prediction(df, vals, row_index=None)
                
        except ValueError:
            st.error("Invalid input: Make sure you only enter numbers separated by commas.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.markdown("<div style='text-align:center; color:gray;'>Made with ❤️ — Epilepsy Detection System</div>", unsafe_allow_html=True)