# app.py — Epilepsy Monitoring & Alert System

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

# Optional imports (email/SMS)
try:
    import smtplib
    from email.mime.text import MIMEText
except:
    smtplib = None

try:
    from twilio.rest import Client as TwilioClient
except:
    TwilioClient = None


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
# ALERT: SOUND
# ------------------------------------------------
def play_alarm_sound():
    if not os.path.exists("alert.wav"):
        st.warning("⚠️ alert.wav missing! Place an alert sound in project folder.")
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
# ALERT: EMAIL
# ------------------------------------------------
def send_email(sender, password, receiver, message):
    if smtplib is None:
        st.error("SMTP not supported environment.")
        return False
    try:
        msg = MIMEText(message)
        msg["From"] = sender
        msg["To"] = receiver
        msg["Subject"] = "🚨 Seizure Alert"

        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email error: {e}")
        return False


# ------------------------------------------------
# ALERT: SMS
# ------------------------------------------------
def send_sms(tw_sid, tw_token, sender_no, receiver_no, msg):
    if TwilioClient is None:
        st.error("Twilio not installed.")
        return False

    try:
        client = TwilioClient(tw_sid, tw_token)
        client.messages.create(body=msg, from_=sender_no, to=receiver_no)
        return True
    except Exception as e:
        st.error(f"SMS error: {e}")
        return False


# ------------------------------------------------
# MASTER ALERT FUNCTION
# ------------------------------------------------
def trigger_alert():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert_message = f"⚠️ Seizure detected at {timestamp}"

    # Alarm
    if enable_sound:
        play_alarm_sound()

    # Email
    if enable_email and email_sender and email_password and email_receiver:
        if send_email(email_sender, email_password, email_receiver, alert_message):
            st.success("📧 Email alert sent")

    # SMS
    if enable_sms and tw_sid and tw_token and tw_from and tw_to:
        if send_sms(tw_sid, tw_token, tw_from, tw_to, alert_message):
            st.success("📱 SMS alert sent")


# ------------------------------------------------
# LOAD MODEL + SCALER
# ------------------------------------------------
try:
    model = joblib.load("rf_model.joblib")
    scaler = joblib.load("scaler.joblib")
except:
    st.error("Model or scaler file missing!")
    st.stop()


# ------------------------------------------------
# SIDEBAR SETTINGS
# ------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Alert Controls")

    enable_sound = st.checkbox("Play Alarm", True)
    enable_email = st.checkbox("Send Email Alert", False)
    enable_sms = st.checkbox("Send SMS Alert", False)

    st.markdown("---")

    if enable_email:
        email_sender = st.text_input("Sender Email")
        email_password = st.text_input("Email App Password", type="password")
        email_receiver = st.text_input("Receiver Email")
    else:
        email_sender = email_password = email_receiver = None

    if enable_sms:
        tw_sid = st.text_input("Twilio SID")
        tw_token = st.text_input("Twilio Token", type="password")
        tw_from = st.text_input("Twilio Number (+1...)")
        tw_to = st.text_input("Receiver Number (+91...)")
    else:
        tw_sid = tw_token = tw_from = tw_to = None


# ------------------------------------------------
# PREDICTION FUNCTION
# ------------------------------------------------
def make_prediction(df, raw=None):

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

        trigger_alert()   # 🔥 Master Alert

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
st.write("Upload EEG CSV or enter values manually to detect seizures.")
st.markdown("</div>", unsafe_allow_html=True)

input_choice = st.sidebar.radio("Input Method:", ["Upload CSV", "Manual Input"])

# CSV INPUT
# CSV INPUT
if input_choice == "Upload CSV":
    file = st.sidebar.file_uploader("Upload CSV (X1..X178 required)")

    if file:
        df = pd.read_csv(file)

        st.write("### Data Preview:")
        st.dataframe(df.head())

        # --------------------------------------------
        # 🔥 Slider to choose row index
        # --------------------------------------------
        row_index = st.sidebar.slider(
            "Select Row for Prediction",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            step=1
        )

        st.sidebar.write(f"Selected Row: **{row_index}**")

        # Show selected row data
        st.write("### Selected EEG Row")
        st.dataframe(df.iloc[[row_index]])

        # --------------------------------------------
        # 🔥 Predict button
        # --------------------------------------------
        if st.sidebar.button("Predict Selected Row"):
            eeg = df.select_dtypes(include=np.number).iloc[[row_index]]
            raw = eeg.values.flatten().tolist()
            make_prediction(eeg, raw)


# MANUAL INPUT
else:
    example = "386,382,356,...(178 values)"
    txt = st.text_area("Enter 178 comma-separated values:", value=example, height=150)

    if st.button("Predict Manually"):
        try:
            vals = [float(v.strip()) for v in txt.split(",") if v.strip()]
            if len(vals) != 178:
                st.error(f"Expected 178 values, got {len(vals)}.")
            else:
                df = pd.DataFrame([vals], columns=[f"X{i}" for i in range(1, 179)])
                make_prediction(df, vals)
        except:
            st.error("Invalid input")


# FOOTER

st.markdown("<div style='text-align:center; color:gray;'>Made with ❤️ — Epilepsy Detection System</div>", unsafe_allow_html=True)
