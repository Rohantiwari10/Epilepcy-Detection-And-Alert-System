import os
import base64
import streamlit as st
from datetime import datetime

# Optional imports
try:
    import smtplib
    from email.mime.text import MIMEText
except:
    smtplib = None

try:
    from twilio.rest import Client as TwilioClient
except:
    TwilioClient = None


# -----------------------
# 1. PLAY ALARM SOUND
# -----------------------
def play_alarm_sound(alert_wav_path="alert.wav"):
    """Plays alarm sound if alert.wav exists."""
    if not os.path.exists(alert_wav_path):
        st.warning("⚠️ alert.wav not found! Add it to project folder.")
        return False

    try:
        with open(alert_wav_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        audio_html = f"""
        <audio autoplay="true">
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
        """
        st.components.v1.html(audio_html, height=0)
        return True
    except Exception as e:
        st.error(f"Alarm sound error: {e}")
        return False


# -----------------------
# 2. EMAIL ALERT
# -----------------------
def send_email_alert(sender, password, receiver, message):
    if smtplib is None:
        st.error("SMTP library not available.")
        return False

    try:
        msg = MIMEText(message)
        msg["Subject"] = "🚨 Seizure Alert"
        msg["From"] = sender
        msg["To"] = receiver

        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()

        return True

    except Exception as e:
        st.error(f"Email sending error: {e}")
        return False


# -----------------------
# 3. SMS ALERT (TWILIO)
# -----------------------
def send_sms_alert(tw_sid, tw_token, sender_no, receiver_no, message):
    if TwilioClient is None:
        st.error("Twilio library not installed.")
        return False

    try:
        client = TwilioClient(tw_sid, tw_token)
        client.messages.create(
            body=message,
            from_=sender_no,
            to=receiver_no,
        )
        return True

    except Exception as e:
        st.error(f"SMS sending error: {e}")
        return False


# -----------------------
# 4. MASTER FUNCTION
# -----------------------
def trigger_alert(enable_sound, enable_email, enable_sms,
                  email_sender=None, email_pass=None, email_recv=None,
                  tw_sid=None, tw_token=None, tw_from=None, tw_to=None):
    """Call this function whenever seizure is detected"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert_message = f"🚨 Seizure detected at {timestamp}"

    # SOUND
    if enable_sound:
        play_alarm_sound()

    # EMAIL
    if enable_email and email_sender and email_pass and email_recv:
        if send_email_alert(email_sender, email_pass, email_recv, alert_message):
            st.success("📧 Email alert sent!")

    # SMS
    if enable_sms and tw_sid and tw_token and tw_from and tw_to:
        if send_sms_alert(tw_sid, tw_token, tw_from, tw_to, alert_message):
            st.success("📱 SMS alert sent!")
