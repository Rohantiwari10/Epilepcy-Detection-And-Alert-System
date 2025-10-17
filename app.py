# app.py (Updated with Manual Input Option)

import streamlit as st
import pandas as pd
import numpy as np
import pywt
from hurst import compute_Hc
import joblib
import smtplib

st.set_page_config(page_title="Epileptic Seizure Detection", page_icon="🧠", layout="wide")

# --- Helper Functions (No changes here) ---
def getHurst(df_eeg):
    hurst_values = [compute_Hc(df_eeg.iloc[i], kind="change", simplified=True)[0] for i in range(len(df_eeg))]
    return hurst_values

def statisticsForWavelet(coefs):
    n5, n25, n75, n95 = np.nanpercentile(coefs, [5, 25, 75, 95])
    median, mean, std, var = np.nanpercentile(coefs, 50), np.nanmean(coefs), np.nanstd(coefs), np.nanvar(coefs)
    rms = np.nanmean(np.sqrt(coefs**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]

def getWaveletFeatures(df_eeg, hurst_values):
    list_features = []
    for i in range(len(df_eeg)):
        list_coeff = pywt.wavedec(df_eeg.iloc[i], "db4")
        features = [hurst_values[i]]
        for coeff in list_coeff:
            features += statisticsForWavelet(coeff)
        list_features.append(features)
    return pd.DataFrame(list_features)
    
def send_alert_email():
    # Placeholder for email functionality
    st.warning("🚨 Seizure Detected! Alert Email Sent (simulation).", icon="⚠️")

# --- Load Pre-trained Model and Scaler ---
try:
    model = joblib.load('rf_model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("Model or scaler not found. Please run `train_model.py` first.")
    st.stop()

# --- Prediction Function ---
def make_prediction(eeg_data_df):
    with st.spinner('Analyzing EEG signal...'):
        # 1. Feature Engineering
        hurst_val = getHurst(eeg_data_df)
        final_features = getWaveletFeatures(eeg_data_df, hurst_val)

        # 2. Scaling
        scaled_features = scaler.transform(final_features)

        # 3. Prediction
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)

    st.header("Prediction Result")
    if prediction[0] == 1:
        st.error("RESULT: Epileptic Seizure Detected", icon="🚨")
        # send_alert_email()
    else:
        st.success("RESULT: Normal Brain Activity Detected", icon="✅")

    st.write("### Prediction Confidence")
    col1, col2 = st.columns(2)
    col1.metric("Normal Activity", f"{prediction_proba[0][0]*100:.2f}%")
    col2.metric("Seizure Activity", f"{prediction_proba[0][1]*100:.2f}%")


# --- UI Layout ---
st.title("🧠 Epileptic Seizure Detection System")
st.write("This application uses a pre-trained model to detect epileptic seizures from EEG signals.")

st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose how to provide data:", ("Upload CSV File", "Enter Single Data Point Manually"))

if input_method == "Upload CSV File":
    st.sidebar.subheader("Upload EEG Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.header("Uploaded EEG Data Preview")
        st.dataframe(input_df.head())
        
        eeg_data = input_df.loc[:, 'X1':'X178']
        if eeg_data.shape[1] != 178:
            st.error(f"Error: Could not find 178 feature columns named X1 to X178.")
        else:
            st.sidebar.write("---")
            row_to_predict = st.sidebar.slider("Select a Signal (Row Number) to Predict", 0, len(eeg_data) - 1, 0)
            
            if st.sidebar.button("Detect Seizure from Row", use_container_width=True):
                single_row_df = eeg_data.iloc[[row_to_predict]]
                make_prediction(single_row_df)
    else:
        st.info("Awaiting CSV file upload.")

else: # Manual Input
    st.sidebar.subheader("Paste EEG Data")
    st.header("Enter a Single EEG Data Point")
    
    # Example data point to guide the user
    example_seizure_data = "386,382,356,331,320,315,307,272,244,232,237,258,212,2, -129,-222,-222,-222,-222,-221,-208,-181,-154,-125,-79,-28,21,69,111,142,168,189,208,230,251,270,289,309,326,346,360,371,382,391,392,392,392,392,392,392,392,392,392,388,381,370,356,342,326,312,300,288,278,268,261,254,249,245,241,238,235,232,229,225,221,216,210,204,197,192,189,188,189,190,192,192,190,188,184,180,174,169,164,161,160,161,164,169,176,184,193,203,213,223,232,240,246,251,255,258,263,269,277,286,293,298,300,302,303,304,303,302,301,301,300,300,299,297,295,293,291,288,285,280,273,264,252,237,220,201,183,168,155,145,137,131,128,126,126,127,130,133,137,142,148,153,158,162,164,150,146,152,157,156,154,143,129"
    
    data_point_str = st.text_area(
        "Paste the 178 comma-separated EEG signal values below:", 
        height=250, 
        placeholder=example_seizure_data
    )
    
    if st.button("Detect Seizure from Manual Input", use_container_width=True):
        if data_point_str:
            try:
                # Convert the string to a list of numbers
                values = [float(v.strip()) for v in data_point_str.split(',')]
                
                if len(values) == 178:
                    # Create a single-row DataFrame
                    column_names = [f'X{i}' for i in range(1, 179)]
                    manual_df = pd.DataFrame([values], columns=column_names)
                    
                    # Make the prediction
                    make_prediction(manual_df)
                else:
                    st.error(f"Error: You entered {len(values)} values, but exactly 178 are required.")
            except ValueError:
                st.error("Error: Could not process the input. Please ensure all values are numbers and separated by commas.")
        else:
            st.warning("Please paste data into the text area before clicking detect.")