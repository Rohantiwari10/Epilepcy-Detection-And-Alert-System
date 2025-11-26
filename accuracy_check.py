import pandas as pd
import numpy as np
import pywt
from hurst import compute_Hc
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. HELPER FUNCTIONS (Must match training!)
# ==========================================

def getHurst(df_copy):
    """Computes Hurst exponent for each signal."""
    # This might take a moment as it calculates for every row
    return [compute_Hc(df_copy.iloc[i], kind="change", simplified=True)[0] for i in range(len(df_copy))]

def statisticsForWavelet(coefs):
    """Calculates statistical features from wavelet coefficients."""
    n5 = np.nanpercentile(coefs, 5)
    n25 = np.nanpercentile(coefs, 25)
    n75 = np.nanpercentile(coefs, 75)
    n95 = np.nanpercentile(coefs, 95)
    median = np.nanpercentile(coefs, 50)
    mean = np.nanmean(coefs)
    std = np.nanstd(coefs)
    var = np.nanvar(coefs)
    rms = np.nanmean(np.sqrt(coefs**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]

def getWaveletFeatures(df_eeg, hurst_values):
    """Extracts Wavelet and Hurst features from the data."""
    features = []
    print(f"Extracting features from {len(df_eeg)} rows. This may take a minute...")
    for i in range(len(df_eeg)):
        # Show progress every 500 rows
        if i % 500 == 0:
            print(f"Processing row {i}/{len(df_eeg)}...")
            
        coeffs = pywt.wavedec(df_eeg.iloc[i], "db4")
        feat_row = [hurst_values[i]] # Start with Hurst
        for c in coeffs:
            feat_row += statisticsForWavelet(c) # Add Wavelet stats
        features.append(feat_row)
    return pd.DataFrame(features)

# ==========================================
# 2. MAIN EVALUATION LOGIC
# ==========================================

def evaluate_model():
    print("--- 🧠 Epilepsy Detection System: Accuracy Check ---")

    # 1. Load Data
    print("1. Loading dataset (data.csv)...")
    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError:
        print("Error: data.csv not found!")
        return

    # 2. Preprocess Labels (Ground Truth)
    # Original: 1=Seizure, 2-5=Normal
    # Our Model: 1=Seizure, 0=Normal
    y_true = df["y"].apply(lambda x: 1 if x == 1 else 0)
    
    # Drop label and index column to get raw EEG data
    if "Unnamed: 0" in df.columns:
        X_raw = df.drop(columns=["Unnamed: 0", "y"])
    else:
        X_raw = df.drop(columns=["y"])

    # 3. Feature Extraction
    # We must process the raw CSV data exactly how the model expects it
    print("2. Calculating Hurst Exponents...")
    hurst_values = getHurst(X_raw)
    
    print("3. extracting Wavelet Features...")
    X_features = getWaveletFeatures(X_raw, hurst_values)

    # 4. Load Model & Scaler
    print("4. Loading trained model and scaler...")
    try:
        model = joblib.load("rf_model.joblib")
        scaler = joblib.load("scaler.joblib")
    except FileNotFoundError:
        print("Error: rf_model.joblib or scaler.joblib not found. Train the model first.")
        return

    # 5. Scale Data
    print("5. Scaling features...")
    X_scaled = scaler.transform(X_features)

    # 6. Predict
    print("6. Running predictions...")
    y_pred = model.predict(X_scaled)

    # ==========================================
    # 3. RESULTS PRESENTATION
    # ==========================================
    
    print("\n" + "="*40)
    print("       MODEL PERFORMANCE REPORT       ")
    print("="*40)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ Overall Accuracy: {acc * 100:.2f}%")

    # Classification Report
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Seizure']))

    # Confusion Matrix Plot
    print("🎨 Generating Confusion Matrix Plot...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Seizure'], yticklabels=['Normal', 'Seizure'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix\nAccuracy: {acc*100:.2f}%')
    
    # Save plot
    plt.savefig("confusion_matrix.png")
    print("✅ Saved plot to 'confusion_matrix.png'")
    plt.show()

if __name__ == "__main__":
    evaluate_model()