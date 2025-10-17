# train_model.py

import pandas as pd
import numpy as np
import pywt
from hurst import compute_Hc
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import joblib

print("Starting model training process...")

# --- All the helper functions from your notebook ---

def prepareData(df):
    """Cleans data and creates a binary target variable."""
    # Create binary target: 1 for seizure, 0 for not
    df["y"] = [1 if y_val == 1 else 0 for y_val in df["y"]]
    target = df["y"]
    # Drop unnecessary columns
    if "Unnamed: 0" in df.columns:
        df_copy = df.drop(["Unnamed: 0", "y"], axis=1)
    else: # If the file uploaded doesn't have it
        df_copy = df.drop(["y"], axis=1)
    return df_copy, target

def getHurst(df_copy):
    """Computes Hurst exponent for each signal."""
    df_copy["hurst_ex"] = [compute_Hc(df_copy.iloc[i], kind="change", simplified=True)[0] for i in range(len(df_copy))]
    return df_copy

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

def getWaveletFeatures(data):
    """Extracts Wavelet and Hurst features from the data."""
    list_features = []
    for signal in range(len(data)):
        list_coeff = pywt.wavedec(data.iloc[signal], "db4")
        features = []
        features.append(data.iloc[signal]["hurst_ex"]) # Hurst exponent
        for coeff in list_coeff:
            features += statisticsForWavelet(coeff)
        list_features.append(features)
    return pd.DataFrame(list_features)

# --- Main Training Logic ---

# 1. Load Data
df = pd.read_csv("data.csv")

# 2. Prepare Data
df_copy, target = prepareData(df)

# 3. Feature Engineering (Hurst + Wavelet)
print("Performing feature engineering...")
df_with_hurst = getHurst(df_copy)
# Drop the original time series columns, keeping only hurst_ex for the wavelet function
df_final_features = getWaveletFeatures(df_with_hurst[['hurst_ex'] + list(df_copy.columns[:-1])])
df_final_features['target'] = target

# 4. Create a balanced dataset for training
print("Balancing dataset...")
X_shuffled = shuffle(df_final_features, random_state=42)
balanced_df = X_shuffled.sort_values(by='target', ascending=False).iloc[:6500]
X_train = balanced_df.drop('target', axis=1)
y_train = balanced_df['target']

# 5. Normalize Data and save the scaler
print("Normalizing data...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'scaler.joblib')
print("Scaler saved to scaler.joblib")

# 6. Train the Random Forest Model
print("Training Random Forest model...")
# Using the best parameters from your notebook
rf_clf = RandomForestClassifier(random_state=42, max_depth=5, max_features='sqrt', min_samples_split=5, n_estimators=150)
rf_clf.fit(X_train_scaled, y_train)

# 7. Save the trained model
joblib.dump(rf_clf, 'rf_model.joblib')
print("Model training complete and saved to rf_model.joblib!")