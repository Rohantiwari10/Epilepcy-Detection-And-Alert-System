import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os

# --- IMPORT FEATURE ENGINEERING FUNCTION ---
# This imports the 'produce_features' function from your other Python file.
# It's the same math used during training.
try:
    from epileptic_seizure_detection import produce_features
except ImportError:
    print("\nCRITICAL ERROR: Could not find 'epileptic_seizure_detection.py'.")
    print("Make sure this script is in the same folder as 'epileptic_seizure_detection.py'.")
    sys.exit(1)

def test_model_on_data(filename):
    print(f"\n--- Starting Test Process for file: '{filename}' ---")

    # 1. Check if file exists
    if not os.path.exists(filename):
        print(f"ERROR: The data file '{filename}' was not found.")
        return

    # 2. Load the Saved 'Brain' (Model) and 'Translator' (Scaler)
    print("Loading saved model and scaler...")
    try:
        model = joblib.load('rf_model.joblib')
        scaler = joblib.load('scaler.joblib')
        print("-> Success: Model and Scaler loaded.")
    except FileNotFoundError:
        print("\nCRITICAL ERROR: Could not find 'rf_model.joblib' or 'scaler.joblib'.")
        print("Make sure these files are in the same folder as this script.")
        sys.exit(1)

    # 3. Load the Data
    print(f"Loading data from CSV...")
    df = pd.read_csv(filename)

    # Basic cleaning: remove the index column if it's there
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    # Separate Features (X) from True Labels (y)
    if 'y' not in df.columns:
         print("ERROR: Your data file must have a column named 'y' for the true labels.")
         return
    X_raw = df.drop('y', axis=1)
    y_true = df['y']

    print(f"-> Data loaded. Testing on {len(df)} examples.")

    # ============================================================
    # CORE PROCESS: REPLICATING THE TRAINING PIPELINE
    # ============================================================

    # Step 4a: Feature Engineering (Apply the complex math)
    print("1/3 Applying Feature Engineering (Wavelets, etc.)...")
    # We pass the raw numerical values to the function
    X_features = produce_features(X_raw.values)

    # Step 4b: Scaling (Use the saved translator)
    print("2/3 Scaling features using the saved scaler...")
    # IMPORTANT: We use .transform(), not .fit_transform(). We use the training data's scale.
    X_scaled = scaler.transform(X_features)

    # Step 4c: Make Predictions
    print("3/3 Asking the model for predictions...")
    y_pred = model.predict(X_scaled)

    # ============================================================
    # REPORTING RESULTS
    # ============================================================

    # Calculate Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    print("\n" + "="*50)
    print(f"   TEST RESULTS FOR: {filename}")
    print("="*50)
    print(f"\n>>> FINAL ACCURACY: {accuracy * 100:.2f}% <<<")
    print("\n" + "-"*50)
    print("Detailed Classification Report:")
    print("Precision: Accuracy of positive predictions.")
    print("Recall: Fraction of positives that were correctly identified.")
    print("-" * 50)
    # Check unique labels to format report correctly
    labels = sorted(y_true.unique())
    print(classification_report(y_true, y_pred, labels=labels))

# ==========================================
# HOW TO RUN: Change the filename below
# ==========================================

# To test on your original data, keep this line:
test_model_on_data('data.csv')

# To test on a NEW file, comment out the line above and uncomment the one below:
# test_model_on_data('my_new_unseen_data.csv')
