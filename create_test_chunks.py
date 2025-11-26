import pandas as pd
import numpy as np
import os

# =================CONFIGURATION=================
# Your master data file containing all 11,500 rows
MASTER_FILE = 'data.csv'

# How many separate test files do you want to create?
NUM_FILES_TO_CREATE = 3

# How many rows should be in each test file?
ROWS_PER_FILE = 500
# ===============================================

def create_chunks():
    print(f"--- Starting process to create {NUM_FILES_TO_CREATE} new test files ---")

    # 1. Check if master file exists
    if not os.path.exists(MASTER_FILE):
        print(f"\nERROR: Could not find '{MASTER_FILE}'. Make sure it is in this folder.")
        return

    # 2. Load the massive dataset
    print(f"Reading master dataset: {MASTER_FILE}...")
    try:
        df = pd.read_csv(MASTER_FILE)
        print(f"Successfully loaded full dataset containing {len(df)} rows.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 3. Shuffle the entire dataset randomly
    # This is crucial so each new file gets a mix of different patients/states.
    # random_state ensures you get the same shuffle every time you run this script.
    print("Shuffling the entire dataset randomly...")
    df_shuffled = df.sample(frac=1, random_state=999).reset_index(drop=True)

    # 4. Slice off chunks and save them
    print(f"\nGenerating {NUM_FILES_TO_CREATE} files with {ROWS_PER_FILE} rows each...")

    start_index = 0
    for i in range(1, NUM_FILES_TO_CREATE + 1):
        end_index = start_index + ROWS_PER_FILE

        # Safety check: Stop if we run out of data in the master file
        if end_index > len(df_shuffled):
            print(f"Warning: Not enough remaining data to create file {i}. Stopping.")
            break

        # Slice the dataframe to get the chunk
        chunk_df = df_shuffled.iloc[start_index:end_index]

        # Define the new filename
        new_filename = f"test_chunk_{i}.csv"

        # Save to CSV. index=False is important to keep the format clean.
        chunk_df.to_csv(new_filename, index=False)
        print(f"-> Created file: '{new_filename}' (Rows {start_index} to {end_index})")

        # Move the starting point for the next loop
        start_index = end_index

    print("\n--- Process Complete ---")
    print(f"You now have {i-1 if end_index > len(df_shuffled) else i} new CSV files ready for testing.")

if __name__ == "__main__":
    create_chunks()