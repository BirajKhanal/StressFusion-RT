# import glob
# import os
# import pickle
#
# import numpy as np
# import pandas as pd
# from scipy.signal import resample
# from tqdm import tqdm
#
# # Set your WESAD data root folder here
# root_dir = "datasets/raw_data/WESAD/"
# all_data = []
#
# pkl_files = glob.glob(os.path.join(root_dir, "S*/S*.pkl"))
#
# for pkl_path in tqdm(pkl_files, desc="Processing participants"):
#     try:
#         with open(pkl_path, "rb") as f:
#             data = pickle.load(f, encoding="latin1")
#
#         subject_id = data["subject"]
#         bvp = data["signal"]["wrist"]["BVP"]  # 64 Hz
#         labels = data["label"]  # 700 Hz
#
#         # Convert nested arrays (if any) to flat 1D array
#         bvp = np.array(bvp).squeeze()
#         if bvp.ndim != 1:
#             raise ValueError("BVP signal is not 1D even after squeezing")
#
#         # Downsample both to 1 Hz for simplicity
#         target_length = min(len(bvp) // 64, len(labels) // 700)
#         bvp_resampled = resample(bvp, target_length)
#         label_resampled = (
#             resample(labels.astype(int), target_length).round().astype(int)
#         )
#
#         df = pd.DataFrame(
#             {
#                 "subject": [subject_id] * target_length,
#                 "label": label_resampled,
#                 "bvp": bvp_resampled,
#             }
#         )
#         all_data.append(df)
#
#     except Exception as e:
#         print(f"❌ Error processing {pkl_path}: {e}")
#
# # Combine and save
# if all_data:
#     combined_df = pd.concat(all_data, ignore_index=True)
#     combined_df.to_csv("wesad_bvp_labels.csv", index=False)
#     print("✅ Saved combined CSV as 'wesad_bvp_labels.csv'")
# else:
#     print("❌ No valid data was processed.")

import glob
import os
import pickle

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, resample  # Ensure find_peaks is imported
from tqdm import tqdm

# --- Settings ---
root_dir = "datasets/raw_data/WESAD/"
bvp_freq = 64  # BVP sampling rate
window_sec = 8  # Window size for HR calculation
step_sec = 1  # Step size for HR calculation
all_data = []


# --- HR Extraction Function ---
def extract_hr_from_bvp(bvp, fs, window_sec=8, step_sec=1):
    """
    Extracts Heart Rate (HR) from Blood Volume Pulse (BVP) data using a sliding window.

    Args:
        bvp (np.array): BVP signal (1D array).
        fs (int): Sampling frequency of the BVP signal.
        window_sec (int): Sliding window size in seconds for HR calculation.
        step_sec (int): Step size in seconds for sliding the window.

    Returns:
        pd.DataFrame: DataFrame with 'timestamp' (in seconds) and 'bpm' (beats per minute).
    """
    hr_list = []
    timestamps = []
    win_size = window_sec * fs
    step_size = step_sec * fs

    # Iterate through the BVP signal with a sliding window
    for i in range(0, len(bvp) - win_size + 1, step_size):
        segment = bvp[i : i + win_size]
        segment = segment - np.mean(segment)  # Remove DC component

        # Find peaks in the BVP segment, which represent heartbeats
        # 'distance=fs*0.5' ensures peaks are at least 0.5 seconds apart
        peaks, _ = find_peaks(segment, distance=fs * 0.5)

        if len(peaks) > 1:
            # Calculate R-R intervals (time between consecutive heartbeats)
            rr_intervals = np.diff(peaks) / fs
            # Convert R-R intervals to Beats Per Minute (BPM)
            bpm = 60 / np.mean(rr_intervals)
            bpm = int(np.round(bpm))
        else:
            bpm = (
                np.nan
            )  # If not enough peaks, HR cannot be reliably calculated
        hr_list.append(bpm)
        timestamps.append(i // fs)  # Store time in seconds

    return pd.DataFrame({"timestamp": timestamps, "bpm": hr_list})


# --- Load and Process All Participants ---
pkl_files = glob.glob(os.path.join(root_dir, "S*", "S*.pkl"))

for pkl_path in tqdm(pkl_files, desc="Processing participants"):
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        subject_id = data["subject"]
        bvp = np.array(data["signal"]["wrist"]["BVP"]).squeeze()
        labels = data["label"]  # Labels are at 700 Hz, will be resampled later

        # Ensure BVP is a 1D array
        if bvp.ndim != 1:
            print(
                f"⚠️ Warning: BVP data for {subject_id} is not 1D after squeezing. Skipping."
            )
            continue

        # 1. Extract HR from BVP
        hr_df = extract_hr_from_bvp(bvp, bvp_freq, window_sec, step_sec)

        # The timestamps in hr_df are based on the BVP signal's original seconds.
        # Now, we need to align labels with these HR timestamps.
        # The HR values are effectively at 1 Hz (since step_sec is 1).
        # We need to resample the original 700 Hz labels to match the length of hr_df.

        # 2. Resample labels to align with HR data
        # 'resample' interpolates the original labels to the new target length
        label_resampled = resample(labels, len(hr_df))
        label_resampled = np.round(label_resampled).astype(int)

        # Add participant ID and resampled label to the HR DataFrame
        hr_df["participant"] = subject_id
        hr_df["label"] = label_resampled

        # Filter out unwanted labels, keeping only baseline (1) and stress (2)
        hr_df = hr_df[hr_df["label"].isin([1, 2, 3, 4])]

        all_data.append(hr_df)

    except Exception as e:
        print(f"❌ Error processing {pkl_path}: {e}")

# --- Combine and Save ---
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(
        "datasets/processed_data/wesad_hr_baseline_stress.csv", index=False
    )
    print(
        "✅ Saved combined HR data for baseline and stress as 'wesad_hr_baseline_stress.csv'"
    )
else:
    print("❌ No valid data was processed.")
