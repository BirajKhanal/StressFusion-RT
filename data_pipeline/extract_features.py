import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

RAW_DATA_DIR = "datasets/raw_data/multi-peng/"
PROCESSED_DATA_PATH = "datasets/processed_data/"


def load_stress_data(stress_path):
    df = pd.read_csv(stress_path)
    selected_columns = [
        "participant_id",
        "session_no",
        "start_ts",
        "end_ts",
        "stress",
    ]

    stress_df = df[selected_columns].copy()
    return stress_df


def load_hr_data():
    samples_path = os.path.join(RAW_DATA_DIR, "Samples")
    selected_column = ["Timestamp", "BPM"]

    all_hr_data = []

    for participants_id in os.listdir(samples_path):
        hr_path = os.path.join(samples_path, participants_id, "HR")
        for participants_hr in os.listdir(hr_path):
            df = pd.read_csv(os.path.join(hr_path, participants_hr))
            hr_df = df[selected_column].copy()
            hr_df["participants_id"] = participants_id
            all_hr_data.append(hr_df)

    merged_hr_df = pd.concat(all_hr_data, ignore_index=True)
    return merged_hr_df


def load_blink_data():
    samples_path = os.path.join(RAW_DATA_DIR, "Samples")
    selected_column = ["Timestamp", "BKID", "BKPMIN"]

    all_blink_data = []

    for participants_id in os.listdir(samples_path):
        blink_path = os.path.join(samples_path, participants_id, "EYE")
        if not os.path.isdir(blink_path):
            continue
        for participants_hr in os.listdir(blink_path):
            df = pd.read_csv(os.path.join(blink_path, participants_hr))
            blink_df = df[selected_column].copy()
            blink_df["participants_id"] = participants_id
            all_blink_data.append(blink_df)

    merged_blink_df = pd.concat(all_blink_data, ignore_index=True)
    return merged_blink_df


def resample_to_minute(df, time_col, group_col, value_cols):
    # Ensure timestamp is datetime and valid
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    result = []

    for participant_id, group in df.groupby(group_col):
        group = group.set_index(time_col).sort_index()

        # Resample all value columns together
        resampled = group[value_cols].resample("1min").mean()
        resampled = resampled.reset_index()

        # Add participant ID back
        resampled[group_col] = participant_id

        # Drop rows with all NaNs in value_cols (i.e., no data during that minute)
        resampled = resampled.dropna(subset=value_cols, how="all")

        result.append(resampled)

    return pd.concat(result, ignore_index=True)


def assign_stress_labels(hr_blink_df, stress_df):
    # Convert all timestamp columns safely to datetime
    hr_blink_df["Timestamp"] = pd.to_datetime(
        hr_blink_df["Timestamp"], errors="coerce", utc=True
    )
    stress_df["start_ts"] = pd.to_datetime(
        stress_df["start_ts"], errors="coerce", utc=True
    )
    stress_df["end_ts"] = pd.to_datetime(
        stress_df["end_ts"], errors="coerce", utc=True
    )

    # Drop rows with invalid timestamps
    hr_blink_df = hr_blink_df.dropna(subset=["Timestamp"])
    stress_df = stress_df.dropna(subset=["start_ts", "end_ts"])

    # Ensure IDs are string
    hr_blink_df["participants_id"] = hr_blink_df["participants_id"].astype(str)
    stress_df["participant_id"] = stress_df["participant_id"].astype(str)

    # Floor timestamps to minute
    hr_blink_df["ts_min"] = hr_blink_df["Timestamp"].dt.floor("T")
    stress_df["start_min"] = stress_df["start_ts"].dt.floor("T")
    stress_df["end_min"] = stress_df["end_ts"].dt.floor("T")

    # Initialize stress column
    hr_blink_df["stress"] = None
    count = 0
    # Assign stress by comparing minute-level timestamp
    for idx, row in hr_blink_df.iterrows():
        count += 1
        print(f"\rCounting: {count}", end="", flush=True)
        pid = row["participants_id"]
        ts = row["ts_min"]

        match = stress_df[
            (stress_df["participant_id"] == pid)
            & (stress_df["start_min"] <= ts)
            & (stress_df["end_min"] >= ts)
        ]

        if not match.empty:
            hr_blink_df.at[idx, "stress"] = match.iloc[0]["stress"]

    return hr_blink_df.drop(columns=["ts_min"])


def main():
    stress_path = os.path.join(RAW_DATA_DIR, "Questionnaire", "submissions.csv")
    stress_df = load_stress_data(stress_path=stress_path)

    hr_df = load_hr_data()
    blink_df = load_blink_data()

    hr_minute = resample_to_minute(
        hr_df,
        time_col="Timestamp",
        group_col="participants_id",
        value_cols=["BPM"],
    )
    hr_minute["BPM"] = hr_minute["BPM"].round().astype("Int64")
    hr_minute.to_csv(PROCESSED_DATA_PATH + "hr.csv", index=False)

    blink_minute = resample_to_minute(
        blink_df,
        time_col="Timestamp",
        group_col="participants_id",
        value_cols=["BKID", "BKPMIN"],
    )
    blink_minute["BKID"] = blink_minute["BKID"].round().astype("Int64")
    blink_minute["BKPMIN"] = blink_minute["BKPMIN"].round().astype("Int64")
    blink_minute.to_csv(PROCESSED_DATA_PATH + "blink.csv", index=False)

    # Merge HR and Blink
    hr_blink_merged = pd.merge(
        hr_minute,
        blink_minute,
        on=["Timestamp", "participants_id"],
        how="outer",
    )
    hr_blink_merged.to_csv(PROCESSED_DATA_PATH + "merged.csv", index=False)

    print("LENGTH OF MERGED:", len(hr_blink_merged))

    final_df = assign_stress_labels(hr_blink_merged, stress_df)

    final_df.to_csv(PROCESSED_DATA_PATH + "combined_dataset.csv", index=False)


if __name__ == "__main__":
    main()
