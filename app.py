import time
from collections import deque
import joblib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

from core.camera import Camera
from core.face import FaceDetector
from core.kalman import KalmanFilter
from core.signals import chrom_recon, estimate_bpm_ensemble, pos_recon
from core.yawn_blink import detect_blink  # keep your working blink function
from utils.landmarks import eye_aspect_ratio, mouth_aspect_ratio

st.set_page_config(page_title="StressFusion-RT", layout="wide", initial_sidebar_state="collapsed")

@st.cache_resource
def load_detector():
    return FaceDetector("models/shape_predictor_68_face_landmarks.dat")

@st.cache_resource
def load_model():
    return joblib.load("models/stress_model.pkl")

def plot_bpm_graph(bpm_history, height=480):
    plt.close("all")
    plt.style.use("dark_background")
    dpi = 96
    fig_height_inches = height / dpi
    fig_width_inches = fig_height_inches * (4 / 3)
    fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))

    if bpm_history:
        ax.plot(bpm_history, color="red", linewidth=2)
        ax.set_ylim(40, 120)
        ax.set_xlim(0, len(bpm_history) + 50)
    else:
        ax.plot([], [])
        ax.set_ylim(40, 120)
        ax.set_xlim(0, 100)

    ax.set_title("Heart Rate (BPM) Over Time", color="white")
    ax.set_xlabel("Frames", color="white")
    ax.set_ylabel("BPM", color="white")
    ax.grid(True, color="gray", alpha=0.3)
    ax.tick_params(colors="white")
    fig.tight_layout()
    return fig

def count_events(timestamps, window_seconds=60):
    now = datetime.now()
    cutoff = now - timedelta(seconds=window_seconds)
    # Keep only events within the window
    while timestamps and timestamps[0] < cutoff:
        timestamps.popleft()
    return len(timestamps)

def main():
    st.markdown("""
    <style>
        .stApp { max-width: 80%; margin: auto; }
        .title-margin { margin-bottom: 1.5rem; }
        .stats-container { display: flex; font-size: 18px; margin-top: 20px; }
        .stats-container div { padding: 10px 20px; min-width: 140px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="title-margin">Real-Time rPPG + Yawn & Blink + Stress</h1>', unsafe_allow_html=True)

    kf = KalmanFilter(process_variance=1e-5, measurement_variance=1.0, initial_estimate=60)
    model = load_model()
    face_detector = load_detector()
    camera = Camera()

    r_signal, g_signal, b_signal = [], [], []
    chrom_signal, chrom_time = [], []
    pos_signal, pos_time = [], []

    bpm_history = deque(maxlen=100)
    stress_history = deque(maxlen=100)
    smoothed_bpm = None

    blink_frame_counter = 0
    blink_count_total = 0
    blink_in_progress = False
    prev_blink_count = 0

    prev_yawn_open = False

    # Use deques to store timestamps of blinks and yawns for moving window count
    blink_timestamps = deque()
    yawn_timestamps = deque()

    feature_names = ['hr', 'blink', 'yawn']

    video_col, graph_col = st.columns([1, 1])
    frame_placeholder = video_col.image(np.zeros((480, 640, 3), dtype=np.uint8), use_container_width=True)
    graph_placeholder = graph_col.pyplot(plot_bpm_graph([], height=480), clear_figure=True)
    stats_placeholder = st.empty()

    last_prediction_time = datetime.now()

    while True:
        ret, frame = camera.read()
        if not ret:
            st.warning("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detect(gray)

        if faces:
            face = max(faces, key=lambda r: r.width() * r.height())
            shape = face_detector.get_landmarks(gray, face)

            b, g, r, face_mask = face_detector.extract_face_roi_mean(frame, shape)
            overlay = frame.copy()
            overlay[face_mask == 255] = (0, 255, 0)
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

            r_signal.append(r)
            g_signal.append(g)
            b_signal.append(b)

            for i in range(68):
                x, y = shape.part(i).x, shape.part(i).y
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Yawn detection with debounce (rising edge detection)
            mar = mouth_aspect_ratio(shape)
            YAWN_THRESHOLD = 0.6  # Adjust if needed

            yawn_occurred = False
            if mar > YAWN_THRESHOLD:
                if not prev_yawn_open:
                    yawn_occurred = True
                prev_yawn_open = True
            else:
                prev_yawn_open = False

            if yawn_occurred:
                yawn_timestamps.append(datetime.now())

            # Blink detection
            left_ear = eye_aspect_ratio(shape, [36, 37, 38, 39, 40, 41])
            right_ear = eye_aspect_ratio(shape, [42, 43, 44, 45, 46, 47])
            ear = (left_ear + right_ear) / 2.0

            blink_frame_counter, current_blink_count = detect_blink(ear, blink_frame_counter, blink_count_total)

            if current_blink_count > prev_blink_count and not blink_in_progress:
                increment = current_blink_count - prev_blink_count
                for _ in range(increment):
                    blink_timestamps.append(datetime.now())
                blink_count_total += increment
                blink_in_progress = True
            elif current_blink_count == prev_blink_count:
                blink_in_progress = False

            prev_blink_count = current_blink_count

            # rPPG signal processing
            if len(r_signal) >= 40:
                r_arr = np.array(r_signal[-40:])
                g_arr = np.array(g_signal[-40:])
                b_arr = np.array(b_signal[-40:])
                chrom = chrom_recon(r_arr, g_arr, b_arr)[-1]
                pos = pos_recon(r_arr, g_arr, b_arr)[-1]
                chrom_signal.append(chrom)
                chrom_time.append(time.time())
                pos_signal.append(pos)
                pos_time.append(time.time())

            if len(chrom_signal) >= 40 and len(pos_signal) >= 40:
                bpm = estimate_bpm_ensemble(chrom_signal, chrom_time, pos_signal, pos_time)
                if bpm:
                    smoothed_bpm = kf.update_batch([bpm])[-1]
                    bpm_history.append(smoothed_bpm)

        now = datetime.now()
        # Continuous prediction every ~10 seconds
        if (now - last_prediction_time).total_seconds() >= 10 and smoothed_bpm is not None:
            blink_count_window = count_events(blink_timestamps)
            yawn_count_window = count_events(yawn_timestamps)

            features_df = pd.DataFrame([[smoothed_bpm, blink_count_window, yawn_count_window]], columns=feature_names)
            prediction = model.predict(features_df)[0]
            stress_history.append(prediction)

            last_prediction_time = now

        stress_display = str(stress_history[-1]) if stress_history else "Waiting..."
        bpm_display = f"{int(smoothed_bpm)}" if smoothed_bpm and not np.isnan(smoothed_bpm) else "Calculating..."
        blink_display = count_events(blink_timestamps)
        yawn_display = count_events(yawn_timestamps)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, use_container_width=True)
        graph_placeholder.pyplot(plot_bpm_graph(bpm_history, height=480), clear_figure=True)

        stats_placeholder.markdown(f"""
        <div class="stats-container">
            <div><b>Heart Rate (BPM)</b><br>{bpm_display}</div>
            <div><b>Yawn Count (per min)</b><br>{yawn_display}</div>
            <div><b>Blink Count (per min)</b><br>{blink_display}</div>
            <div><b>Stress Level</b><br>{stress_display}</div>
        </div>
        """, unsafe_allow_html=True)

    camera.release()

if __name__ == "__main__":
    main()
