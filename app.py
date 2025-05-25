import time
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from core.camera import Camera
from core.face import FaceDetector
from core.kalman import KalmanFilter
from core.signals import chrom_recon, estimate_bpm_ensemble, pos_recon
from core.yawn_blink import detect_blink, detect_yawn
from utils.landmarks import eye_aspect_ratio, mouth_aspect_ratio

st.set_page_config(
    page_title="Real-Time rPPG + Yawn & Blink Detection",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_detector():
    return FaceDetector("models/shape_predictor_68_face_landmarks.dat")


def plot_bpm_graph(bpm_history, height=480):
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


def main():
    # Style
    st.markdown(
        """
        <style>
        .stApp {
            max-width: 80%;
            margin: auto;
        }
        .title-margin {
            margin-bottom: 1.5rem;
        }
        .stats-container {
            display: flex;
            font-size: 18px;
            margin-top: 20px;
        }
        .stats-container div {
            padding: 10px 20px;
            min-width: 140px;
            text-align: center;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<h1 class="title-margin">Real-Time rPPG + Yawn & Blink Detection</h1>',
        unsafe_allow_html=True,
    )

    kf = KalmanFilter(
        process_variance=1e-5, measurement_variance=1.0, initial_estimate=60
    )

    yawn_count = 0
    yawn_open = False

    blink_count = 0
    blink_frame_counter = 0

    r_signal, g_signal, b_signal = [], [], []
    chrom_signal, chrom_time = [], []
    pos_signal, pos_time = [], []

    bpm_history = deque(maxlen=100)
    smoothed_bpm = None

    face_detector = load_detector()
    camera = Camera()

    # Layout: image and graph side by side
    video_col, graph_col = st.columns([1, 1])
    frame_placeholder = video_col.image(
        np.zeros((480, 640, 3), dtype=np.uint8), use_container_width=True
    )
    graph_placeholder = graph_col.pyplot(
        plot_bpm_graph([], height=480), clear_figure=True
    )

    stats_placeholder = st.empty()

    while True:
        ret, frame = camera.read()
        if not ret:
            st.warning("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detect(gray)

        if faces:
            face = max(faces, key=lambda r: r.width() * r.height())
            shape = face_detector.get_landmarks(gray, face)

            b, g, r, face_mask = face_detector.extract_face_roi_mean(
                frame, shape
            )
            overlay = frame.copy()
            overlay[face_mask == 255] = (0, 255, 0)
            alpha = 0.1
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            r_signal.append(r)
            g_signal.append(g)
            b_signal.append(b)

            for i in range(68):
                x, y = shape.part(i).x, shape.part(i).y
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            if len(r_signal) >= 30:
                r_arr = np.array(r_signal[-30:])
                g_arr = np.array(g_signal[-30:])
                b_arr = np.array(b_signal[-30:])

                chrom = chrom_recon(r_arr, g_arr, b_arr)[-1]
                pos = pos_recon(r_arr, g_arr, b_arr)[-1]
                chrom_signal.append(chrom)
                chrom_time.append(time.time())
                pos_signal.append(pos)
                pos_time.append(time.time())

            mar = mouth_aspect_ratio(shape)
            yawn_open, yawn_count = detect_yawn(mar, yawn_open, yawn_count)

            left_ear = eye_aspect_ratio(shape, [36, 37, 38, 39, 40, 41])
            right_ear = eye_aspect_ratio(shape, [42, 43, 44, 45, 46, 47])
            ear = (left_ear + right_ear) / 2.0

            blink_frame_counter, blink_count = detect_blink(
                ear, blink_frame_counter, blink_count
            )

            bpm = estimate_bpm_ensemble(
                chrom_signal, chrom_time, pos_signal, pos_time
            )
            if bpm:
                smoothed_bpm = kf.update_batch([bpm])[-1]
                bpm_history.append(smoothed_bpm)

        # Update video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, use_container_width=True)

        # Update graph
        graph_placeholder.pyplot(
            plot_bpm_graph(bpm_history, height=480), clear_figure=True
        )

        # Update stats
        bpm_display = (
            str(int(smoothed_bpm))
            if smoothed_bpm and not np.isnan(smoothed_bpm)
            else "Calculating..."
        )
        stats_placeholder.markdown(
            f"""
            <div class="stats-container">
                <div><b>Heart Rate (BPM)</b><br>{bpm_display}</div>
                <div><b>Yawn Count</b><br>{yawn_count}</div>
                <div><b>Blink Count</b><br>{blink_count}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    camera.release()


if __name__ == "__main__":
    main()
