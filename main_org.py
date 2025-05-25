import time

import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, filtfilt

# ——— Init ———
detector = dlib.get_frontal_face_detector()  # type: ignore
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # type: ignore

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ——— Buffers ———
r_signal, g_signal, b_signal = [], [], []
chrom_signal, chrom_time = [], []
pos_signal, pos_time = [], []
raw_bpm_values = []
smooth_bpm_values = []

# ——— Yawn Configs ———
yawn_count = 0
yawn_open = False
YAWN_OPEN_THRESH = 0.42
YAWN_CLOSE_THRESH = 0.35

# ——— Blink Configs ———
blink_count = 0
blink_frame_counter = 0
BLINK_THRESH = 0.21
CONSEC_FRAMES = 3

# ——— Plot Setup ———
plt.style.use("ggplot")
fig, ax = plt.subplots()
(line_bpm,) = ax.plot([], [], "r-", label="BPM")
bpm_text = ax.text(0.75, 0.9, "", transform=ax.transAxes, fontsize=12)
ax.set_xlim(0, 300)
ax.set_ylim(40, 180)
ax.set_xlabel("Frame")
ax.set_ylabel("BPM")
ax.legend(loc="upper left")
ax.set_title("Real-Time Heart Rate (BPM)")


# ——— Kalman Filter Implementation ———
class KalmanFilter:
    def __init__(
        self, process_variance, measurement_variance, initial_estimate=0.0
    ):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_estimate
        self.error_estimate = 1.0

    def update_batch(self, measurements):
        estimates = []
        for measurement in measurements:
            self.error_estimate += self.process_variance
            kalman_gain = self.error_estimate / (
                self.error_estimate + self.measurement_variance
            )
            self.estimate += kalman_gain * (measurement - self.estimate)
            self.error_estimate *= 1 - kalman_gain
            estimates.append(self.estimate)
        return estimates


# ——— Helpers ———
def get_smooth_outer_contour(landmarks):
    jaw = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17)]
    left_eyebrow = [
        (landmarks.part(i).x, landmarks.part(i).y) for i in range(26, 21, -1)
    ]
    right_eyebrow = [
        (landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 22)
    ]
    points = jaw + left_eyebrow + right_eyebrow[::-1]
    return np.array(points)


def extract_face_roi_mean(frame, landmarks):
    points = get_smooth_outer_contour(landmarks)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], (255,))
    b_channel, g_channel, r_channel = cv2.split(frame)
    b_values = b_channel[mask == 255]
    g_values = g_channel[mask == 255]
    r_values = r_channel[mask == 255]
    b_mean = np.mean(b_values) if len(b_values) > 0 else 0
    g_mean = np.mean(g_values) if len(g_values) > 0 else 0
    r_mean = np.mean(r_values) if len(r_values) > 0 else 0
    return b_mean, g_mean, r_mean, mask


def mouth_aspect_ratio(shape):
    A = np.linalg.norm(
        [
            shape.part(61).x - shape.part(67).x,
            shape.part(61).y - shape.part(67).y,
        ]
    )
    B = np.linalg.norm(
        [
            shape.part(62).x - shape.part(66).x,
            shape.part(62).y - shape.part(66).y,
        ]
    )
    C = np.linalg.norm(
        [
            shape.part(63).x - shape.part(65).x,
            shape.part(63).y - shape.part(65).y,
        ]
    )
    vertical = (A + B + C) / 3.0
    horizontal = np.linalg.norm(
        [
            shape.part(60).x - shape.part(64).x,
            shape.part(60).y - shape.part(64).y,
        ]
    )
    mar = vertical / horizontal
    return mar


def eye_aspect_ratio(shape, eye_indices):
    A = np.linalg.norm(
        [
            shape.part(eye_indices[1]).x - shape.part(eye_indices[5]).x,
            shape.part(eye_indices[1]).y - shape.part(eye_indices[5]).y,
        ]
    )
    B = np.linalg.norm(
        [
            shape.part(eye_indices[2]).x - shape.part(eye_indices[4]).x,
            shape.part(eye_indices[2]).y - shape.part(eye_indices[4]).y,
        ]
    )
    C = np.linalg.norm(
        [
            shape.part(eye_indices[0]).x - shape.part(eye_indices[3]).x,
            shape.part(eye_indices[0]).y - shape.part(eye_indices[3]).y,
        ]
    )
    ear = (A + B) / (2.0 * C)
    return ear


def bandpass(sig, fs=30, low=0.7, high=3.0, order=6):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)


def chrom_recon(r, g, b):
    rn, gn, bn = r / np.mean(r), g / np.mean(g), b / np.mean(b)
    X = 3 * rn - 2 * gn
    Y = 1.5 * rn + gn - 1.5 * bn
    α = np.std(X) / np.std(Y)
    return X - α * Y


def pos_recon(r, g, b):
    rn, gn, bn = r / np.mean(r), g / np.mean(g), b / np.mean(b)
    Xs = 2 * rn - gn - bn
    Ys = gn - bn
    return Xs / np.std(Xs) - Ys / np.std(Ys)


def estimate_fps(timestamps):
    if len(timestamps) < 2:
        return 30
    intervals = np.diff(timestamps)
    return 1 / np.mean(intervals)


def estimate_bpm_fft(times, sig, fs=30, window=12):
    now = time.time()
    idx = [i for i, t in enumerate(times) if now - t <= window]
    if len(idx) < fs * 5:
        return None
    s = np.array([sig[i] for i in idx])
    s = s - np.mean(s)
    s = bandpass(s, fs)
    freqs = np.fft.rfftfreq(len(s), 1 / fs)
    P = np.abs(np.fft.rfft(s))
    mask = (freqs >= 0.7) & (freqs <= 3.0)
    if not mask.any():
        return None
    peak = freqs[mask][np.argmax(P[mask])]
    return 60 * peak


def estimate_bpm_ensemble():
    bpm_list = []
    fs = estimate_fps(chrom_time)
    for sig, ts in ((chrom_signal, chrom_time), (pos_signal, pos_time)):
        bpm = estimate_bpm_fft(ts, sig, fs)
        if bpm:
            bpm_list.append(bpm)
    return int(np.median(bpm_list)) if bpm_list else None


# ——— Animation ———
def update(frame):
    bpm = estimate_bpm_ensemble()
    if bpm:
        raw_bpm_values.append(bpm)
        smoothed = (
            kf.update_batch([bpm])[-1] if len(raw_bpm_values) > 1 else bpm
        )
        smooth_bpm_values.append(smoothed)
        line_bpm.set_data(range(len(smooth_bpm_values)), smooth_bpm_values)
        ax.set_xlim(
            max(0, len(smooth_bpm_values) - 300), len(smooth_bpm_values) + 100
        )
        bpm_text.set_text(f"BPM: {int(smoothed)}")
    return line_bpm, bpm_text


kf = KalmanFilter(
    process_variance=1e-5, measurement_variance=1.0, initial_estimate=60
)
ani = FuncAnimation(fig, update, interval=50, blit=False)
plt.ion()
plt.show()

# ——— Main Loop ———
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if faces:
        face = max(faces, key=lambda r: r.width() * r.height())
        shape = predictor(gray, face)

        b, g, r, face_mask = extract_face_roi_mean(frame, shape)

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
            r_arr, g_arr, b_arr = (
                np.array(r_signal[-30:]),
                np.array(g_signal[-30:]),
                np.array(b_signal[-30:]),
            )
            chrom = chrom_recon(r_arr, g_arr, b_arr)[-1]
            pos = pos_recon(r_arr, g_arr, b_arr)[-1]
            chrom_signal.append(chrom)
            chrom_time.append(time.time())
            pos_signal.append(pos)
            pos_time.append(time.time())

        mar = mouth_aspect_ratio(shape)
        # Detect the beginning of a yawn
        if mar > YAWN_OPEN_THRESH and not yawn_open:
            yawn_open = True
            yawn_count += 1

        # Detect the end of a yawn
        elif mar < YAWN_CLOSE_THRESH and yawn_open:
            yawn_open = False

        left_ear = eye_aspect_ratio(shape, [36, 37, 38, 39, 40, 41])
        right_ear = eye_aspect_ratio(shape, [42, 43, 44, 45, 46, 47])
        ear = (left_ear + right_ear) / 2.0

        # Blink detection
        if ear < BLINK_THRESH:
            blink_frame_counter += 1
        else:
            if blink_frame_counter >= CONSEC_FRAMES:
                blink_count += 1
            blink_frame_counter = 0

        cv2.putText(
            frame,
            f"Yawn Count: {yawn_count}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Blinks: {blink_count}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    cv2.imshow("Face+Ensemble rPPG", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
