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
raw_bpm_values = []  # To hold raw BPM values for smoothing
smooth_bpm_values = []

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
        self.process_variance = process_variance  # Process (model) variance
        self.measurement_variance = measurement_variance  # Measurement variance
        self.estimate = initial_estimate  # Initial estimate
        self.error_estimate = 1.0  # Initial estimate error

    def update(self, measurement):
        # Prediction step
        self.error_estimate += self.process_variance

        # Kalman gain
        kalman_gain = self.error_estimate / (
            self.error_estimate + self.measurement_variance
        )

        # Update estimate with measurement
        self.estimate += kalman_gain * (measurement - self.estimate)

        # Update error estimate
        self.error_estimate *= 1 - kalman_gain

        return self.estimate


# ——— Helpers ———
def get_smooth_outer_contour(landmarks):
    jaw = [
        (landmarks.part(i).x, landmarks.part(i).y) for i in range(17)
    ]  # 0-16
    left_eyebrow = [
        (landmarks.part(i).x, landmarks.part(i).y) for i in range(26, 21, -1)
    ]  # 26 to 22
    right_eyebrow = [
        (landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 22)
    ]  # 17 to 21

    points = jaw + left_eyebrow + right_eyebrow[::-1]
    return np.array(points)


def extract_face_roi_mean(frame, landmarks):
    # Create mask
    points = get_smooth_outer_contour(landmarks)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], (255,))

    # Extract masked pixels for each channel
    b_channel, g_channel, r_channel = cv2.split(frame)
    b_values = b_channel[mask == 255]
    g_values = g_channel[mask == 255]
    r_values = r_channel[mask == 255]

    b_mean = np.mean(b_values) if len(b_values) > 0 else 0
    g_mean = np.mean(g_values) if len(g_values) > 0 else 0
    r_mean = np.mean(r_values) if len(r_values) > 0 else 0

    return b_mean, g_mean, r_mean, mask


def bandpass(sig, fs=30, low=0.7, high=3.0, order=6):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")  # type: ignore
    return filtfilt(b, a, sig)


def chrom_recon(r, g, b):
    # your existing CHROM
    rn, gn, bn = r / np.mean(r), g / np.mean(g), b / np.mean(b)
    X = 3 * rn - 2 * gn
    Y = 1.5 * rn + gn - 1.5 * bn
    α = np.std(X) / np.std(Y)
    return X - α * Y


def pos_recon(r, g, b):
    # de Haan POS
    rn, gn, bn = r / np.mean(r), g / np.mean(g), b / np.mean(b)
    # projection
    Xs = 2 * rn - gn - bn
    Ys = 0 * rn + gn - bn
    std_Xs, std_Ys = np.std(Xs), np.std(Ys)
    return Xs / std_Xs - Ys / std_Ys


def estimate_bpm_fft(times, sig, fs=30, window=12):
    now = time.time()
    idx = [i for i, t in enumerate(times) if now - t <= window]
    if len(idx) < fs * 5:
        return None
    s = np.array([sig[i] for i in idx])
    s = s - np.mean(s)
    s = bandpass(s, fs)
    n = len(s)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    P = np.abs(np.fft.rfft(s))
    mask = (freqs >= 0.7) & (freqs <= 3.0)
    if not mask.any():
        return None
    peak = freqs[mask][np.argmax(P[mask])]
    return 60 * peak


def estimate_bpm_ensemble():
    bpm_list = []
    for sig, ts in (
        (chrom_signal, chrom_time),
        (pos_signal, pos_time),
    ):
        bpm = estimate_bpm_fft(ts, sig)
        if bpm:
            bpm_list.append(bpm)
    return int(np.median(bpm_list)) if bpm_list else None


# ——— Animation ———
def update(frame):
    bpm = estimate_bpm_ensemble()

    if bpm:
        raw_bpm_values.append(bpm)
        smoothed_bpm = kf.update(bpm)  # Apply Kalman filter smoothing
        smooth_bpm_values.append(smoothed_bpm)
        line_bpm.set_data(range(len(smooth_bpm_values)), smooth_bpm_values)

        ax.set_xlim(
            max(0, len(smooth_bpm_values) - 300), len(smooth_bpm_values) + 100
        )
        bpm_text.set_text(f"BPM: {int(smoothed_bpm)}" if smoothed_bpm else "")

    return line_bpm, bpm_text


# ——— Initialize Kalman Filter for BPM smoothing ———
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
        face = faces[0]  # Use first detected face
        shape = predictor(gray, face)

        # Get mean BGR from full face ROI mask
        b, g, r, face_mask = extract_face_roi_mean(frame, shape)

        # show mask overlay (transparent green)
        overlay = frame.copy()
        overlay[face_mask == 255] = (0, 255, 0)
        alpha = 0.1
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        r_signal.append(r)
        g_signal.append(g)
        b_signal.append(b)

        # Draw landmarks
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        if len(r_signal) >= 10:
            arr = np.array(r_signal), np.array(g_signal), np.array(b_signal)
            # compute both signals
            chrom = chrom_recon(*arr)[-1]
            pos = pos_recon(*arr)[-1]
            chrom_signal.append(chrom)
            chrom_time.append(time.time())
            pos_signal.append(pos)
            pos_time.append(time.time())
        else:
            chrom_signal.append(0)
            chrom_time.append(time.time())
            pos_signal.append(0)
            pos_time.append(time.time())

    cv2.imshow("Face+Ensemble rPPG", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
