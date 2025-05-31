import time

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass(sig, fs=30, low=0.7, high=3.0, order=6):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)


def chrom_recon(r, g, b):
    rn, gn, bn = r / np.mean(r), g / np.mean(g), b / np.mean(b)
    X = 3 * rn - 2 * gn
    Y = 1.5 * rn + gn - 1.5 * bn
    alpha = np.std(X) / np.std(Y)
    return X - alpha * Y


def pos_recon(r, g, b):
    rn, gn, bn = r / np.mean(r), g / np.mean(g), b / np.mean(b)
    Xs = gn + bn - 2 * rn
    Ys = gn - bn
    return Xs / np.std(Xs) + Ys / np.std(Ys)


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


def estimate_bpm_ensemble(chrom_signal, chrom_time, pos_signal, pos_time):
    bpm_list = []
    fs = estimate_fps(chrom_time)
    for sig, ts in ((chrom_signal, chrom_time), (pos_signal, pos_time)):
        bpm = estimate_bpm_fft(ts, sig, fs)
        if bpm:
            bpm_list.append(bpm)
    return int(np.median(bpm_list)) if bpm_list else None
