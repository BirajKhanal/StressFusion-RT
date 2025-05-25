import cv2
import dlib
import numpy as np

from utils.landmarks import get_smooth_outer_contour


class FaceDetector:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect(self, gray_frame):
        return self.detector(gray_frame)

    def get_landmarks(self, gray_frame, face_rect):
        return self.predictor(gray_frame, face_rect)

    def extract_face_roi_mean(self, frame, landmarks):
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
