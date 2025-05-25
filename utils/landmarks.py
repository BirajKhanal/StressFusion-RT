import numpy as np


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


def mouth_aspect_ratio(shape):
    import numpy.linalg as LA

    A = LA.norm(
        [
            shape.part(61).x - shape.part(67).x,
            shape.part(61).y - shape.part(67).y,
        ]
    )
    B = LA.norm(
        [
            shape.part(62).x - shape.part(66).x,
            shape.part(62).y - shape.part(66).y,
        ]
    )
    C = LA.norm(
        [
            shape.part(63).x - shape.part(65).x,
            shape.part(63).y - shape.part(65).y,
        ]
    )
    vertical = (A + B + C) / 3.0
    horizontal = LA.norm(
        [
            shape.part(60).x - shape.part(64).x,
            shape.part(60).y - shape.part(64).y,
        ]
    )
    mar = vertical / horizontal
    return mar


def eye_aspect_ratio(shape, eye_indices):
    import numpy.linalg as LA

    A = LA.norm(
        [
            shape.part(eye_indices[1]).x - shape.part(eye_indices[5]).x,
            shape.part(eye_indices[1]).y - shape.part(eye_indices[5]).y,
        ]
    )
    B = LA.norm(
        [
            shape.part(eye_indices[2]).x - shape.part(eye_indices[4]).x,
            shape.part(eye_indices[2]).y - shape.part(eye_indices[4]).y,
        ]
    )
    C = LA.norm(
        [
            shape.part(eye_indices[0]).x - shape.part(eye_indices[3]).x,
            shape.part(eye_indices[0]).y - shape.part(eye_indices[3]).y,
        ]
    )
    ear = (A + B) / (2.0 * C)
    return ear
