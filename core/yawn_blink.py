def detect_yawn(
    mar, yawn_open, yawn_count, open_thresh=0.42, close_thresh=0.35
):
    if mar > open_thresh and not yawn_open:
        yawn_open = True
        yawn_count += 1
    elif mar < close_thresh and yawn_open:
        yawn_open = False
    return yawn_open, yawn_count


def detect_blink(
    ear, blink_frame_counter, blink_count, blink_thresh=0.21, consec_frames=3
):
    if ear < blink_thresh:
        blink_frame_counter += 1
    else:
        if blink_frame_counter >= consec_frames:
            blink_count += 1
        blink_frame_counter = 0
    return blink_frame_counter, blink_count
