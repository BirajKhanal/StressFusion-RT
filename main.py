import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from face_detect import FaceDetectorYunet

# Initialize the detector
detector = FaceDetectorYunet()

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Set a desired resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Store time series
r_signal = []
g_signal = []
b_signal = []

# --- Matplotlib Setup ---
plt.style.use("ggplot")  # Or choose a style you like

fig, ax = plt.subplots()
(line_r,) = ax.plot([], [], color="red", label="Red Channel")
(line_g,) = ax.plot([], [], color="green", label="Green Channel")
(line_b,) = ax.plot([], [], color="blue", label="Blue Channel")

ax.set_xlim(0, 300)  # Show last 300 frames
ax.set_ylim(0, 255)  # Intensity range from 0 to 255
ax.set_title("Live RGB Channel Signals")
ax.set_xlabel("Frame")
ax.set_ylabel("Mean Intensity")
ax.legend()


# Update function for animation
def update_plot(frame):
    if len(r_signal) == 0:
        return line_r, line_g, line_b
    line_r.set_data(range(len(r_signal)), r_signal)
    line_g.set_data(range(len(g_signal)), g_signal)
    line_b.set_data(range(len(b_signal)), b_signal)

    ax.set_xlim(max(0, len(r_signal) - 300), len(r_signal))  # Scroll window
    return line_r, line_g, line_b


ani = FuncAnimation(fig, update_plot, interval=50, blit=True)

# Show the plot window in non-blocking mode
plt.ion()
plt.show()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect faces in the current frame
    predictions = detector.detect(frame)

    # Draw results if faces are found
    if predictions is not None:
        face = predictions[0]  # Use first detected face
        x1, y1, x2, y2 = face["x1"], face["y1"], face["x2"], face["y2"]
        roi = frame[y1:y2, x1:x2]

        # Compute mean R, G, B
        mean_rgb = cv2.mean(roi)[:3]  # (B, G, R) order
        b, g, r = mean_rgb

        # Append to time series
        r_signal.append(r)
        g_signal.append(g)
        b_signal.append(b)

        frame = detector.draw_faces(
            frame, predictions, draw_landmarks=True, show_confidence=True
        )

    # Show the frame
    cv2.imshow("Face Detection", frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
