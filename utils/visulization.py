import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def init_bpm_plot():
    """Initialize the BPM plot and return figure, axis, line, and bpm_text."""
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
    return fig, ax, line_bpm, bpm_text


def update_bpm_plot(frame_num, bpm_values, line_bpm, bpm_text, ax, max_len=300):
    """
    Update the BPM plot animation.
    :param frame_num: Frame number from FuncAnimation
    :param bpm_values: List of BPM values (smoothed)
    :param line_bpm: The matplotlib line object to update
    :param bpm_text: The matplotlib text object to update BPM number
    :param ax: matplotlib axis object
    :param max_len: max x-axis length to show
    """
    if not bpm_values:
        return line_bpm, bpm_text

    xdata = range(len(bpm_values))
    ydata = bpm_values

    line_bpm.set_data(xdata, ydata)
    start_x = max(0, len(bpm_values) - max_len)
    ax.set_xlim(start_x, start_x + max_len)
    bpm_text.set_text(f"BPM: {int(ydata[-1])}")

    return line_bpm, bpm_text


def create_bpm_animation(fig, ax, line_bpm, bpm_text, bpm_values, interval=50):
    """
    Create and return a matplotlib FuncAnimation object for BPM updates.
    :param fig: Matplotlib figure
    :param ax: Matplotlib axis
    :param line_bpm: Line object to update
    :param bpm_text: Text object to update
    :param bpm_values: List of BPM values (smoothed)
    :param interval: update interval in ms
    """

    def anim_func(frame_num):
        return update_bpm_plot(frame_num, bpm_values, line_bpm, bpm_text, ax)

    ani = FuncAnimation(fig, anim_func, interval=interval, blit=False)
    return ani
