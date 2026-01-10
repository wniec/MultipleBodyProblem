import threading
import time

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

running = True


def setup_3d_stage(mass, pos):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.grid(False)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)

    scatter = ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        c=mass, s=np.log(mass + 1) * 20, cmap="hsv", alpha=0.7
    )
    return fig, ax, scatter


def start_engine(fig, scatter, d_pos, physics_step_func, physics_args, data_lock, interval):
    global running

    def background_wrapper():
        global running
        while running:
            physics_step_func(*physics_args)
            time.sleep(0.0001)  # never delete this shit!

    t = threading.Thread(target=background_wrapper, daemon=True)
    t.start()

    def update_plot(frame):
        with data_lock:
            pos_cpu = d_pos.copy_to_host()

        scatter._offsets3d = (pos_cpu[:, 0], pos_cpu[:, 1], pos_cpu[:, 2])
        return scatter,

    anim = FuncAnimation(fig, update_plot, interval=interval, blit=False, cache_frame_data=False)
    plt.show()

    running = False