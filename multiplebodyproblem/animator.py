import threading
import time

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

running = True

physics_counter = 0
last_time = time.time()
last_count = 0

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

    stats_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, color="#00FF00", family='monospace')

    return fig, ax, scatter, stats_text


def start_engine(fig, scatter, stats_text, d_pos, physics_step_func, physics_args, data_lock, interval):
    global running

    def background_wrapper():
        global running, physics_counter
        while running:
            physics_step_func(*physics_args)
            physics_counter += 1
            time.sleep(0)  # never delete this shit!

        # this might be faster / better. Testing required
        # while running:
        #     for _ in range(5):
        #         physics_step_func(*physics_args)
        #     physics_counter += 5
        #     time.sleep(0.0001)

    t = threading.Thread(target=background_wrapper, daemon=True)
    t.start()


    def update_plot(frame):
        global last_time, last_count, physics_counter

        with data_lock:
            pos_cpu = d_pos.copy_to_host()

        scatter._offsets3d = (pos_cpu[:, 0], pos_cpu[:, 1], pos_cpu[:, 2])

        current_time = time.time()
        elapsed = current_time - last_time

        if elapsed >= 0.5:  # Update text every half second to avoid flickering
            current_count = physics_counter
            steps_done = current_count - last_count

            gpu_fps = int(steps_done / elapsed)
            stats_text.set_text(f"GPU PHYSICS: {gpu_fps:,} steps/sec")

            last_time = current_time
            last_count = current_count

        return scatter, stats_text


    anim = FuncAnimation(fig, update_plot, interval=interval, blit=False, cache_frame_data=False)
    plt.show()

    running = False