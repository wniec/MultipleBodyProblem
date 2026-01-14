from numba import cuda
from kernel_algorithms import nbody_kernel
from generator import generate_stars
import threading
import time
import animator


def physics_step(d_pos, d_vel, d_mass, dt, n_bodies, G, SOFTENING, blocks, tpb):
    nbody_kernel[blocks, tpb](d_pos, d_vel, d_mass, dt, n_bodies, G, SOFTENING)


def run_simulation(n_bodies, threads_per_block, dt, G, SOFTENING, interval, seed=None, headless=False):
    blocks = (n_bodies + (threads_per_block - 1)) // threads_per_block

    mass, pos, vel = generate_stars(n_stars=n_bodies, seed=seed)
    d_pos = cuda.to_device(pos)
    d_vel = cuda.to_device(vel)
    d_mass = cuda.to_device(mass)

    physics_args = (d_pos, d_vel, d_mass, dt, n_bodies, G, SOFTENING, blocks, threads_per_block)
    if headless:
        TARGET_STEPS = 10000

        print(f"Running headless benchmark ({TARGET_STEPS} steps)...")

        start_time = time.time()

        for _ in range(TARGET_STEPS):
            physics_step(*physics_args)


        end_time = time.time()
        elapsed = end_time - start_time

        mean_fps = TARGET_STEPS / elapsed
        print(f"Mean FPS: {mean_fps:.2f}")

    else:
        my_lock = threading.Lock()
        fig, ax, scatter, stats_text = animator.setup_3d_stage(mass, pos)

        animator.start_engine(fig, scatter, stats_text, d_pos, physics_step, physics_args, my_lock, interval=interval)