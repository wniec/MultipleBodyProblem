from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import cuda, njit
import numpy as np
from kernel_algorithms import nbody_kernel
from generator import generate_stars
import threading
import animator


def physics_step(d_pos, d_vel, d_mass, dt, n_bodies, G, SOFTENING, blocks, tpb):
    nbody_kernel[blocks, tpb](d_pos, d_vel, d_mass, dt, n_bodies, G, SOFTENING)


def run_simulation(n_bodies, threads_per_block, dt, G, SOFTENING, interval, seed=None):
    blocks = (n_bodies + (threads_per_block - 1)) // threads_per_block

    my_lock = threading.Lock()

    mass, pos, vel = generate_stars(n_stars=n_bodies, seed=seed)
    d_pos = cuda.to_device(pos)
    d_vel = cuda.to_device(vel)
    d_mass = cuda.to_device(mass)

    fig, ax, scatter = animator.setup_3d_stage(mass, pos)

    physics_args = (d_pos, d_vel, d_mass, dt, n_bodies, G, SOFTENING, blocks, threads_per_block)

    animator.start_engine(fig, scatter, d_pos, physics_step, physics_args, my_lock, interval=interval)

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection="3d")
    #
    # pos_cpu = d_pos.copy_to_host()
    #
    # scatter = ax.scatter(
    #     pos_cpu[:, 0],
    #     pos_cpu[:, 1],
    #     pos_cpu[:, 2],
    #     c=mass,
    #     s=np.log(mass) * 20,
    #     cmap="hsv",
    #     alpha=0.7,
    #     edgecolors="none"
    # )
    #
    # ax.set_xlim(-3, 3)
    # ax.set_ylim(-3, 3)
    # ax.set_zlim(-3, 3)
    #
    # ax.set_facecolor("black")
    # fig.patch.set_facecolor("black")
    # ax.grid(False)
    #
    # ax.set_title("3D N-Body Simulation (CUDA)")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    #
    # # ------------------
    # # Animation function
    # # ------------------
    # def update(frame):
    #     nbody_kernel[blocks, threads_per_block](
    #         d_pos, d_vel, d_mass, dt, n_bodies, G, SOFTENING
    #     )
    #
    #     pos_cpu = d_pos.copy_to_host()
    #
    #     scatter._offsets3d = (
    #         pos_cpu[:, 0],
    #         pos_cpu[:, 1],
    #         pos_cpu[:, 2]
    #     )
    #
    #     return scatter,
    #
    # anim = FuncAnimation(
    #     fig,
    #     update,
    #     frames=10,
    #     interval=1,
    #     blit=False,
    #     repeat=True
    # )
    #
    # plt.show()
