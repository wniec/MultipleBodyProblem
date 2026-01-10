import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import cuda

G = 1e1
SOFTENING = 1e-1
N_BODIES = 10

SEED = 8


def generate_stars(n_stars: int, n_dimensions: int = 3):
    np.random.seed(SEED)
    # masses = np.exp(np.random.normal(size=n_stars)) * 10
    masses = np.random.uniform(low=1, high=5, size=n_stars) * 1000
    positions = np.random.uniform(low=-1, high=1, size=(n_stars, n_dimensions)) * 2
    velocities = np.random.normal(size=(n_stars, n_dimensions)) * 10000
    return masses, positions, velocities


def static_visualization_3d(masses, positions):
    pos_cpu = positions.get()
    mass_cpu = masses.get()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        pos_cpu[:, 0],
        pos_cpu[:, 1],
        pos_cpu[:, 2],
        c=mass_cpu,
        s=mass_cpu, # Scaling size for visibility
        cmap='hsv',
        alpha=0.6,
        edgecolors='none'
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Mass Intensity')

    ax.set_title(f"3D Distribution of {len(mass_cpu)} Stars")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set a dark background for a "space" feel
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.show()

def save_hists(masses, positions, velocities):
    plt.hist(masses.get())
    plt.savefig("masses.png")
    plt.close()
    plt.hist(positions[:, 0].get())
    plt.savefig("positions.png")
    plt.close()
    plt.hist(velocities[:, 0].get())
    plt.savefig("velocities.png")
    plt.close()


@cuda.jit
def nbody_kernel(pos, vel, mass, dt, n_bodies):
    i = cuda.grid(1)
    if i >= n_bodies:
        return

    ax, ay, az = 0.0, 0.0, 0.0

    xi = pos[i, 0]
    yi = pos[i, 1]
    zi = pos[i, 2]

    for j in range(n_bodies):
        xj = pos[j, 0]
        yj = pos[j, 1]
        zj = pos[j, 2]

        dx = xj - xi
        dy = yj - yi
        dz = zj - zi

        dist_sqr = dx**2 + dy**2 + dz**2 + SOFTENING

        inv_dist = 1.0 / math.sqrt(dist_sqr)

        inv_dist_cube = inv_dist * inv_dist * inv_dist

        s = G * mass[j] * inv_dist_cube

        ax += s * dx
        ay += s * dy
        az += s * dz

    vel[i, 0] += ax * dt
    vel[i, 1] += ay * dt
    vel[i, 2] += az * dt

    pos[i, 0] += vel[i, 0] * dt
    pos[i, 1] += vel[i, 1] * dt
    pos[i, 2] += vel[i, 2] * dt


def run_simulation():
    n_bodies = N_BODIES
    threads_per_block = 32
    blocks = (n_bodies + (threads_per_block - 1)) // threads_per_block
    dt = 0.00001

    mass, pos, vel = generate_stars(n_bodies)

    d_pos = cuda.to_device(pos)
    d_vel = cuda.to_device(vel)
    d_mass = cuda.to_device(mass)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    pos_cpu = d_pos.copy_to_host()

    scatter = ax.scatter(
        pos_cpu[:, 0],
        pos_cpu[:, 1],
        pos_cpu[:, 2],
        c=mass,
        s=np.log(mass) * 20,
        cmap="hsv",
        alpha=0.7,
        edgecolors="none"
    )

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)

    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.grid(False)

    ax.set_title("3D N-Body Simulation (CUDA)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # ------------------
    # Animation function
    # ------------------
    def update(frame):
        nbody_kernel[blocks, threads_per_block](
            d_pos, d_vel, d_mass, dt, n_bodies
        )

        pos_cpu = d_pos.copy_to_host()

        scatter._offsets3d = (
            pos_cpu[:, 0],
            pos_cpu[:, 1],
            pos_cpu[:, 2]
        )

        return scatter,

    anim = FuncAnimation(
        fig,
        update,
        frames=10,
        interval=1,
        blit=False,
        repeat=True
    )

    plt.show()

    # for step in range(100):
    #     # print(d_pos.copy_to_host())
    #     nbody_kernel[blocks, threads_per_block](d_pos, d_vel, d_mass, dt, n_bodies)
    #
    #     pos_cpu = d_pos.copy_to_host()
    #     mass_cpu = d_mass.copy_to_host()


if __name__ == "__main__":
    run_simulation()
