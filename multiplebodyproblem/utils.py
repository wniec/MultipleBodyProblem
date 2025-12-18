import numpy as np
import math
from numba import cuda

G = 6.67430e-11
SOFTENING = 1e-9


def generate_stars(n_stars: int, n_dimensions: int = 3):
    masses = np.exp(np.random.normal(size=n_stars))
    positions = np.random.uniform(low=-1, high=1, size=(n_stars, n_dimensions))
    velocities = np.random.normal(size=(n_stars, n_dimensions))
    return masses, positions, velocities


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
    n_bodies = 500
    threads_per_block = 128
    blocks = (n_bodies + (threads_per_block - 1)) // threads_per_block
    dt = 0.01

    mass, vel, pos = generate_stars(n_bodies)

    d_pos = cuda.to_device(pos)
    d_vel = cuda.to_device(vel)
    d_mass = cuda.to_device(mass)

    for step in range(100):
        print(d_pos.copy_to_host())
        nbody_kernel[blocks, threads_per_block](d_pos, d_vel, d_mass, dt, n_bodies)


if __name__ == "__main__":
    run_simulation()
