from numba import cuda
import math


@cuda.jit
def nbody_kernel(pos, vel, mass, dt, n_bodies, G, SOFTENING):
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
