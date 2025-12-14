import cupy as cp


def generate_stars(n_stars: int, n_dimensions: int = 3):
    masses = cp.exp(cp.random.normal(size=n_stars))
    positions = cp.random.uniform(low=-1, high=1, size=(n_stars, n_dimensions))
    velocities = cp.random.normal(size=(n_stars, n_dimensions))
    return masses, positions, velocities
