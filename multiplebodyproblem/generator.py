import numpy as np


def generate_stars(n_stars: int, n_dimensions: int = 3, seed=None):
    if seed is not None:
        np.random.seed(seed)

    masses = np.exp(np.random.normal(loc=0, scale=1, size=n_stars)) * 1000
    positions = np.random.uniform(low=-1, high=1, size=(n_stars, n_dimensions)) * 2
    velocities = np.random.normal(size=(n_stars, n_dimensions)) * 10000
    return masses, positions, velocities
