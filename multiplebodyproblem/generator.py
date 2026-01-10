import cupy as cp
import numpy as np


# def generate_stars(n_stars: int, n_dimensions: int = 3):
#     masses = cp.exp(cp.random.normal(size=n_stars))
#     positions = cp.random.uniform(low=-1, high=1, size=(n_stars, n_dimensions))
#     velocities = cp.random.normal(size=(n_stars, n_dimensions))
#     return masses, positions, velocities


def generate_stars(n_stars: int, n_dimensions: int = 3, seed=None):
    if seed is not None:
        np.random.seed(seed)

    masses = np.random.uniform(low=1, high=5, size=n_stars) * 1000
    positions = np.random.uniform(low=-1, high=1, size=(n_stars, n_dimensions)) * 2
    velocities = np.random.normal(size=(n_stars, n_dimensions)) * 10000
    return masses, positions, velocities