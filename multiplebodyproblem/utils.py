import cupy as cp

G = 3


def force(
    mass_a: float, mass_b: float, position_a: cp.ndarray, position_b: cp.ndarray
) -> cp.ndarray:
    return (
        G
        * mass_a
        * mass_b
        * (position_b - position_a)
        / cp.linalg.norm(position_b - position_a) ** 3
    )
