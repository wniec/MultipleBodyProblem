from generator import generate_stars
from utils import save_hists, static_visualization_3d

if __name__ == "__main__":
    n_stars = 100
    masses, positions, velocities = generate_stars(n_stars)

    save_hists(masses, positions, velocities)
    static_visualization_3d(masses, positions)
