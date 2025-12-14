from matplotlib import pyplot as plt

from generator import generate_stars

if __name__ == "__main__":
    masses, positions, velocities = generate_stars(500)
    plt.hist(masses.get())
    plt.savefig("masses.png")
    plt.close()
    plt.hist(positions[:, 0].get())
    plt.savefig("positions.png")
    plt.close()
    plt.hist(velocities[:, 0].get())
    plt.savefig("velocities.png")
    plt.close()