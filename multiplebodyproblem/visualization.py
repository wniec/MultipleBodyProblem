from matplotlib import pyplot as plt


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


def static_visualization_3d(masses, positions):
    pos_cpu = positions.get()
    mass_cpu = masses.get()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        pos_cpu[:, 0],
        pos_cpu[:, 1],
        pos_cpu[:, 2],
        c=mass_cpu,
        s=mass_cpu,  # Scaling size for visibility
        cmap="hsv",
        alpha=0.6,
        edgecolors="none",
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label("Mass Intensity")

    ax.set_title(f"3D Distribution of {len(mass_cpu)} Stars")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set a dark background for a "space" feel
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.show()
