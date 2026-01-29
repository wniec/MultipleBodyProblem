import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

INPUT_FILE = "benchmarks.csv"
OUTPUT_IMAGE = "fps_benchmark_plot.png"


def plot_benchmark():
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find '{INPUT_FILE}'. Make sure to run the parsing script first.")
        return

    plt.figure(figsize=(10, 6))

    tpb_values = sorted(df['TPB'].unique())

    for tpb in tpb_values:
        subset = df[df['TPB'] == tpb]

        subset = subset.sort_values(by='N')

        plt.plot(subset['N'], subset['FPS'], marker='o', label=f'TPB: {tpb}')

    plt.yscale('function', functions=(np.sqrt, np.square))
    plt.xscale("log")

    unique_n = sorted(df['N'].unique())
    plt.xticks(unique_n, unique_n)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())

    plt.xlabel('Number of Bodies (N) - Log Scale')
    plt.ylabel('FPS')
    plt.title('Simulation Performance: N vs FPS by Threads Per Block')
    plt.legend(title="Threads Per Block")
    plt.grid(True, ls="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"Plot saved to {OUTPUT_IMAGE}")
    plt.show()


if __name__ == "__main__":
    plot_benchmark()