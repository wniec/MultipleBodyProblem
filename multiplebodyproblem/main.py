import argparse
from utils import run_simulation

if __name__ == "__main__":
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="GPU N-Body Gravity Simulation")

    parser.add_argument(
        "--bodies", "-n",
        type=int,
        default=64,
        help="Number of stars/bodies to simulate (default: 64)"
    )

    parser.add_argument(
        "--tpb", "-t",
        type=int,
        default=128,
        help="Threads per block for CUDA kernel (default: 128)"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no animation) for benchmarking"
    )

    args = parser.parse_args()

    N_BODIES = args.bodies
    THREADS_PER_BLOCK = args.tpb
    HEADLESS = args.headless

    DT = 1e-8
    G = 1e5
    SOFTENING = 1e-1
    SEED = 213769421

    # for animation (1000/interval = FPS)
    INTERVAL = 16

    print(f"Running Simulation with {N_BODIES} bodies...")
    print(f"Threads Per Block: {THREADS_PER_BLOCK}")
    print(f"Mode: {'HEADLESS (Benchmarking)' if HEADLESS else 'GUI (Visual)'}")

    run_simulation(
        N_BODIES,
        threads_per_block=THREADS_PER_BLOCK,
        dt=DT,
        G=G,
        SOFTENING=SOFTENING,
        seed=SEED,
        interval=INTERVAL,
        headless=HEADLESS,
    )