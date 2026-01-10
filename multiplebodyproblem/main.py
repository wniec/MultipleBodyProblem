from utils import run_simulation

if __name__ == "__main__":
    N_BODIES = 10
    THREADS_PER_BLOCK = 2
    DT = 0.00000001
    G = 1e5
    SOFTENING = 1e-1
    SEED = 213769421

    # for animation (1000/interval = FPS)
    INTERVAL = 16

    # balancing N_BODIES and G is tricky. I noticed that values 100 and 1e4 works well (1000, 1e3), (10, 1e5). I assume
    # we need "constant total value of gravity (XD) balance". Maybe it will be better to set some formula for it?

    # BE AWARE of DT. For faster gpu even SMALLER DT is needed

    # We might not need locking. Require further testing on animation quality and performance impact

    # TODO
    # Benchmarks for different N_BODIES and THREADS_PER_BLOCK
    # Easy way of disabling animation to test max performance
    # Is my project structure correct? Maybe there is nicer way for this project. I did my best
    # Center camera on avg mass
    # Separate physics_counter from gpu thread

    run_simulation(N_BODIES, threads_per_block=THREADS_PER_BLOCK, dt=DT, G=G, SOFTENING=SOFTENING, seed=SEED, interval=INTERVAL)


