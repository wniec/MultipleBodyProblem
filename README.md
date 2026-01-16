# GPU-Accelerated N-Body Gravity Simulation

## Overview
This project implements a high-performance **N-Body Gravity Simulation** using Python and CUDA (via Numba). It simulates the gravitational interaction between multiple celestial bodies in 3D space. The simulation includes a real-time, threaded 3D visualization using Matplotlib and a headless mode for performance benchmarking.

## Mathematical Model
The simulation relies on Newton's Law of Universal Gravitation. For every pair of bodies $(i, j)$, the force exerted on body $i$ by body $j$ is calculated. To avoid numerical instability when bodies get too close (singularities), a **softening parameter** ($\epsilon$) is added to the distance calculation.

The acceleration vector $\mathbf{a}_i$ for body $i$ is calculated as:

$$
\mathbf{a}_i = G \sum_{j \neq i} \frac{m_j (\mathbf{r}_j - \mathbf{r}_i)}{(|\mathbf{r}_j - \mathbf{r}_i|^2 + \epsilon^2)^{3/2}}
$$

Where:
* $G$ is the gravitational constant.
* $m_j$ is the mass of body $j$.
* $\mathbf{r}$ represents the position vector.
* $\epsilon$ is the softening parameter (set to `1e-1`).

The system uses **Euler integration** to update velocities and positions based on calculated acceleration over a time step $dt$ ($10^{-8}$).

## Implementation Details
* **Language:** Python 3.x
* **Acceleration:** `numba.cuda` is used to compile the physics kernel directly to the GPU.
* **Algorithm:** The kernel implements a direct $O(N^2)$ summation, where every body interacts with every other body.
* **Visualization:**
    * Uses `matplotlib` with a Qt5 backend for 3D rendering.
    * The physics engine runs on a separate daemon thread to ensure the GUI remains responsive while the GPU calculates updates.
    * Visuals include star size scaling based on mass and plasma coloring.

## Installation
This project uses `uv` for dependency management.

```bash
uv sync
```

## Results:
each cell contains mean frames per second while simulating Multiple Body Problem.

| n (Bodies) \ TPB (Threads per block) | 64      | 128       | 256      |
|--------------------------------------|---------|-----------|----------| 
| 16                                   | 9326.92 | 10276.06  | 9267.25  |
| 32                                   | 9457.30 | 10209.93  | 9803.85  |
| 64                                   | 9525.59 | 10140.10  | 10338.70 |
| 128                                  | 7898.57 | 6070.70   | 6096.60  |
| 256                                  | 4755.47 | 3404.22   | 1851.61  |
| 512                                  | 2488.25 | 1807.65   | 959.67   |
| 1024                                 | 1371.32 | 930.85    | 488.60   |
