"""
Lab 5: Particle Swarm Optimization (PSO) for Global Minima
Concepts: velocity update, position update, pBest and gBest
"""

import numpy as np
import matplotlib.pyplot as plt


class PSO:
    def __init__(self, func, dim, bounds, num_particles=30, max_iter=100,
                 w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, vmax=None, seed=None):
        self.func = func
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w_max = w_max          # inertia at the start
        self.w_min = w_min          # inertia at the end (linear decrease)
        self.c1 = c1                # cognitive coefficient (pull to pBest)
        self.c2 = c2                # social coefficient (pull to gBest)
        self.rng = np.random.default_rng(seed)

        # Normalize bounds -> shape (dim, 2)
        b = np.array(bounds, dtype=float)
        if b.ndim == 1:                           # e.g. (-5.12, 5.12)
            b = np.tile(b, (dim, 1))
        self.bounds = b
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]

        # Max velocity (default = full span of the search space)
        span = self.high - self.low
        self.vmax = span if vmax is None else np.full(dim, vmax, dtype=float)

        # Initialize positions and velocities
        self.pos = self._init_positions()
        self.vel = self._init_velocities()

        # Evaluate initial fitness -> pBest and gBest
        self.fit = np.array([self.func(p) for p in self.pos])
        self.pbest_pos = self.pos.copy()
        self.pbest_val = self.fit.copy()
        g = np.argmin(self.pbest_val)
        self.gbest_pos = self.pbest_pos[g].copy()
        self.gbest_val = self.pbest_val[g]

    def _init_positions(self):
        """Random positions uniformly inside the bounds."""
        return self.rng.uniform(self.low, self.high,
                                size=(self.num_particles, self.dim))

    def _init_velocities(self):
        """Random velocities in [-vmax, +vmax]."""
        return self.rng.uniform(-self.vmax, self.vmax,
                                size=(self.num_particles, self.dim))

    def optimize(self):
        history = [self.gbest_val]

        for t in range(self.max_iter):
            # Inertia weight decreases linearly from w_max to w_min
            w = self.w_max - (self.w_max - self.w_min) * t / self.max_iter

            # Random matrices
            r1 = self.rng.random((self.num_particles, self.dim))
            r2 = self.rng.random((self.num_particles, self.dim))

            # Velocity update: inertia + cognitive + social
            self.vel = (w * self.vel
                        + self.c1 * r1 * (self.pbest_pos - self.pos)
                        + self.c2 * r2 * (self.gbest_pos - self.pos))
            self.vel = np.clip(self.vel, -self.vmax, self.vmax)

            # Position update
            self.pos = self.pos + self.vel
            self.pos = np.clip(self.pos, self.low, self.high)

            # Evaluate fitness of all particles
            self.fit = np.array([self.func(p) for p in self.pos])

            # Update pBest
            improved = self.fit < self.pbest_val
            self.pbest_pos[improved] = self.pos[improved]
            self.pbest_val[improved] = self.fit[improved]

            # Update gBest
            g = np.argmin(self.pbest_val)
            if self.pbest_val[g] < self.gbest_val:
                self.gbest_val = self.pbest_val[g]
                self.gbest_pos = self.pbest_pos[g].copy()

            history.append(self.gbest_val)

        return self.gbest_pos, self.gbest_val, history


# -----------------------------
# Benchmark function
# -----------------------------
def rastrigin(x):
    x = np.asarray(x)
    A = 10.0
    return A * x.size + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    dim = 2
    bounds = (-5.12, 5.12)

    pso = PSO(func=rastrigin, dim=dim, bounds=bounds,
              num_particles=30, max_iter=100,
              w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, seed=42)

    best_pos, best_val, history = pso.optimize()

    print("Best position:", best_pos)
    print("Best value   :", best_val)

    plt.figure(figsize=(8, 5))
    plt.plot(history, linewidth=2)
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("gBest value (log scale)")
    plt.title("PSO Convergence on Rastrigin (2D)")
    plt.grid(True)
    plt.show()
