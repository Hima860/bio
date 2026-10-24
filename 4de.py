"""
Lab 6: Differential Evolution (DE) for Function Optimization
Concepts: mutation, crossover, selection strategies
"""

import numpy as np
import matplotlib.pyplot as plt


def differential_evolution(func, bounds, pop_size=30, F=0.8, CR=0.9,
                           generations=200, strategy="rand/1", seed=None):
    rng = np.random.default_rng(seed)

    bounds = np.array(bounds, dtype=float)       # list of (low, high)
    dim = len(bounds)
    low, high = bounds[:, 0], bounds[:, 1]

    # a. Initialization: uniform random population inside bounds
    pop = rng.uniform(low, high, size=(pop_size, dim))

    # b. Evaluate initial population
    fitness = np.array([func(x) for x in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx].copy()
    best_val = fitness[best_idx]

    # c. Track progress
    history = [best_val]
    diversity = [np.mean(np.std(pop, axis=0))]   # avg std-dev across dims

    for g in range(generations):
        for i in range(pop_size):
            # ---- Mutation: pick 3 distinct individuals (all != i)
            choices = [k for k in range(pop_size) if k != i]
            r1, r2, r3 = rng.choice(choices, size=3, replace=False)
            a, b, c = pop[r1], pop[r2], pop[r3]

            if strategy == "rand/1":
                mutant = a + F * (b - c)
            elif strategy == "best/1":
                mutant = best + F * (b - c)
            elif strategy == "current-to-best/1":
                mutant = pop[i] + F * (best - pop[i]) + F * (b - c)
            else:
                raise ValueError("Unknown strategy: " + strategy)

            mutant = np.clip(mutant, low, high)

            # ---- Binomial crossover (jrand guarantees >= 1 gene from mutant)
            cross = rng.random(dim) < CR
            jrand = rng.integers(dim)
            cross[jrand] = True
            trial = np.where(cross, mutant, pop[i])

            # ---- Selection: greedy one-to-one replacement
            trial_val = func(trial)
            if trial_val < fitness[i]:
                pop[i] = trial
                fitness[i] = trial_val
                if trial_val < best_val:
                    best_val = trial_val
                    best = trial.copy()

        history.append(best_val)
        diversity.append(np.mean(np.std(pop, axis=0)))

    return best, best_val, history, diversity


# -----------------------------
# Benchmark functions
# -----------------------------
def sphere(x):
    """Smooth, convex. Global min 0 at x = 0."""
    x = np.asarray(x)
    return np.sum(x ** 2)


def rastrigin(x):
    """Rugged, multi-modal. Global min 0 at x = 0."""
    x = np.asarray(x)
    return 10.0 * x.size + np.sum(x ** 2 - 10.0 * np.cos(2 * np.pi * x))


# -----------------------------
# Main: compare strategies
# -----------------------------
if __name__ == "__main__":
    dim = 10
    bounds = [(-5.12, 5.12)] * dim
    strategies = ["rand/1", "best/1", "current-to-best/1"]

    for name, func in [("Sphere", sphere), ("Rastrigin", rastrigin)]:
        plt.figure(figsize=(8, 5))
        print(f"\n=== {name} ({dim}-D) ===")
        for s in strategies:
            best, best_val, hist, div = differential_evolution(
                func, bounds, pop_size=40, F=0.8, CR=0.9,
                generations=200, strategy=s, seed=42)
            print(f"{s:20s} best value = {best_val:.6f}")
            plt.plot(hist, linewidth=2, label=s)

        plt.yscale('log')
        plt.xlabel("Generation")
        plt.ylabel("Best fitness (log scale)")
        plt.title(f"DE strategies on {name}")
        plt.legend()
        plt.grid(True)
        plt.show()
