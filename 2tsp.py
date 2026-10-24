"""
Lab 3: Simulated Annealing for the Travelling Salesman Problem (TSP)
Concepts: temperature schedule, acceptance probability, path cost evaluation
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt


# -----------------------------
# Utility functions
# -----------------------------
def euclidean_distance_matrix(coords):
    """Compute pairwise Euclidean distances for a set of coordinates."""
    diff = coords[:, None, :] - coords[None, :, :]      # shape (n, n, 2)
    return np.sqrt(np.sum(diff ** 2, axis=-1))           # shape (n, n)


def tour_length(route, dist_matrix):
    """Total length of a closed tour (wraps back to the start city)."""
    n = len(route)
    total = 0.0
    for i in range(n):
        total += dist_matrix[route[i], route[(i + 1) % n]]
    return total


def two_opt_swap(route):
    """2-opt move: pick two positions a < b and reverse route[a:b+1]."""
    n = len(route)
    a, b = sorted(random.sample(range(n), 2))
    return route[:a] + route[a:b + 1][::-1] + route[b + 1:]


# -----------------------------
# Simulated Annealing
# -----------------------------
def simulated_annealing(coords, T0=100, Tmin=1e-3, alpha=0.995, max_iter=20000):
    n = len(coords)
    dist_matrix = euclidean_distance_matrix(coords)

    # Initialize with a random route
    route = list(range(n))
    random.shuffle(route)
    best_route = route[:]
    best_cost = tour_length(route, dist_matrix)
    current_cost = best_cost

    T = T0
    history = [best_cost]

    for it in range(max_iter):
        # Generate a neighbor by 2-opt
        candidate = two_opt_swap(route)
        candidate_cost = tour_length(candidate, dist_matrix)
        delta = candidate_cost - current_cost

        # Acceptance criterion (Metropolis)
        if delta < 0 or random.random() < math.exp(-delta / T):
            route = candidate
            current_cost = candidate_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_route = route[:]

        # Store history
        history.append(best_cost)

        # Cool down (geometric schedule)
        T *= alpha
        if T < Tmin:
            break

    return best_route, best_cost, history


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    # Generate random cities
    np.random.seed(42)
    random.seed(42)          # seeds Python's random too, so results repeat
    coords = np.random.rand(30, 2) * 100       # 30 cities in 100x100 grid

    best_route, best_cost, history = simulated_annealing(coords)

    print(f"Best tour length: {best_cost:.2f}")

    # Plot progress
    plt.figure(figsize=(6, 4))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Best Tour Length")
    plt.title("Simulated Annealing Progress")
    plt.show()

    # Plot final tour
    plt.figure(figsize=(6, 6))
    ordered_coords = coords[best_route + [best_route[0]]]
    plt.plot(ordered_coords[:, 0], ordered_coords[:, 1], 'o-', markersize=8)
    plt.title(f"Final Tour (Length = {best_cost:.2f})")
    plt.show()
