import numpy as np
import random
import json
from datetime import datetime
from joblib import Parallel, delayed
from trading_bot.tools.simulator_engine import run_simulation
from trading_bot.config.parameters import INDICATOR_NAMES as FEATURES  # ✅ Single source of truth

# === Optimization Parameters ===
POPULATION_SIZE = 50
GENERATIONS = 30
MUTATION_RATE = 0.1
ELITISM_COUNT = 1  # ✅ Number of top individuals carried forward each generation

# === Fitness Function ===
def fitness(weights, full_df, active_features):
    weight_dict = dict(zip(active_features, weights))
    result = run_simulation(full_df, weight_dict)
    total = sum(result.final.values())  # ✅ use result.final.values()
    return total

# === Optimizer Entry ===
def optimize_weights(train_df):
    active_features = [f for f in FEATURES if f in train_df.columns]
    print(f"🧪 Optimizing {len(active_features)} features out of {len(FEATURES)} total...")

    # === Initialize population ===
    population = [
        [random.uniform(0, 1) for _ in active_features]
        for _ in range(POPULATION_SIZE)
    ]

    # === Genetic Algorithm Loop ===
    for gen in range(GENERATIONS):
        # ✅ Parallel fitness evaluation
        fitness_scores = Parallel(n_jobs=-1)(
            delayed(fitness)(ind, train_df, active_features) for ind in population
        )

        best_equity = max(fitness_scores)
        print(f"Generation {gen+1}: Best Equity = {best_equity:.2f}")

        # === Elitism: retain top performers ===
        sorted_pairs = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        elites = [x[0] for x in sorted_pairs[:ELITISM_COUNT]]

        parents = select_parents(population, fitness_scores)
        next_gen = elites.copy()

        while len(next_gen) < POPULATION_SIZE:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            next_gen.append(child)

        population = next_gen

    # === Final selection ===
    final_fitness_scores = Parallel(n_jobs=-1)(
        delayed(fitness)(ind, train_df, active_features) for ind in population
    )
    best_idx = np.argmax(final_fitness_scores)
    best_weights = population[best_idx]
    best_equity = final_fitness_scores[best_idx]
    best_weights_dict = dict(zip(active_features, best_weights))

    print("\n✅ Optimization Complete")
    print(f"🏆 Best Equity: {best_equity:.2f}")
    print(json.dumps(best_weights_dict, indent=4))

    return best_weights_dict

# === Helper Functions ===
def select_parents(population, fitness_scores):
    sorted_pairs = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_pairs[:int(POPULATION_SIZE / 2)]]

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    return parent1[:crossover_point] + parent2[crossover_point:]

def mutate(individual):
    return [
        min(1.0, max(0.0, gene + np.random.normal(0, 0.1))) if random.random() < MUTATION_RATE else gene
        for gene in individual
    ]