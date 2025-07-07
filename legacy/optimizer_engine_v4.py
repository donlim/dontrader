# trading_bot/tools/optimizer_engine_v4.py

import numpy as np
import random
import json
from datetime import datetime
from tools.simulator_engine_v4 import run_simulation

# === Optimization Parameters ===

FEATURES = [
    "DELTA_FLOW", "BOOK_IMB", "PRESSURE", "SLOPE", "LIQUIDITY_GAP",
    "SPREAD", "VOLATILITY", "EMA10", "EMA50", "MACD", "RSI",
    "MOMENTUM", "ATR", "VWAP", "OBV", "AD", "STDDEV", "SKEW", "KURTOSIS"
]

POPULATION_SIZE = 50
GENERATIONS = 30
MUTATION_RATE = 0.1

# === Fitness Function ===

def fitness(weights, train_df):
    weight_dict = dict(zip(FEATURES, weights))
    final_equity = run_simulation(train_df, weight_dict)
    return final_equity

# === Optimizer Entry ===

def optimize_weights(train_df):
    population = [
        [random.uniform(0, 1) for _ in FEATURES]
        for _ in range(POPULATION_SIZE)
    ]

    for gen in range(GENERATIONS):
        fitness_scores = [fitness(ind, train_df) for ind in population]
        best_equity = max(fitness_scores)
        print(f"Generation {gen+1}: Best Equity = {best_equity:.2f}")

        parents = select_parents(population, fitness_scores)
        next_gen = []

        while len(next_gen) < POPULATION_SIZE:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            next_gen.append(child)

        population = next_gen

    final_fitness_scores = [fitness(ind, train_df) for ind in population]
    best_idx = np.argmax(final_fitness_scores)
    best_weights = population[best_idx]
    best_equity = final_fitness_scores[best_idx]
    best_weights_dict = dict(zip(FEATURES, best_weights))

    print("\n✅ Optimization Complete")
    print(f"Best Equity: {best_equity:.2f}")
    print(json.dumps(best_weights_dict, indent=4))

    return best_weights_dict

# === Helper Functions ===

def select_parents(population, fitness_scores):
    sorted_pairs = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_pairs[:int(POPULATION_SIZE/2)]]

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(FEATURES) - 1)
    return parent1[:crossover_point] + parent2[crossover_point:]

def mutate(individual):
    return [
        min(1.0, max(0.0, gene + np.random.normal(0, 0.1))) if random.random() < MUTATION_RATE else gene
        for gene in individual
    ]