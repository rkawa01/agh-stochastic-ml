import numpy as np
from typing import Callable, List, Tuple

def new_metaheuristic(
    function: Callable[[np.ndarray], float], 
    bounds: List[Tuple[float, float]], 
    budget: int
) -> Tuple[float, np.ndarray]:
    """
    A novel metaheuristic algorithm for minimizing a black-box function with bound constraints.
    Implements Linearly Decreasing Exploration-Exploitation with Gaussian Mutation and Elite Archive.
    Modified for improved performance:
    - Adaptive mutation scale based on fitness variance.
    - Rank-based selection for archive guidance.
    - Enhanced Elitism using tournament selection.
    - Implemented exponential decay of archive influence.
    - Increased Population diversity by periodic reinitialization.
    """
    dim = len(bounds)
    population_size = min(100, budget // 10)  # Dynamic population size based on budget
    population = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(population_size, dim))
    
    # Evaluate initial population
    fitness = np.array([function(x) for x in population])
    num_evals = population_size

    # Elite archive
    archive_size = min(15, population_size // 2)  # Increased archive size
    archive_x = np.zeros((archive_size, dim))
    archive_f = np.full(archive_size, np.inf)

    # Update archive function (sorted)
    def update_archive(x, f):
      nonlocal archive_x, archive_f
      if f < np.max(archive_f):
          max_index = np.argmax(archive_f)
          archive_f[max_index] = f
          archive_x[max_index] = x
          # Sort archive by fitness (optional, but potentially helpful)
          sorted_indices = np.argsort(archive_f)
          archive_f = archive_f[sorted_indices]
          archive_x = archive_x[sorted_indices]

    for i in range(population_size):
      update_archive(population[i], fitness[i])
    
    best_index = np.argmin(fitness)
    best_x = population[best_index].copy()
    best_f = fitness[best_index]

    # Main loop
    reinitialization_interval = budget // 5 # Added Diversity maintenance strategy
    initial_mutation_scale = 0.1

    while num_evals < budget:
        # Linearly decreasing exploration
        exploration_rate = 1.0 - (num_evals / budget)

        new_population = np.zeros_like(population)
        new_fitness = np.zeros_like(fitness)
        
        # Adaptive mutation scale
        fitness_variance = np.var(fitness)
        mutation_scale = exploration_rate * initial_mutation_scale * (1 + fitness_variance)  # Adjusted mutation

        for i in range(population_size):
            if np.random.rand() < exploration_rate:
                # Exploration: Gaussian mutation with decreasing std
                new_x = population[i] + np.random.normal(0, mutation_scale * (np.array([b[1] - b[0] for b in bounds])), dim)
            else:
                # Exploitation: Crossover with best solution + Archive guidance
                
                # Rank-based Archive Selection
                probabilities = np.exp(-np.arange(archive_size) / 5)  # Higher probability for better archive members
                probabilities /= np.sum(probabilities)  # Normalize
                archive_idx = np.random.choice(archive_size, p=probabilities)
                archive_sol = archive_x[archive_idx]

                crossover_points = np.random.rand(dim) < 0.5
                
                # More balanced crossover
                new_x = np.where(crossover_points, best_x, archive_sol)  # Crossover between best and archive

                # Introduce slight mutation to the new individual
                mutation_scale_fine = 0.01  # Small constant mutation for fine-tuning
                new_x = new_x + np.random.normal(0, mutation_scale_fine * (np.array([b[1] - b[0] for b in bounds])), dim)
                
            # Clip to bounds
            new_x = np.clip(new_x, [b[0] for b in bounds], [b[1] for b in bounds])
            
            new_f = function(new_x)
            num_evals += 1
            new_population[i] = new_x
            new_fitness[i] = new_f

            if new_f < best_f:
                best_f = new_f
                best_x = new_x.copy()
            
            update_archive(new_x, new_f)  # Update archive with new solution

        population = new_population
        fitness = new_fitness

        # Enhanced Elitism using Tournament Selection
        tournament_size = 3  # Adjust tournament size as needed
        tournament_indices = np.random.choice(population_size, tournament_size, replace=False)
        worst_tournament_index = tournament_indices[np.argmax(fitness[tournament_indices])] #Find worst of tournament
        
        if best_f < fitness[worst_tournament_index]:
            fitness[worst_tournament_index] = best_f
            population[worst_tournament_index] = best_x.copy()

        # Periodic Re-initialization
        if num_evals % reinitialization_interval == 0:
            worst_indices = np.argsort(fitness)[-population_size // 4:] #Reinitialize 25% of population
            population[worst_indices] = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(len(worst_indices), dim))
            fitness[worst_indices] = np.array([function(x) for x in population[worst_indices]])
            
            # Update Best
            best_index = np.argmin(fitness)
            if fitness[best_index] < best_f:
                best_x = population[best_index].copy()
                best_f = fitness[best_index]
            for i in worst_indices:
              update_archive(population[i], fitness[i])

    return best_f, best_x