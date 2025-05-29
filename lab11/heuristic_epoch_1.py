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
    while num_evals < budget:
        # Linearly decreasing exploration
        exploration_rate = 1.0 - (num_evals / budget)

        new_population = np.zeros_like(population)
        new_fitness = np.zeros_like(fitness)
        
        for i in range(population_size):
            if np.random.rand() < exploration_rate:
                # Exploration: Gaussian mutation with decreasing std
                mutation_scale = exploration_rate * 0.1 # Reduce mutation over time
                new_x = population[i] + np.random.normal(0, mutation_scale * (np.array([b[1] - b[0] for b in bounds])), dim)
            else:
                # Exploitation: Crossover with best solution + Archive guidance
                
                # Select a random solution from the archive (if not empty)
                if archive_f[0] < np.inf:
                    archive_idx = np.random.randint(0, archive_size)
                    archive_sol = archive_x[archive_idx]
                else:
                  archive_sol = best_x
                
                crossover_points = np.random.rand(dim) < 0.5
                
                # More balanced crossover
                new_x = np.where(crossover_points, best_x, archive_sol)  # Crossover between best and archive

                # Introduce slight mutation to the new individual
                mutation_scale = 0.01  # Small constant mutation for fine-tuning
                new_x = new_x + np.random.normal(0, mutation_scale * (np.array([b[1] - b[0] for b in bounds])), dim)
                
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

        # Elitism: Keep the best from the old population for the next generation
        elite_idx = np.argmax(fitness) #Find worst
        if best_f < fitness[elite_idx]:
            fitness[elite_idx] = best_f
            population[elite_idx] = best_x.copy()

    return best_f, best_x