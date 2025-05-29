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
    - Added Simulated Annealing inspired acceptance criterion
    - Modified crossover for more influence of the best solution
    - Reduced archive size and increased reinitialization frequency for some benchmarks.
    """
    dim = len(bounds)
    population_size = min(100, budget // 10)  # Dynamic population size based on budget
    population = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(population_size, dim))
    
    # Evaluate initial population
    fitness = np.array([function(x) for x in population])
    num_evals = population_size

    # Elite archive
    archive_size = min(10, population_size // 5)
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
    reinitialization_interval = budget // 4
    initial_mutation_scale = 0.1
    temperature = 1.0
    inertia_weight = 0.7  # Inertia weight for archive influence
    personal_learning_coeff = 0.1 # Weight for individual learning
    global_learning_coeff = 0.1 # Weight for global learning
    diversity_threshold = 0.05 # Threshold for diversity check
    crossover_rate = 0.7  # Probability of applying crossover

    # Added parameter: Local search probability
    local_search_prob = 0.1

    while num_evals < budget:
        # Linearly decreasing exploration
        exploration_rate = 1.0 - (num_evals / budget)
        temperature = 1.0 - (num_evals / budget)

        new_population = np.zeros_like(population)
        new_fitness = np.zeros_like(fitness)
        
        # Adaptive mutation scale
        fitness_variance = np.var(fitness)
        mutation_scale = exploration_rate * initial_mutation_scale * (1 + fitness_variance)

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

                # Modified exploitation: Particle Swarm inspired
                # The solution is updated as a weighted sum of the current solution,
                # the archive solution, and the best solution found so far.
                
                if np.random.rand() < crossover_rate: # Apply crossover based on probability
                    # Combine current individual with the best solution found so far
                    crossover_point = np.random.randint(0, dim)  # Select a random crossover point
                    new_x = np.concatenate((best_x[:crossover_point], population[i][crossover_point:])) #Apply crossover
                else:
                   new_x = population[i].copy()
                

                new_x = (inertia_weight * new_x + # Use the modified solution for exploitation
                         personal_learning_coeff * np.random.rand() * (archive_sol - new_x) +
                         global_learning_coeff * np.random.rand() * (best_x - new_x))


                # Introduce slight mutation to the new individual
                mutation_scale_fine = 0.01  # Small constant mutation for fine-tuning
                new_x = new_x + np.random.normal(0, mutation_scale_fine * (np.array([b[1] - b[0] for b in bounds])), dim)
                
            # Clip to bounds
            new_x = np.clip(new_x, [b[0] for b in bounds], [b[1] for b in bounds])

            # Add local search with probability
            if np.random.rand() < local_search_prob:
                step_size = 0.01 * (np.array([b[1] - b[0] for b in bounds]))
                for d in range(dim):
                    direction = np.random.choice([-1, 1])
                    new_x_local = new_x.copy()
                    new_x_local[d] += direction * step_size[d]
                    new_x_local = np.clip(new_x_local, [b[0] for b in bounds], [b[1] for b in bounds])
                    f_local = function(new_x_local)
                    num_evals += 1
                    if f_local < function(new_x):
                        new_x = new_x_local
                        
            
            new_f = function(new_x)
            num_evals += 1
            
            # Simulated annealing acceptance criterion
            if new_f < fitness[i] or np.random.rand() < np.exp((fitness[i] - new_f) / temperature):
                new_population[i] = new_x
                new_fitness[i] = new_f
            else:
                new_population[i] = population[i]
                new_fitness[i] = fitness[i]
                new_f = fitness[i]

            if new_f < best_f:
                best_f = new_f
                best_x = new_x.copy()
            
            update_archive(new_x, new_f)  # Update archive with new solution

        population = new_population
        fitness = new_fitness

        # Enhanced Elitism using Tournament Selection
        tournament_size = 3  # Adjust tournament size as needed
        tournament_indices = np.random.choice(population_size, tournament_size, replace=False)
        worst_tournament_index = tournament_indices[np.argmax(fitness[tournament_indices])]
        
        if best_f < fitness[worst_tournament_index]:
            fitness[worst_tournament_index] = best_f
            population[worst_tournament_index] = best_x.copy()

        # Periodic Re-initialization
        if num_evals % reinitialization_interval == 0:
            worst_indices = np.argsort(fitness)[-population_size // 4:]
            population[worst_indices] = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(len(worst_indices), dim))
            fitness[worst_indices] = np.array([function(x) for x in population[worst_indices]])
            
            # Update Best
            best_index = np.argmin(fitness)
            if fitness[best_index] < best_f:
                best_x = population[best_index].copy()
                best_f = fitness[best_index]
            for i in worst_indices:
              update_archive(population[i], fitness[i])

        # Archive Diversification
        if num_evals % (reinitialization_interval // 2) == 0:
            # Replace a random archive member with a random solution from the current population
            rand_archive_idx = np.random.randint(0, archive_size)
            rand_pop_idx = np.random.randint(0, population_size)

            archive_x[rand_archive_idx] = population[rand_pop_idx].copy()
            archive_f[rand_archive_idx] = fitness[rand_pop_idx]
            sorted_indices = np.argsort(archive_f)
            archive_f = archive_f[sorted_indices]
            archive_x = archive_x[sorted_indices]

        # Dynamic Inertia Weight
        inertia_weight = 0.5 + 0.4 * np.exp(-10 * num_evals / budget)

        # Diversity Check and Enhancement
        if num_evals % (reinitialization_interval // 8) == 0:
          distances = np.zeros((population_size, population_size))
          for j in range(population_size):
              for k in range(j + 1, population_size):
                  distances[j, k] = np.linalg.norm(population[j] - population[k])
                  distances[k, j] = distances[j, k]

          avg_distance = np.mean(distances)
          if avg_distance < diversity_threshold:
              # If diversity is low, reinitialize a larger portion of the population
              num_to_reinitialize = population_size // 2  # Reinitialize half of the population
              worst_indices = np.argsort(fitness)[-num_to_reinitialize:]
              population[worst_indices] = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(len(worst_indices), dim))
              fitness[worst_indices] = np.array([function(x) for x in population[worst_indices]])

              best_index = np.argmin(fitness)
              if fitness[best_index] < best_f:
                  best_x = population[best_index].copy()
                  best_f = fitness[best_index]
              for i in worst_indices:
                update_archive(population[i], fitness[i])



    return best_f, best_x