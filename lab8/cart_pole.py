import gymnasium as gym
import numpy as np
import cma
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------
# CMA-ES Optimization for CartPole-v1 with Linear Policy
# --------------------------------------------------------------------------------
# **Problem Overview**:
# CartPole-v1 is a classic reinforcement learning control task.
# A pole is hinged on a cart that moves horizontally along a track. The agent
# must choose to push the cart left or right at each step to keep the pole
# balanced upright. An episode ends when the pole falls too far or the cart
# goes out of bounds. The goal is to maximize cumulative reward (one point
# per time-step until failure).
#
# **Gymnasium**:
# We use the [Gymnasium](https://gymnasium.farama.org/) library (a maintained
# fork of OpenAI Gym) for standardized RL environments and API compatibility.
# It provides simple `env.reset()` and `env.step()` interfaces along with
# a variety of benchmarks (CartPole, LunarLander, Atari, and more).
#
# **State Representation**:
# The environment returns a 4-dimensional observation:
# 1. Cart position
# 2. Cart velocity
# 3. Pole angle
# 4. Pole angular velocity
# To include a bias term in our linear policy, we append a constant feature of
# 1.0, yielding a 5-dimensional feature vector at each step.
#
# **Policy Parameters (`weights`)**:
# We use an affine (linear + bias) policy: a weight vector w \in \mathbb{R}^5.
# For each state feature vector x \in \mathbb{R}^5, compute activation u = w^T x.
# If u > 0, select action=1 (push right); otherwise, action=0 (push left).
# By learning the weight for each observation component and the bias,
# the policy effectively draws a separating hyperplane in feature space.
#
# **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy) is an evolution-based
# optimizer that iteratively samples, evaluates, and updates a multivariate
# Gaussian distribution over the policy parameters to maximize reward.
# ---------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Recommended Reading:
# - Cart Pole: https://gymnasium.farama.org/environments/classic_control/cart_pole/
# - Gymnasium Documentation: https://gymnasium.farama.org/
# - CMA-ES Tutorial: https://arxiv.org/abs/1604.00772
# ---------------------------------------------------------------------------------


def play_episode(env: gym.Env, weights: np.ndarray, render: bool = False) -> float:
    """
    Run one episode in `env` using a simple linear policy defined by `weights`.

    Args:
        env: A Gymnasium environment instance.
        weights: 1D array of length obs_dim+1 (including bias weight).
        render: Whether to display each time step visually.

    Returns:
        total_reward: Sum of rewards over the episode (higher is better).
    """
    obs, _ = env.reset()
    total_reward = 0.0
    terminated, truncated = False, False

    while not (terminated or truncated):
        features = np.append(obs, 1.0)
        activation = float(np.dot(weights, features))
        action = 1 if activation > 0.0 else 0
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if render:
            env.render()

    return total_reward


def evaluate_weights(
    weights: np.ndarray, env_name: str = "CartPole-v1", num_episodes: int = 5
) -> float:
    """
    Compute the average return of a linear policy over several episodes.
    Note: In some optimization problems, the objective function is noisy, so we
    may want to average over multiple episodes to get a more stable estimate.

    Args:
        weights: Policy parameter vector.
        env_name: Gym environment ID to test on.
        num_episodes: Number of complete runs to average.

    Returns:
        Mean total reward across episodes.
    """
    env = gym.make(env_name)
    rewards: list[float] = []
    for _ in range(num_episodes):
        r = play_episode(env, weights, render=False)
        rewards.append(r)
    env.close()
    return float(np.mean(rewards))


def optimize_policy(
    env_name: str = "CartPole-v1",
    dim: int = 5,
    popsize: int = 20,
    sigma: float = 1.0,
    max_gens: int = 100,
    episodes_per_eval: int = 5,
) -> tuple[list[float], np.ndarray]:
    """
    Optimize a linear policy using CMA-ES to maximize average reward.

    Args:
        env_name: ID of the Gym environment.
        dim: Number of policy parameters (state dim + bias).
        popsize: Number of candidate solutions per generation.
        sigma: Initial sampling std deviation for search.
        max_gens: Upper limit on optimization generations.
        episodes_per_eval: Episode count per candidate evaluation.

    Returns:
        best_rewards: List of highest average rewards by generation.
        best_weights: Optimized weight vector found by CMA-ES.
    """
    # TODO: Implement CMA-ES optimization
    # Use the cma.CMAEvolutionStrategy class to run the CMA-ES optimization
    # Note: CMA-ES runs minimization, so we need tonegate the fitness function
    initial_solution = np.zeros(dim)
    initial_negative_fitness = -evaluate_weights(
        initial_solution, env_name, num_episodes=episodes_per_eval
    )
    return np.full(max_gens, -1 * initial_negative_fitness), initial_solution


def plot_results(rewards: list[float]) -> None:
    generations = list(range(1, len(rewards) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(generations, rewards, marker="o")
    plt.title("CMA-ES Optimization Progress")
    plt.xlabel("Generation")
    plt.ylabel("Best Average Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    env_name = "CartPole-v1"
    obs_dim = gym.make(env_name).observation_space.shape[0]
    dim = obs_dim + 1  # include bias

    rewards, best_weights = optimize_policy(
        env_name=env_name,
        dim=dim,
        popsize=30,
        sigma=1.0,
        max_gens=10,
        episodes_per_eval=5,
    )

    plot_results(rewards)

    render_env = gym.make(env_name, render_mode="human")
    final_reward = play_episode(render_env, best_weights, render=True)
    print(f"Final evaluation reward: {final_reward:.2f}")
    render_env.close()


if __name__ == "__main__":
    main()
