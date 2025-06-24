import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math

T = 1000 # Time horizon
C = 250 # Capacity
price_classes = [2000, 1000, 500]
base_lambda = {
    2000: 0.025,
    1000: 0.25,
    500: 2.5
}

def f(t, k=2/T):
    return math.exp(-k * t)

def lambda_t_price(t, price):
    if price == price_classes[-1]:
        return base_lambda[price] * f(t)
    elif price == price_classes[1]:
        return base_lambda[price]
    else:
        return base_lambda[price] * (1/f(t))

def sale_probability(t, price):
    lam_p = lambda_t_price(t, price)
    return 1 - math.exp(-lam_p)

def f_vectorized(t_array, k=2/T):
    return np.exp(-k * t_array)

def lambda_t_price_vectorized(t_array, price):
    # Vectorized time-dependent arrival rate for a given price
    if price == 500:
        return base_lambda[price] * f_vectorized(t_array)
    elif price == 1000:
        return base_lambda[price] * np.ones_like(t_array)
    elif price == 2000:
        return base_lambda[price] / f_vectorized(t_array)
    else:
        raise ValueError("Unsupported price class")

def sale_probability_vectorized(t_array, price):
    # Vectorized probability of at least one arrival if this price is chosen
    lam_p = lambda_t_price_vectorized(t_array, price)
    return 1 - np.exp(-lam_p)

# Compute sale probabilities for all prices over time
t_array = np.arange(1, T + 1)
sale_probabilities = {}
for price in price_classes:
    sale_probabilities[price] = sale_probability_vectorized(t_array, price)

plt.figure(figsize=(14, 8))
for price in price_classes:
    plt.plot(t_array, sale_probabilities[price], label=f'Price ${price}')
plt.xlabel('Time (t)')
plt.ylabel('Sale Probability')
plt.title('Sale Probabilities Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def summary_stats(samples: np.ndarray) -> tuple[float, float]:
    mean_val = float(np.mean(samples))
    std_val  = float(np.std(samples, ddof=1)) # ddof=1 for unbiased
    return mean_val, std_val

# Initialize DP arrays
V = np.zeros((T + 1, C + 1))
optimal_policy = np.zeros((T, C + 1))

print("Starting DP")

for t in range(T, 0, -1):
    if t % 100 == 0 or t == 1:
        print(f"Processing time period: {t}/{T}")
    for x in range(1, C + 1):
        expected_revenues = []
        for p in price_classes:
            p_sale = sale_probability(t, p)
            # Expected revenue if choose price p:
            # sale: p_sale * (p + V[t][x-1])
            # no sale: (1 - p_sale)*V[t][x]
            exp_revenue = p_sale * (p + V[t][x - 1]) + (1 - p_sale) * V[t][x]
            expected_revenues.append(exp_revenue)

        max_revenue = max(expected_revenues)
        best_price = price_classes[np.argmax(expected_revenues)]
        V[t - 1][x] = max_revenue
        optimal_policy[t - 1][x] = best_price

print("DP completed")

expected_max_reward = V[0][C]
print("\nExpected Max Reward (DP):", expected_max_reward)

print("\nStarting simulations")
np.random.seed(42) # set seed to be able to reproduce results
num_simulations = 1000
rewards = []

# Initialize array to store remaining capacities for each simulation
remaining_capacities = np.zeros((num_simulations, T + 1), dtype=int)

for sim in range(num_simulations):
    if (sim + 1) % 100 == 0 or sim == 0:
        print(f"Running simulation {sim + 1}/{num_simulations}")
    total_reward = 0
    remaining_capacity = C
    # Store the initial capacity
    remaining_capacities[sim, 0] = remaining_capacity

    for t_sim in range(T):
        if remaining_capacity <= 0:
            # Fill the rest of the time steps with zero remaining capacity
            remaining_capacities[sim, t_sim + 1] = 0
            continue
        chosen_price = optimal_policy[t_sim, remaining_capacity]
        p_sale_chosen = sale_probability(t_sim + 1, chosen_price)

        # Simulate based on p_sale_chosen
        if np.random.rand() < p_sale_chosen:
            total_reward += chosen_price
            remaining_capacity -= 1
        # Record the remaining capacity after this time step
        remaining_capacities[sim, t_sim + 1] = remaining_capacity

    rewards.append(total_reward)
print("Simulations completed")

average_reward, reward_std = summary_stats(np.array(rewards))
print(f"\nAverage revenue (simulations): {average_reward:,.2f} ")
print(f"Expected max revenue   (DP):     {expected_max_reward:,.2f}")

# Compute average remaining capacity over time
average_remaining_capacity = remaining_capacities.mean(axis=0)

# Visualization of the optimal policy with expected remaining capacity overlay
unique_prices = sorted(set(price_classes))
cmap_colors = ['blue', 'green', 'red']
cmap = ListedColormap(cmap_colors[:len(unique_prices)])
plt.figure(figsize=(14, 8))
plt.imshow(optimal_policy.T, aspect='auto', origin='lower', extent=[0, T, 0, C], cmap=cmap)
cbar = plt.colorbar(ticks=unique_prices)
cbar.ax.set_yticklabels([f"${price}" for price in unique_prices])
cbar.set_label("Optimal Price")
plt.title("Optimal Policy")
plt.xlabel("Time (T)")
plt.ylabel("Remaining Capacity (C)")

# Overlay the expected remaining capacity
plt.plot(range(T + 1), average_remaining_capacity, color='yellow', linewidth=2, label='Expected Remaining Capacity')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Histogram of revenues
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(rewards, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(average_reward, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {average_reward:.2f}')
plt.axvline(expected_max_reward, color='green', linestyle='dotted', linewidth=1.5,
            label=f'DP Expected: {expected_max_reward:.2f}')
plt.title(f"Revenues Over {num_simulations} Simulations")
plt.xlabel("Total Revenue")
plt.ylabel("Frequency")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

