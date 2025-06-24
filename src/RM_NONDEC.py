import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math

T = 1000
C = 250
price_classes = [500, 1000, 2000]
n_prices = len(price_classes)
base_lambda = {
    500: 2.5,
    1000: 0.25,
    2000: 0.025
}

def f(t, k=2 / T):
    return math.exp(-k * t)

def lambda_t_price(t: int, price: int) -> float:
    if price == price_classes[0]:
        return base_lambda[price] * f(t)
    if price == price_classes[1]:
        return base_lambda[price]
    else:
        return base_lambda[price] / f(t)
    raise ValueError

def sale_prob(t: int, price: int) -> float:
    return 1 - math.exp(-lambda_t_price(t, price))

t_arr = np.arange(1, T + 1)
def lambda_vec(t_array, price):
    if price == 500:
        return base_lambda[price] * np.exp(-2 * t_array / T)
    if price == 1000:
        return base_lambda[price] * np.ones_like(t_array)
    if price == 2000:
        return base_lambda[price] / np.exp(-2 * t_array / T)

def sale_prob_vec(t_array, price):
    return 1 - np.exp(-lambda_vec(t_array, price))

print("Starting DP")
V = np.zeros((T + 1, C + 1, n_prices))
policy_idx = np.zeros((T,     C + 1, n_prices), dtype=int)

for t in range(T, 0, -1):
    if t % 100 == 0 or t == 1:
        print(f"Processing time period {t}/{T}")
    for seats in range(1, C + 1):
        for last_idx in range(n_prices):
            best_val, best_dec = -np.inf, last_idx
            for dec_idx in range(last_idx, n_prices): # non-decreasing constraint
                p       = price_classes[dec_idx]
                p_sale  = sale_prob(t, p)
                val     = p_sale * (p + V[t][seats - 1][dec_idx]) \
                          + (1 - p_sale) * V[t][seats][dec_idx]
                if val > best_val:
                    best_val, best_dec = val, dec_idx
            V[t - 1][seats][last_idx]       = best_val
            policy_idx[t - 1][seats][last_idx] = best_dec
print("DP completed")

dp_expectation = [V[0][C][idx] for idx in range(n_prices)]
for idx, ev in enumerate(dp_expectation):
    print(f"DP expected revenue if start at ${price_classes[idx]:4}: {ev:,.2f}")

def simulate(initial_idx, sims=1000, seed=42): # set seed to be able to reproduce results
    np.random.seed(seed)
    rewards = []
    rem_cap_all = np.zeros((sims, T + 1), dtype=int)
    for s in range(sims):
        seats, last_idx, reward = C, initial_idx, 0
        rem_cap_all[s, 0] = seats
        for t in range(T):
            if seats == 0:
                rem_cap_all[s, t + 1:] = 0
                break
            dec_idx = policy_idx[t, seats, last_idx]
            price = price_classes[dec_idx]
            if np.random.rand() < sale_prob(t + 1, price):
                seats -= 1; reward += price
            rem_cap_all[s, t + 1] = seats
            last_idx = dec_idx
        rewards.append(reward)
    return np.array(rewards), rem_cap_all


state_labels   = ["LOW start ($500)", "MID start ($1 000)", "HIGH start ($2 000)"]
cmap_prices    = ListedColormap(["#3B4CC0", "#78D153", "#D71E16"])  # blue-green-red

for start_idx in range(n_prices):
    print(f"\nSTARTING STATE: {state_labels[start_idx]}")

    # Simulations
    rewards, remcap = simulate(start_idx)
    mean_reward = rewards.mean()
    print(f"    mean simulated revenue: {mean_reward:,.2f}")

    # Policy heat-map
    plt.figure(figsize=(14, 8))
    policy_price = np.vectorize(lambda i: price_classes[i])(
        policy_idx[:, :, start_idx])
    plt.imshow(policy_price.T, origin='lower', aspect='auto',
               extent=[0, T, 0, C], cmap=cmap_prices,
               vmin=min(price_classes), vmax=max(price_classes))
    cbar = plt.colorbar(ticks=price_classes)
    cbar.ax.set_yticklabels([f"${p}" for p in price_classes])
    cbar.set_label("Optimal Price")
    plt.title(f"Optimal Policy (Non-Decreasing Price contraint)")
    plt.xlabel("Time (T)"); plt.ylabel("Remaining Capacity (C)")

    avg_rem = remcap.mean(axis=0)
    plt.plot(range(T + 1), avg_rem, color='yellow', linewidth=2,
             label="Expected Remaining Capacity")
    plt.legend(loc='upper right'); plt.tight_layout(); plt.show()

    # Reward histogram
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=30, edgecolor='black', alpha=.7)
    plt.axvline(mean_reward, linestyle='--', linewidth=1.5,
                label=f"Mean: {mean_reward:,.0f}")
    plt.axvline(dp_expectation[start_idx], color='red', linestyle=':',
                linewidth=1.5, label=f"DP Expected: {dp_expectation[start_idx]:,.0f}")
    plt.title(f"Revenues Over 1000 Simulations (Non-Decreasing Price constraint)")
    plt.xlabel("Total Revenue"); plt.ylabel("Frequency")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()

