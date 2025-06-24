import math
from typing import Dict, List, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

T, C = 1000, 250
REVEAL_TIME = 300
price_classes: List[int] = [2000, 1000, 500]

base_lambda: Dict[int, float] = {2000: 0.025, 1000: 0.25, 500: 2.5}
COEFF = {"LOW": 0.7, "MED": 1.0, "HIGH": 1.3}
PROBS = {"LOW": 1/3, "MED": 1/3, "HIGH": 1/3}
NUM_SIM, RNG_SEED = 1_000, 42 # set seed to reproduce result

def f(t: Union[int, np.ndarray], k: float = 2 / T):
    return np.exp(-k * t)

def lambda_base(t, price):
    if price == 500:   return base_lambda[price] * f(t)
    if price == 1000:  return base_lambda[price] * np.ones_like(t)
    if price == 2000:  return base_lambda[price] / f(t)
    raise ValueError

def sale_prob(t: int, price: int, coeff: float) -> float:
    return 1.0 - math.exp(-lambda_base(t, price) * coeff)

V = {s: np.zeros((T + 1, C + 1)) for s in (*COEFF, "UNK")}
policy = {s: np.zeros((T, C + 1), dtype=int) for s in (*COEFF, "UNK")}

print("Starting DPs")
# DP for LOW/MED/HIGH
for state, coeff in COEFF.items():
    for t in range(T, 0, -1):
        coeff_t = 1.0 if t < REVEAL_TIME else coeff
        for x in range(1, C + 1):
            best_val, best_p = -float('inf'), None
            for p in price_classes:
                ps = sale_prob(t, p, coeff_t)
                val = ps * (p + V[state][t][x-1]) + (1-ps) * V[state][t][x]
                if val > best_val:
                    best_val, best_p = val, p
            V[state][t-1][x] = best_val
            policy[state][t - 1][x] = best_p

# DP for UNK
for t in range(T, 0, -1):
    if t >= REVEAL_TIME:
        V["UNK"][t-1] = V["MED"][t-1] #for initialization
        policy["UNK"][t - 1] = policy["MED"][t - 1]
        continue

    for x in range(1, C + 1):
        best_val, best_p = -float('inf'), None
        for p in price_classes:
            ps = sale_prob(t, p, 1.0) # pre-reveal coeff=1
            if t + 1 == REVEAL_TIME: # expectation step
                post_s = sum(PROBS[s] * V[s][t][x-1] for s in COEFF)
                post_n = sum(PROBS[s] * V[s][t][x  ] for s in COEFF)
            else:
                post_s = V["UNK"][t][x-1]
                post_n = V["UNK"][t][x]
            val = ps * (p + post_s) + (1-ps) * post_n
            if val > best_val:
                best_val, best_p = val, p
        V["UNK"][t-1][x] = best_val
        policy["UNK"][t - 1][x] = best_p

dp_expected = V["UNK"][0][C]
print(f"Expected DP value (before t = 0): {dp_expected:,.2f}")

vec_t = np.arange(T)
unique_prices = sorted(price_classes)
cmap_prices = ListedColormap(['blue', 'green', 'red'])
norm_prices = BoundaryNorm([0, 750, 1500, 2500], cmap_prices.N)
rng_master = np.random.default_rng(RNG_SEED)

# store expected capacity paths to plot them together later
avg_cap_branch = {}

for branch in ("LOW", "MED", "HIGH"):
    coeff_after = COEFF[branch]

    # simulation
    rem = np.zeros((NUM_SIM, T + 1), dtype=int)
    rewards = []
    for sim in range(NUM_SIM):
        rng = np.random.default_rng(rng_master.integers(1 << 32))
        cap, tot = C, 0
        rem[sim, 0] = cap
        for t in range(T):
            if cap == 0:
                rem[sim, t+1:] = 0
                break
            grid     = policy["UNK"] if t < REVEAL_TIME else policy[branch]
            coeff_t  = 1.0 if t < REVEAL_TIME else coeff_after
            price    = int(grid[t, cap])
            if rng.random() < sale_prob(t+1, price, coeff_t):
                tot += price
                cap -= 1
            rem[sim, t+1] = cap
        rewards.append(tot)

    avg_r, std_r = float(np.mean(rewards)), float(np.std(rewards))
    avg_cap, std_cap = rem.mean(0), rem.std(0)

    # store for combined capacity figure
    avg_cap_branch[branch] = avg_cap
    print(f"{branch}:  mean reward = {avg_r:,.2f}")

    # Optimal policy + expected remaining capacity heatmap plot
    plt.figure(figsize=(14, 5.5))
    policy_vis = np.copy(policy["UNK"])
    policy_vis[REVEAL_TIME:, :] = policy[branch][REVEAL_TIME:, :]
    plt.imshow(policy_vis.T, origin="lower", aspect="auto",
               extent=[0, T, 0, C], cmap=cmap_prices, norm=norm_prices)
    cbar = plt.colorbar(ticks=unique_prices)
    cbar.ax.set_yticklabels([f"${p}" for p in unique_prices])
    cbar.set_label("Optimal Price")
    plt.axvline(REVEAL_TIME, color="k", ls="--")
    plt.plot(np.arange(T+1), avg_cap, lw=2, color="yellow",
             label="Expected Remaining Capacity")
    plt.title(f"Optimal Policy – {branch} demand")
    plt.xlabel("Time (T)");  plt.ylabel("Remaining Capacity (C)")
    plt.legend();  plt.tight_layout();  plt.show()

    # Histogram of revenues
    plt.figure(figsize=(8, 5))
    plt.hist(rewards, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(avg_r,       color="red",   ls="--", lw=1.5,
                label=f"Mean: {avg_r:,.1f}")
    plt.axvline(dp_expected, color="green", ls=":", lw=1.5,
                label=f"DP Expected: {dp_expected:,.1f}")
    plt.title(f"Revenues Over 1000 Simulations – {branch} demand")
    plt.xlabel("Total Revenue");  plt.ylabel("Frequency");  plt.legend()
    plt.grid(axis="y", alpha=0.4);  plt.tight_layout();  plt.show()

# Combined remaining capacities
branch_color = {"LOW": "tab:orange", "MED": "tab:blue", "HIGH": "tab:green"}
pre_cap = avg_cap_branch["MED"][:REVEAL_TIME + 1]   # same for all branches
t_grid = np.arange(T + 1)
plt.figure(figsize=(10, 5))
plt.plot(t_grid[:REVEAL_TIME + 1], pre_cap,
         lw=2, color="black", label="Expected Capacity (t < reveal)")
for br, cap in avg_cap_branch.items():
    plt.plot(t_grid[REVEAL_TIME:], cap[REVEAL_TIME:], lw=2,
             color=branch_color[br], label=f"{br} demand revealed")
plt.axvline(REVEAL_TIME, color="k", ls="--", lw=1)
plt.title("Expected Remaining Capacity")
plt.xlabel("Time (T)")
plt.ylabel("Remaining Capacity")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
