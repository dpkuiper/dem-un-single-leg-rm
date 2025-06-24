import math
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

T, C = 1000, 250
REVEAL_TIME = 300
prices: List[int] = [500, 1000, 2000]
base_lambda = {2000: .025, 1000: .25, 500: 2.5}
COEFF = {"LOW": .7, "MED": 1.0, "HIGH": 1.3}
PROBS = {"LOW": 1/3, "MED": 1/3, "HIGH": 1/3}
NUM_SIM, RNG_SEED  = 1000, 42 # seed to reproduce results
nP = len(prices)

def f(t, k=2 / T):
    return np.exp(-k * t)

def lambda_base(t, p):
    if p == prices[0]:   return base_lambda[p] * f(t)
    if p == prices[1]:  return base_lambda[p] * np.ones_like(t)
    if p == prices[2]:  return base_lambda[p] / f(t)
    raise ValueError

def sale_prob(t, p, coeff):
    return 1.0 - math.exp(-lambda_base(t, p) * coeff)

shape_V = (T + 1, C + 1, nP)
shape_policy_index = (T, C + 1, nP)
V = {s: np.zeros(shape_V)                  for s in (*COEFF, "UNK")}
policy_idx = {s: np.zeros(shape_policy_index, dtype=np.int8) for s in (*COEFF, "UNK")}

# DP for LOW/MED/HIGH
for state, coeff in COEFF.items():
    for t in range(T, 0, -1):
        coeff_t = 1.0 if t < REVEAL_TIME else coeff
        for x in range(1, C + 1):
            for last in range(nP):
                best_val, best_idx = -float('inf'), last
                for idx in range(last, nP): # non-decreasing constraint
                    p  = prices[idx]
                    ps = sale_prob(t, p, coeff_t)
                    val = ps * (p + V[state][t, x-1, idx]) + (1-ps) * V[state][t, x, last]
                    if val > best_val:
                        best_val, best_idx = val, idx
                V[state][t-1, x, last] = best_val
                policy_idx[state][t - 1, x, last] = best_idx

# DP for UNK
for t in range(T, 0, -1):
    if t >= REVEAL_TIME:
        V["UNK"][t-1] = V["MED"][t-1] #for initialization
        policy_idx["UNK"][t - 1] = policy_idx["MED"][t - 1]
        continue
    for x in range(1, C + 1):
        for last in range(nP):
            best_val, best_idx = -float('inf'), last
            for idx in range(last, nP):
                p  = prices[idx]
                ps = sale_prob(t, p, 1.0)
                if t + 1 == REVEAL_TIME:
                    post_s = sum(PROBS[s] * V[s][t, x-1, idx] for s in COEFF)
                    post_n = sum(PROBS[s] * V[s][t, x,   last] for s in COEFF)
                else:
                    post_s = V["UNK"][t, x-1, idx]
                    post_n = V["UNK"][t, x,   last]
                val = ps * (p + post_s) + (1-ps) * post_n
                if val > best_val:
                    best_val, best_idx = val, idx
            V["UNK"][t-1, x, last] = best_val
            policy_idx["UNK"][t - 1, x, last] = best_idx

dp_expected = V["UNK"][0, C, 0]
print(f"DP expected value: {dp_expected:,.2f}")

t_full = np.arange(T + 1)
cmap_prices = ListedColormap(['blue', 'green', 'red'])
norm_idx    = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_prices.N)
branch_col  = {"LOW":"tab:orange","MED":"tab:blue","HIGH":"tab:green"}
avg_cap_branch = {}
rng_master = np.random.default_rng(RNG_SEED)

# simulation and individual plots
for branch in ("LOW", "MED", "HIGH"):
    coeff_after = COEFF[branch]
    rem = np.zeros((NUM_SIM, T + 1), dtype=int)
    rewards = []
    for sim in range(NUM_SIM):
        rng = np.random.default_rng(rng_master.integers(1 << 32))
        cap, tot, last_idx = C, 0, 0
        rem[sim, 0] = cap
        for t in range(T):
            if cap == 0:
                rem[sim, t+1:] = 0; break
            grid    = policy_idx["UNK"] if t < REVEAL_TIME else policy_idx[branch]
            coeff_t = 1.0 if t < REVEAL_TIME else coeff_after
            idx     = int(grid[t, cap, last_idx])
            price   = prices[idx]
            if rng.random() < sale_prob(t+1, price, coeff_t):
                tot += price; cap -= 1
            last_idx = idx
            rem[sim, t+1] = cap
        rewards.append(tot)

    avg_cap = rem.mean(0)
    avg_cap_branch[branch] = avg_cap
    avg_r, std_r = np.mean(rewards), np.std(rewards)
    print(f"{branch}: mean reward = {avg_r:,.2f}")

    # Optimal policy heat-map + remaining capacity
    plt.figure(figsize=(14,5.5))
    vis = np.copy(policy_idx["UNK"][:, :, 0])
    vis[REVEAL_TIME:, :] = policy_idx[branch][REVEAL_TIME:, :, 0]
    plt.imshow(vis.T, origin="lower", aspect="auto",
               extent=[0, T, 0, C], cmap=cmap_prices, norm=norm_idx)
    cbar = plt.colorbar(ticks=[0,1,2])
    cbar.ax.set_yticklabels(["$500", "$1000", "$2000"])
    cbar.set_label("Optimal Price")
    plt.axvline(REVEAL_TIME, color="k", ls="--")
    plt.plot(t_full, avg_cap, lw=2, color="yellow",
             label="Expected Remaining Capacity")
    plt.title(f"Optimal Policy (Non-Decr.) – {branch} demand")
    plt.xlabel("Time"); plt.ylabel("Capacity"); plt.legend(); plt.tight_layout(); plt.show()

    # Histogram of revenues
    plt.figure(figsize=(8, 5))
    plt.hist(rewards, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(avg_r,      color="red",   ls="--", lw=1.5,
                label=f"Mean: {avg_r:,.1f}")
    plt.axvline(dp_expected, color="green", ls=":", lw=1.5,
                label=f"DP Expected: {dp_expected:,.1f}")
    plt.title(f"Revenues Over {NUM_SIM} Simulations – {branch} demand")
    plt.xlabel("Total Revenue");  plt.ylabel("Frequency");  plt.legend()
    plt.grid(axis="y", alpha=0.4);  plt.tight_layout();  plt.show()

# Combined capacity plot
pre_cap = avg_cap_branch["MED"][:REVEAL_TIME+1]
plt.figure(figsize=(10,5))
plt.plot(t_full[:REVEAL_TIME+1], pre_cap, lw=2, color="black",
         label="Expected Capacity (t < reveal)")
for br, cap in avg_cap_branch.items():
    plt.plot(t_full[REVEAL_TIME:], cap[REVEAL_TIME:], lw=2,
             color=branch_col[br], label=f"{br} demand revealed")
plt.axvline(REVEAL_TIME, color="k", ls="--")
plt.title("Expected Remaining Capacity (Non-Decreasing Prices)")
plt.xlabel("Time"); plt.ylabel("Remaining Capacity"); plt.legend()
plt.tight_layout(); plt.show()
