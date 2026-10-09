"""
Reinforcement Learning for Finance & Trading
Optimal Trade Execution: Linear SARSA (RBF features) vs Deep Q-Network (PyTorch)
------------------------------------------------------------------------------
Starter code for the student assignment in ASSIGNMENT.md.

Scenario: a trading desk must liquidate a large parent order (e.g. 500,000
shares) before the market closes. Each decision interval the execution agent
chooses how aggressively to trade the next child order:

  0 = Passive    : post a limit order  -> small size, earns the spread, may not fill
  1 = Moderate   : TWAP-style slice    -> medium size, pays spread + small impact
  2 = Aggressive : market order sweep  -> large size, pays spread + large impact

State  x in [0, 1] = fraction of the parent order already executed.
Reward (in basis points of the parent order's arrival value):
  - execution cost of the child order   (spread + temporary market impact)
  + mark-to-market P&L on the unsold inventory (random price moves)
  - risk-aversion penalty on unsold inventory (volatility exposure)
If the order is not complete by the close (max_steps), the remainder is
force-liquidated with a single market order at a steep impact cost.

This is a stylised, single-asset version of the Almgren-Chriss optimal
execution problem: trading fast costs impact, trading slow costs risk.

Requirements: numpy, pandas, matplotlib>=3.10 (+ torch for the DQN agent)
Run:          python trade_execution_rl.py
"""
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)  # used by the replay buffer in Task 2

# ------------------------------------------------------------------------------
# 0. Market microstructure parameters (students will vary these in Task 3)
# ------------------------------------------------------------------------------
MKT = dict(
    half_spread_bps=5.0,  # cost of crossing half the bid-ask spread
    impact_bps=200.0,     # temporary impact coefficient: cost = impact * q per unit
    sigma_bps=3.0,        # per-interval price volatility (bps)
    risk_aversion=2.0,    # penalty per interval for holding 100% of the order
    max_steps=60,         # decision intervals until the close (e.g. 60 x 6.5 min)
)

ACTION_NAMES = ["Passive (limit)", "Moderate (TWAP)", "Aggressive (market)"]
ACTION_SIZE = np.array([0.03, 0.06, 0.15])    # fraction of parent order per child order
FILL_PROB = np.array([0.50, 1.00, 1.00])      # limit orders only fill sometimes
SPREAD_DIR = np.array([-1.0, 1.0, 1.0])       # -1 = earn the spread, +1 = pay it
HAS_IMPACT = np.array([0.0, 1.0, 1.0])
N_ACTIONS = 3


def slice_cost(q, action, mkt):
    """Execution cost (bps of parent order) of selling a child order of size q."""
    return q * (SPREAD_DIR[action] * mkt["half_spread_bps"]
                + HAS_IMPACT[action] * mkt["impact_bps"] * q)


# ------------------------------------------------------------------------------
# 1. Trade-execution environment (replaces the original "wind" track)
# ------------------------------------------------------------------------------
def step_env(state, action, t, mkt, rng):
    remaining = 1.0 - state

    # Child order size (cannot sell more than what is left); limit orders may miss
    filled = rng.random() < FILL_PROB[action]
    q = min(ACTION_SIZE[action], remaining) if filled else 0.0

    exec_cost = slice_cost(q, action, mkt)
    next_state = min(1.0, state + q)
    inventory = 1.0 - next_state

    # Market risk on unsold inventory during the interval
    mtm_pnl = inventory * rng.normal(0.0, mkt["sigma_bps"])
    risk_pen = mkt["risk_aversion"] * inventory

    done = next_state >= 1.0 - 1e-9
    forced_cost = 0.0
    if not done and t >= mkt["max_steps"]:
        # Market close: dump the remainder in one block with a market order
        forced_cost = slice_cost(inventory, 2, mkt)
        next_state = 1.0
        done = True

    reward = -exec_cost - forced_cost + mtm_pnl - risk_pen
    return dict(next_state=next_state, reward=reward, done=done,
                exec_cost=exec_cost + forced_cost, mtm_pnl=mtm_pnl,
                forced=forced_cost > 0)


# ------------------------------------------------------------------------------
# 2. High-density RBF feature map
# ------------------------------------------------------------------------------
RBF_CENTERS = np.linspace(0.0, 1.0, 15)
RBF_SIGMA = 0.08


def get_dense_features(state):
    phi = np.exp(-(state - RBF_CENTERS) ** 2 / (2 * RBF_SIGMA ** 2))
    return phi / phi.sum()  # normalised so features sum to 1


def select_action_linear(W, phi, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    return int(np.argmax(phi @ W))


# ------------------------------------------------------------------------------
# 3. On-policy linear SARSA execution agent
# ------------------------------------------------------------------------------
def train_sarsa_linear(mkt, episodes=600, alpha=0.05, gamma=0.99, epsilon=0.1, rng=rng):
    W = np.zeros((len(RBF_CENTERS), N_ACTIONS))
    history = np.zeros(episodes)

    for ep in range(episodes):
        state = 0.0
        phi = get_dense_features(state)
        action = select_action_linear(W, phi, epsilon, rng)
        ep_reward, t, done = 0.0, 0, False

        while not done:
            t += 1
            step = step_env(state, action, t, mkt, rng)
            ep_reward += step["reward"]

            phi_next = get_dense_features(step["next_state"])
            action_next = select_action_linear(W, phi_next, epsilon, rng)

            q_current = phi @ W[:, action]
            q_next = 0.0 if step["done"] else phi_next @ W[:, action_next]

            td_target = step["reward"] + gamma * q_next
            W[:, action] += alpha * (td_target - q_current) * phi

            state, phi, action, done = step["next_state"], phi_next, action_next, step["done"]
        history[ep] = ep_reward
    return dict(weights=W, history=history)


print("Training Linear SARSA execution agent...")
sarsa_results = train_sarsa_linear(MKT)

# ------------------------------------------------------------------------------
# 4. Deep Q-Network execution agent (runs only if torch is installed)
# ------------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    torch.manual_seed(SEED)

    class ExecutionDQN(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(1, 32), nn.ReLU(),
                nn.Linear(32, 32), nn.ReLU(),
                nn.Linear(32, N_ACTIONS),  # Q-values for Passive / Moderate / Aggressive
            )

        def forward(self, state):
            return self.network(state)

    def as_state_tensor(x):
        return torch.tensor([[x]], dtype=torch.float32)

    def train_dqn(mkt, episodes=300, gamma=0.99, epsilon=0.1, rng=rng):
        model = ExecutionDQN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        history = np.zeros(episodes)

        for ep in range(episodes):
            state, ep_reward, t, done = 0.0, 0.0, 0, False

            while not done:
                t += 1
                state_t = as_state_tensor(state)

                if rng.random() < epsilon:
                    action = int(rng.integers(N_ACTIONS))
                else:
                    with torch.no_grad():
                        action = int(model(state_t).argmax(dim=1).item())

                step = step_env(state, action, t, mkt, rng)
                ep_reward += step["reward"]

                # Off-policy (Q-learning) TD target -- TODO (Task 2): replay buffer + target network
                q_current = model(state_t)[0, action]
                if step["done"]:
                    td_target = torch.tensor(step["reward"], dtype=torch.float32)
                else:
                    with torch.no_grad():
                        q_next_max = model(as_state_tensor(step["next_state"])).max()
                    td_target = step["reward"] + gamma * q_next_max
                loss = F.mse_loss(q_current, td_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                state, done = step["next_state"], step["done"]
            history[ep] = ep_reward
        return dict(model=model, history=history)

    print("Training Deep Q-Network execution agent...")
    dqn_results = train_dqn(MKT)
else:
    print("Package 'torch' not available -- skipping the DQN agent.\n"
          "Install with: pip install torch")
    dqn_results = None


# ------------------------------------------------------------------------------
# 5. Risk assessment: evaluate greedy policies vs classic execution benchmarks
# ------------------------------------------------------------------------------
# A policy is a function(state) -> action in {0, 1, 2}
def sarsa_policy(x):
    return int(np.argmax(get_dense_features(x) @ sarsa_results["weights"]))


policies = {
    "All Passive": lambda x: 0,
    "TWAP (Moderate)": lambda x: 1,
    "All Aggressive": lambda x: 2,
    "Front-loaded": lambda x: 2 if x < 0.4 else (1 if x < 0.85 else 0),
    "Linear SARSA": sarsa_policy,
}
if HAS_TORCH:
    def dqn_policy(x):
        with torch.no_grad():
            return int(dqn_results["model"](as_state_tensor(x)).argmax(dim=1).item())
    policies["Deep DQN"] = dqn_policy


def evaluate_policy(policy, mkt, n_episodes=500, seed=2024):
    eval_rng = np.random.default_rng(seed)  # common random numbers: every policy faces the same markets
    rows = []
    for _ in range(n_episodes):
        state, t, done = 0.0, 0, False
        reward, shortfall, forced = 0.0, 0.0, False
        while not done:
            t += 1
            step = step_env(state, policy(state), t, mkt, eval_rng)
            reward += step["reward"]
            # Realised implementation shortfall = execution cost - P&L from price moves
            shortfall += step["exec_cost"] - step["mtm_pnl"]
            forced = forced or step["forced"]
            state, done = step["next_state"], step["done"]
        rows.append((reward, shortfall, t, forced))
    return pd.DataFrame(rows, columns=["reward", "shortfall_bps", "steps", "forced"])


def risk_report(eval_df, level=0.95):
    is_bps = eval_df["shortfall_bps"].to_numpy()
    var_q = np.quantile(is_bps, level)
    return pd.Series(dict(
        MeanReward=eval_df["reward"].mean(),
        MeanIS_bps=is_bps.mean(),
        SdIS_bps=is_bps.std(ddof=1),
        VaR95_bps=var_q,
        CVaR95_bps=is_bps[is_bps >= var_q].mean(),
        MeanSteps=eval_df["steps"].mean(),
        PctForced=100 * eval_df["forced"].mean(),
    ))


print("Evaluating policies (risk assessment)...")
evaluations = {name: evaluate_policy(p, MKT) for name, p in policies.items()}
risk_table = pd.DataFrame({name: risk_report(df) for name, df in evaluations.items()}).T
with pd.option_context("display.width", 140, "display.max_columns", None):
    print(risk_table.round(2))

# ------------------------------------------------------------------------------
# 6. Visualisations
# ------------------------------------------------------------------------------
def window_filter(x, n=25):
    """Centred moving average (like R's stats::filter(sides = 2)), NaN at the edges."""
    out = np.full(len(x), np.nan)
    smoothed = np.convolve(x, np.ones(n) / n, mode="valid")
    start = (n - 1) // 2
    out[start:start + len(smoothed)] = smoothed
    return out


COL_SARSA, COL_DQN = "darkblue", "darkorange"
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# --- PLOT 1: Learning curves ---
ax = axes[0, 0]
ax.plot(window_filter(sarsa_results["history"]), color=COL_SARSA, lw=2, label="Linear SARSA (15 RBF)")
if HAS_TORCH:
    ax.plot(window_filter(dqn_results["history"]), color=COL_DQN, lw=2, label="Deep DQN (torch)")
ax.set(xlabel="Episode (trading day)", ylabel="Smoothed episode reward (bps)", title="Learning Curves")
ax.grid(color="0.85")
ax.legend(loc="lower right", frameon=False)

# --- PLOT 2: Value surfaces (expected cost-to-go) ---
test_grid = np.linspace(0, 1, 100)
ax = axes[0, 1]
linear_q_profile = [np.max(get_dense_features(x) @ sarsa_results["weights"]) for x in test_grid]
ax.plot(test_grid, linear_q_profile, color=COL_SARSA, lw=2.5)
if HAS_TORCH:
    with torch.no_grad():
        grid_t = torch.tensor(test_grid, dtype=torch.float32).unsqueeze(1)
        dqn_q_profile = dqn_results["model"](grid_t).max(dim=1).values.numpy()
    ax.plot(test_grid, dqn_q_profile, color=COL_DQN, lw=2.5)
ax.axvline(1.0, color="darkgreen", ls="--")
ax.set(xlabel="Fraction of order executed (x)", ylabel="Estimated max Q(x, a)  [bps]",
       title="Value Surface (cost-to-go)")
ax.grid(color="0.85")

# --- PLOT 3: Learned execution schedule (greedy action vs inventory) ---
ax = axes[1, 0]
ax.step(test_grid, [policies["Linear SARSA"](x) for x in test_grid], where="post",
        color=COL_SARSA, lw=2.5, label="Linear SARSA")
if HAS_TORCH:
    ax.step(test_grid, [policies["Deep DQN"](x) + 0.05 for x in test_grid], where="post",
            color=COL_DQN, lw=2.5, ls="--", label="Deep DQN")
ax.set_yticks(range(N_ACTIONS), ["Passive", "Moderate", "Aggr."])
ax.set_ylim(-0.2, 2.2)
ax.set(xlabel="Fraction of order executed (x)", title="Greedy Execution Policy")
ax.grid(color="0.85")
ax.legend(frameon=False)

# --- PLOT 4: Implementation-shortfall distribution (risk) ---
ax = axes[1, 1]
names = list(evaluations)
ax.boxplot([evaluations[n]["shortfall_bps"] for n in names], orientation="horizontal",
           tick_labels=names, showfliers=False, patch_artist=True,
           boxprops=dict(facecolor="0.9"), medianprops=dict(color="black", lw=2))
ax.scatter(risk_table.loc[names, "CVaR95_bps"], range(1, len(names) + 1),
           marker="x", color="red", s=60, lw=2, zorder=3, label="CVaR95")
ax.set(xlabel="Realised implementation shortfall (bps)", title="Execution Cost Risk")
ax.tick_params(axis="y", labelsize=8)
ax.legend(frameon=False)

fig.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 7. TODO (Task 4): Portfolio-management extension
# ------------------------------------------------------------------------------
# Implement a rebalancing environment with the SAME interface as step_env() so
# the SARSA / DQN agents can be reused unchanged. See ASSIGNMENT.md, Task 4.
#
# def step_env_portfolio(state, action, t, mkt, rng):
#     # state  : current equity weight w in [0, 1] (rest held in cash/bonds)
#     # action : 0 = de-risk (w - 0.10), 1 = hold, 2 = re-risk (w + 0.10)
#     # reward : w * r_equity + (1 - w) * r_cash - cost * |dw| - lam * w**2 * sigma**2
#     # done   : t >= horizon
#     ...
