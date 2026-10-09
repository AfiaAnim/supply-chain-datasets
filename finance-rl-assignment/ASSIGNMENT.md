# Assignment: Reinforcement Learning for Finance & Trading

**Topic:** Function approximation in RL (linear SARSA with RBF features vs Deep Q-Networks)
**Application:** Automated trade execution, portfolio management and risk assessment
**Language:** R (base R + the `torch` package)
**Estimated effort:** 10–12 hours  **Total:** 100 points
**Starter code:** [`trade_execution_rl.R`](trade_execution_rl.R)

---

## 1. Learning objectives

By the end of this assignment you should be able to:

1. Frame a trading problem as a Markov Decision Process: define its state, actions, reward and termination.
2. Explain the trade-off between **market impact** (trading too fast) and **inventory risk** (trading too slowly) behind optimal execution.
3. Train and compare a **linear SARSA** agent on Gaussian radial-basis-function (RBF) features with a **Deep Q-Network (DQN)**.
4. Assess strategies with **risk metrics**, not just average reward: implementation shortfall, standard deviation, VaR and CVaR.
5. Reuse the same RL machinery for a **portfolio rebalancing** problem.
6. Critically discuss model risk, overfitting to a simulator, and regulatory constraints on algorithmic trading.

---

## 2. Background: from a toy track to a trading desk

The starter code adapts a classic continuous "move along a track" control problem to finance. Every part of the original problem has a financial counterpart:

| Original toy problem | Finance version (this assignment) |
|---|---|
| Position on a track, `x ∈ [0, 1]` | Fraction of the parent order already executed |
| Goal at `x = 1` | Parent order fully liquidated |
| Actions: step −0.05 / +0.01 / +0.05 | Child order: **Passive** limit order / **Moderate** TWAP slice / **Aggressive** market order |
| Wind noise on movement | Limit-order fill uncertainty and random price moves |
| Reward −1 per step (time penalty) | −(execution cost) + (mark-to-market P&L) − (risk penalty on unsold shares) |
| Episode | One trading day |
| No time limit | Market close after `max_steps` intervals forces liquidation of what is left |

### 2.1 The environment (`step_env`)

A desk must sell a large block of shares (for example, 500,000) before the close. All quantities are measured in **basis points (bps) of the order's arrival value**.

**Actions:**

| # | Action | Child size `q` | Fill probability | Spread | Market impact |
|---|---|---|---|---|---|
| 1 | Passive (limit order) | 3 % | 50 % | *earn* the half-spread | none |
| 2 | Moderate (TWAP slice) | 6 % | 100 % | pay the half-spread | `η·q` per unit |
| 3 | Aggressive (market order) | 15 % | 100 % | pay the half-spread | `η·q` per unit |

**Reward at each interval:**

$$
r_t = -\underbrace{q\,(\pm s + \eta\, q)}_{\text{execution cost}} \;+\; \underbrace{(1-x_{t+1})\,\sigma\,\varepsilon_t}_{\text{mark-to-market P\&L}} \;-\; \underbrace{\lambda\,(1-x_{t+1})}_{\text{inventory-risk penalty}}, \qquad \varepsilon_t \sim N(0,1)
$$

where `s = half_spread_bps`, `η = impact_bps`, `σ = sigma_bps` and `λ = risk_aversion` (all set in the `mkt` list).

**Termination:** the episode ends when the order is fully executed. If the order is still open after `max_steps` intervals, the remainder is sold in a single market order, which has a very large impact cost.

**Implementation shortfall (IS):** the realised execution cost minus the P&L from price moves. This is the standard industry measure of execution quality. Lower is better.

This is a stylised, single-asset version of the **Almgren–Chriss (2000)** optimal execution model.

### 2.2 What the starter code already does

1. Trains a **linear SARSA** agent on 15 normalised Gaussian RBF features.
2. Trains a **DQN** with a 1→32→32→3 ReLU network (only if `torch` is installed).
3. Evaluates both greedy policies against four classic benchmarks (All Passive, TWAP, All Aggressive, Front-loaded). Every policy faces the **same random markets**, a technique called common random numbers.
4. Prints a **risk table** and draws four plots: learning curves, value surface, greedy execution schedule, and the IS distribution with CVaR markers.

### 2.3 Running it

```r
# One-time setup for the DQN part
install.packages("torch"); torch::install_torch()

source("trade_execution_rl.R")
```

The SARSA agent and the benchmarks need only base R and run in a few seconds. If `torch` is missing, the DQN is skipped with a message.

As a sanity check, with the default parameters and `set.seed(42)` your risk table should look roughly like this (exact values vary slightly by platform):

| Policy | Mean reward | Mean IS (bps) | VaR95 (bps) | CVaR95 (bps) | % forced at close |
|---|---|---|---|---|---|
| All Passive | ≈ −66 | ≈ 1 | ≈ 28 | ≈ 34 | ≈ 79 |
| TWAP (Moderate) | ≈ −33 | ≈ 18 | ≈ 28 | ≈ 30 | 0 |
| All Aggressive | ≈ −40 | ≈ 34 | ≈ 41 | ≈ 42 | 0 |
| Linear SARSA | ≈ −31 | ≈ 16 | ≈ 26 | ≈ 27 | 0 |

---

## 3. Tasks

### Task 1: Understand the trading MDP (15 pts)

1. **(4 pts)** Explain in plain financial language what each of the three reward terms represents. Why does the passive action have a fill probability below 1, and why does it *earn* the spread?
2. **(4 pts)** "All Passive" has the **lowest mean IS** but the **worst mean reward**. Use the risk table to explain this. What would a risk-neutral trader prefer? What would a risk manager prefer?
3. **(4 pts)** Describe the greedy schedule the SARSA agent learned (Plot 3). Relate its shape to the Almgren–Chriss result that a risk-averse trader should *front-load* execution.
4. **(3 pts)** The state contains only the fraction executed, not the time left until the close. Explain why this makes the environment **not fully Markov**. Give one concrete situation where the agent's choice would be wrong because of this.

### Task 2: Make the DQN trade properly (25 pts)

The DQN in the starter code learns online from one transition at a time and bootstraps from its own, constantly moving estimates. This is known to be unstable.

1. **(10 pts)** Add an **experience replay buffer**: store at least 5,000 transitions and train on random mini-batches of 32.
2. **(8 pts)** Add a **target network** that is synchronised every *N* steps. Choose *N* yourself and justify the choice.
3. **(4 pts)** Train each version (original, + replay, + replay + target) with **5 different seeds**. Plot the mean learning curve with a ±1 standard deviation band for each version.
4. **(3 pts)** SARSA is *on-policy* and the DQN uses an *off-policy* (max) target. With ε = 0.1, exploration can trigger an accidental aggressive order. Explain how each algorithm accounts for that cost, and which one you would trust on a live desk.

### Task 3: Market-regime sensitivity (20 pts)

Re-train the SARSA agent (and the DQN if you have it) under each regime below. Change only the listed parameter.

| Regime | Change |
|---|---|
| Base | defaults |
| High volatility | `sigma_bps = 9` |
| Very risk-averse desk | `risk_aversion = 6` |
| Illiquid stock | `impact_bps = 600`, `half_spread_bps = 15` |
| Short deadline | `max_steps = 20` |

1. **(8 pts)** For each regime, report the risk table and plot the learned greedy schedule.
2. **(6 pts)** Explain how and why the schedule moves (more aggressive or more passive) in each regime. Check whether this matches financial intuition.
3. **(6 pts)** **Model-risk stress test:** take the policy trained in the **Base** regime and evaluate it, *without retraining*, in the **High volatility** and **Illiquid** regimes. Compare its VaR95 and CVaR95 with policies trained in those regimes. What does this tell you about deploying an RL policy trained on a simulator?

### Task 4: Portfolio-management extension (20 pts)

Implement `step_env_portfolio(state, action, t, mkt)` (stub at the bottom of the script). It must keep the **same interface** as `step_env` so both agents can be reused unchanged.

- **State:** equity weight `w ∈ [0, 1]`; the rest is held in cash or bonds.
- **Actions:** 1 = de-risk (`w − 0.10`), 2 = hold, 3 = re-risk (`w + 0.10`). Clip to [0, 1].
- **Returns:** simulate daily equity returns from a **two-regime** model, with "calm" (μ = 0.05 %, σ = 0.8 %) and "stressed" (μ = −0.10 %, σ = 2.5 %) regimes and a 2 % daily switching probability. Use a cash return of 0.01 % per day.
- **Reward:** `w·r_eq + (1 − w)·r_cash − c·|Δw| − λ·w²·σ²`, with transaction cost `c = 10 bps`.
- **Episode:** 252 trading days.

1. **(10 pts)** Implement the environment and train the SARSA agent on it.
2. **(6 pts)** Compare the learned policy with **buy-and-hold 60/40** and **monthly rebalanced 60/40** on 500 out-of-sample years. Report annualised return, volatility, Sharpe ratio, maximum drawdown and turnover.
3. **(4 pts)** The agent can only see `w`, not the market regime. Discuss what extra state information a real portfolio manager would add, and how you would encode it as features.

### Task 5: Risk assessment deep-dive (10 pts)

1. **(5 pts)** VaR95 and CVaR95 computed from 500 episodes are themselves noisy estimates. Use a **bootstrap** with 2,000 resamples to report 95 % confidence intervals for the CVaR95 of SARSA and of TWAP. Is SARSA's improvement statistically meaningful?
2. **(5 pts)** Change the SARSA objective to be explicitly **risk-sensitive**, for example by penalising squared per-step P&L or by changing λ. Show the resulting **mean–CVaR efficient frontier** across at least 4 settings.

### Task 6: Report and professional reflection (10 pts)

Write a report of at most 6 pages (plots and tables included). It must contain the answers to Tasks 1–5 and a short reflection (about 400 words) that covers:

- the gap between this simulator and real limit-order-book dynamics, such as permanent impact, queue position and adverse selection;
- the risk of **overfitting to a backtest or simulator**;
- regulatory and ethical constraints on algorithmic execution: best-execution duties (MiFID II / Reg NMS), kill switches, and why an RL agent must never learn manipulative patterns such as spoofing or layering.

---

## 4. Deliverables

1. `trade_execution_rl.R` with your modifications. It must run top to bottom with `source()`, and any new code must be clearly commented with `# Task X`.
2. `portfolio_rl.R` (or a clearly marked section) for Task 4.
3. `report.pdf`.
4. Fix `set.seed()` everywhere so that your numbers can be reproduced.

## 5. Grading rubric

| Criterion | Excellent (100 %) | Adequate (60 %) | Weak (≤ 30 %) |
|---|---|---|---|
| Correctness of code | Runs cleanly; replay, target network and portfolio environment are correct | Minor bugs; results still interpretable | Does not run or has wrong logic |
| Financial interpretation | Every result is linked to market microstructure or portfolio theory | Some interpretation, mostly descriptive | Numbers reported without meaning |
| Risk analysis | Uses VaR/CVaR with confidence intervals and stress tests | Reports risk metrics without uncertainty | Average reward only |
| Experimental rigour | Multiple seeds, common random numbers, fair baselines | Single seed, but fair comparison | Cherry-picked runs |
| Communication | Clear, concise, well-labelled plots | Understandable but cluttered | Hard to follow |

## 6. Hints

- Keep features in [0, 1]. If you add **time-to-close** as a second state variable (a great bonus idea for Task 1.4), use a 2-D RBF grid, for example `expand.grid` over 8 × 8 centres.
- In R `torch`, `torch_argmax` returns **1-based** indices, so it maps directly to actions 1–3.
- Wrap evaluation-time network calls in `with_no_grad()` so no autograd graph is built.
- When comparing policies, always reuse the same seed in `evaluate_policy()`. Otherwise differences may just be market noise.

## 7. References

- Almgren, R. & Chriss, N. (2000). *Optimal execution of portfolio transactions.* Journal of Risk, 3(2).
- Nevmyvaka, Y., Feng, Y. & Kearns, M. (2006). *Reinforcement learning for optimized trade execution.* ICML.
- Moody, J. & Saffell, M. (2001). *Learning to trade via direct reinforcement.* IEEE Trans. Neural Networks, 12(4).
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518.
- Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Chapters 9–10.
- Rockafellar, R. T. & Uryasev, S. (2000). *Optimization of conditional value-at-risk.* Journal of Risk, 2(3).
