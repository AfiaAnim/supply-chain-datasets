# ==============================================================================
# Reinforcement Learning for Finance & Trading
# Optimal Trade Execution: Linear SARSA (RBF features) vs Deep Q-Network (torch)
# ------------------------------------------------------------------------------
# Starter code for the student assignment in ASSIGNMENT.md.
#
# Scenario: a trading desk must liquidate a large parent order (e.g. 500,000
# shares) before the market closes. Each decision interval the execution agent
# chooses how aggressively to trade the next child order:
#
#   1 = Passive    : post a limit order  -> small size, earns the spread, may not fill
#   2 = Moderate   : TWAP-style slice    -> medium size, pays spread + small impact
#   3 = Aggressive : market order sweep  -> large size, pays spread + large impact
#
# State  x in [0, 1] = fraction of the parent order already executed.
# Reward (in basis points of the parent order's arrival value):
#   - execution cost of the child order   (spread + temporary market impact)
#   + mark-to-market P&L on the unsold inventory (random price moves)
#   - risk-aversion penalty on unsold inventory (volatility exposure)
# If the order is not complete by the close (max_steps), the remainder is
# force-liquidated with a single market order at a steep impact cost.
#
# This is a stylised, single-asset version of the Almgren-Chriss optimal
# execution problem: trading fast costs impact, trading slow costs risk.
# ==============================================================================
set.seed(42)

# ------------------------------------------------------------------------------
# 0. Market microstructure parameters (students will vary these in Task 3)
# ------------------------------------------------------------------------------
mkt <- list(
  half_spread_bps = 5,     # cost of crossing half the bid-ask spread
  impact_bps      = 200,   # temporary impact coefficient: cost = impact * q per unit
  sigma_bps       = 3,     # per-interval price volatility (bps)
  risk_aversion   = 2,     # penalty per interval for holding 100% of the order
  max_steps       = 60     # decision intervals until the close (e.g. 60 x 6.5 min)
)

exec_actions <- data.frame(
  name       = c("Passive (limit)", "Moderate (TWAP)", "Aggressive (market)"),
  size       = c(0.03, 0.06, 0.15),  # fraction of parent order per child order
  fill_prob  = c(0.50, 1.00, 1.00),  # limit orders only fill sometimes
  spread_dir = c(-1, 1, 1),          # -1 = earn the spread, +1 = pay it
  has_impact = c(0, 1, 1),
  stringsAsFactors = FALSE
)

# Execution cost (bps of parent order) of selling a child order of size q
slice_cost <- function(q, action, mkt) {
  q * (exec_actions$spread_dir[action] * mkt$half_spread_bps +
         exec_actions$has_impact[action] * mkt$impact_bps * q)
}

# ------------------------------------------------------------------------------
# 1. Trade-execution environment (replaces the original "wind" track)
# ------------------------------------------------------------------------------
step_env <- function(state, action, t, mkt) {
  remaining <- 1 - state

  # Child order size (cannot sell more than what is left); limit orders may miss
  filled <- runif(1) < exec_actions$fill_prob[action]
  q <- if (filled) min(exec_actions$size[action], remaining) else 0

  exec_cost  <- slice_cost(q, action, mkt)
  next_state <- min(1, state + q)
  inventory  <- 1 - next_state

  # Market risk on unsold inventory during the interval
  mtm_pnl  <- inventory * rnorm(1, 0, mkt$sigma_bps)
  risk_pen <- mkt$risk_aversion * inventory

  done <- next_state >= 1 - 1e-9
  forced_cost <- 0
  if (!done && t >= mkt$max_steps) {
    # Market close: dump the remainder in one block with a market order
    forced_cost <- slice_cost(inventory, 3, mkt)
    next_state  <- 1
    done        <- TRUE
  }

  reward <- -exec_cost - forced_cost + mtm_pnl - risk_pen
  list(next_state = next_state, reward = reward, done = done,
       exec_cost = exec_cost + forced_cost, mtm_pnl = mtm_pnl,
       forced = forced_cost > 0)
}

# ------------------------------------------------------------------------------
# 2. High-density RBF feature map (unchanged from the original grid world)
# ------------------------------------------------------------------------------
rbf_centers <- seq(0, 1, length.out = 15)
rbf_sigma   <- 0.08

get_dense_features <- function(state) {
  phi <- exp(-(state - rbf_centers)^2 / (2 * rbf_sigma^2))
  phi / sum(phi)  # normalised so features sum to 1
}

select_action_linear <- function(W, phi, epsilon) {
  if (runif(1) < epsilon) return(sample(1:3, 1))
  which.max(as.numeric(t(W) %*% phi))
}

# ------------------------------------------------------------------------------
# 3. On-policy linear SARSA execution agent
# ------------------------------------------------------------------------------
train_sarsa_linear <- function(episodes = 600, alpha = 0.05, gamma = 0.99,
                               epsilon = 0.1, mkt) {
  W <- matrix(0, nrow = length(rbf_centers), ncol = 3)
  history <- numeric(episodes)

  for (ep in 1:episodes) {
    state  <- 0
    phi    <- get_dense_features(state)
    action <- select_action_linear(W, phi, epsilon)
    ep_reward <- 0
    t    <- 0
    done <- FALSE

    while (!done) {
      t <- t + 1
      step <- step_env(state, action, t, mkt)
      ep_reward <- ep_reward + step$reward

      phi_next    <- get_dense_features(step$next_state)
      action_next <- select_action_linear(W, phi_next, epsilon)

      q_current <- sum(W[, action] * phi)
      q_next    <- if (step$done) 0 else sum(W[, action_next] * phi_next)

      td_target   <- step$reward + gamma * q_next
      W[, action] <- W[, action] + alpha * (td_target - q_current) * phi

      state  <- step$next_state
      phi    <- phi_next
      action <- action_next
      done   <- step$done
    }
    history[ep] <- ep_reward
  }
  list(Weights = W, history = history)
}

cat("Training Linear SARSA execution agent...\n")
sarsa_results <- train_sarsa_linear(episodes = 600, mkt = mkt)

# ------------------------------------------------------------------------------
# 4. Deep Q-Network execution agent (runs only if torch is installed)
# ------------------------------------------------------------------------------
has_torch <- requireNamespace("torch", quietly = TRUE) &&
  torch::torch_is_installed()

if (has_torch) {
  library(torch)
  torch_manual_seed(42)

  ExecutionDQN <- nn_module(
    "ExecutionDQN",
    initialize = function() {
      self$network <- nn_sequential(
        nn_linear(1, 32), nn_relu(),
        nn_linear(32, 32), nn_relu(),
        nn_linear(32, 3)   # Q-values for Passive / Moderate / Aggressive
      )
    },
    forward = function(state) self$network(state)
  )

  as_state_tensor <- function(x) torch_tensor(matrix(x, ncol = 1), dtype = torch_float())

  train_dqn <- function(episodes = 300, gamma = 0.99, epsilon = 0.1, mkt) {
    model     <- ExecutionDQN()
    optimizer <- optim_adam(model$parameters, lr = 0.005)
    history   <- numeric(episodes)

    for (ep in 1:episodes) {
      state <- 0
      ep_reward <- 0
      t    <- 0
      done <- FALSE

      while (!done) {
        t <- t + 1
        state_t <- as_state_tensor(state)

        if (runif(1) < epsilon) {
          action <- sample(1:3, 1)
        } else {
          with_no_grad({ q_vals <- model(state_t) })
          action <- as.numeric(torch_argmax(q_vals, dim = 2))
        }

        step <- step_env(state, action, t, mkt)
        ep_reward <- ep_reward + step$reward

        # Off-policy (Q-learning) TD target -- TODO (Task 2): replay buffer + target network
        q_current <- model(state_t)[1, action]
        if (step$done) {
          td_target <- torch_tensor(step$reward, dtype = torch_float())$squeeze()  # scalar, like q_current
        } else {
          with_no_grad({ q_next_max <- torch_max(model(as_state_tensor(step$next_state))) })
          td_target <- step$reward + gamma * q_next_max
        }
        loss <- nnf_mse_loss(q_current, td_target)

        optimizer$zero_grad()
        loss$backward()
        optimizer$step()

        state <- step$next_state
        done  <- step$done
      }
      history[ep] <- ep_reward
    }
    list(model = model, history = history)
  }

  cat("Training Deep Q-Network execution agent...\n")
  dqn_results <- train_dqn(episodes = 300, mkt = mkt)
} else {
  cat("Package 'torch' not available -- skipping the DQN agent.\n",
      "Install with: install.packages('torch'); torch::install_torch()\n")
  dqn_results <- NULL
}

# ------------------------------------------------------------------------------
# 5. Risk assessment: evaluate greedy policies vs classic execution benchmarks
# ------------------------------------------------------------------------------
# A policy is a function(state) -> action in {1, 2, 3}
policies <- list(
  "All Passive"    = function(x) 1,
  "TWAP (Moderate)" = function(x) 2,
  "All Aggressive" = function(x) 3,
  "Front-loaded"   = function(x) if (x < 0.4) 3 else if (x < 0.85) 2 else 1,
  "Linear SARSA"   = function(x) which.max(as.numeric(t(sarsa_results$Weights) %*%
                                                       get_dense_features(x)))
)
if (has_torch) {
  policies[["Deep DQN"]] <- function(x) {
    with_no_grad(as.numeric(torch_argmax(dqn_results$model(as_state_tensor(x)), dim = 2)))
  }
}

evaluate_policy <- function(policy, n_episodes = 500, mkt, seed = 2024) {
  set.seed(seed)  # common random numbers: every policy faces the same markets
  out <- data.frame(reward = numeric(n_episodes), shortfall_bps = numeric(n_episodes),
                    steps = integer(n_episodes), forced = logical(n_episodes))
  for (ep in 1:n_episodes) {
    state <- 0; t <- 0; done <- FALSE
    reward <- 0; shortfall <- 0; forced <- FALSE
    while (!done) {
      t <- t + 1
      step <- step_env(state, policy(state), t, mkt)
      reward    <- reward + step$reward
      # Realised implementation shortfall = execution cost - P&L from price moves
      shortfall <- shortfall + step$exec_cost - step$mtm_pnl
      forced    <- forced || step$forced
      state <- step$next_state
      done  <- step$done
    }
    out[ep, ] <- list(reward, shortfall, t, forced)
  }
  out
}

risk_report <- function(eval_df, level = 0.95) {
  is_bps <- eval_df$shortfall_bps
  var_q  <- unname(quantile(is_bps, level))
  c(MeanReward   = mean(eval_df$reward),
    MeanIS_bps   = mean(is_bps),
    SdIS_bps     = sd(is_bps),
    VaR95_bps    = var_q,
    CVaR95_bps   = mean(is_bps[is_bps >= var_q]),
    MeanSteps    = mean(eval_df$steps),
    PctForced    = 100 * mean(eval_df$forced))
}

cat("Evaluating policies (risk assessment)...\n")
evaluations <- lapply(policies, evaluate_policy, mkt = mkt)
risk_table  <- t(sapply(evaluations, risk_report))
print(round(risk_table, 2))

# ------------------------------------------------------------------------------
# 6. Visualisations
# ------------------------------------------------------------------------------
par(mfrow = c(2, 2), mar = c(4.5, 4.5, 3, 1.5))
window_filter <- function(x, n = 25) stats::filter(x, rep(1 / n, n), sides = 2)
col_sarsa <- "darkblue"; col_dqn <- "darkorange"

# --- PLOT 1: Learning curves ---
curves <- list(window_filter(sarsa_results$history))
if (has_torch) curves[[2]] <- window_filter(dqn_results$history)
plot(curves[[1]], type = "l", col = col_sarsa, lwd = 2,
     xlab = "Episode (trading day)", ylab = "Smoothed episode reward (bps)",
     main = "Learning Curves", ylim = range(unlist(curves), na.rm = TRUE))
if (has_torch) lines(curves[[2]], col = col_dqn, lwd = 2)
grid(col = "gray80")
legend("bottomright", legend = c("Linear SARSA (15 RBF)", "Deep DQN (torch)")[seq_along(curves)],
       col = c(col_sarsa, col_dqn)[seq_along(curves)], lty = 1, lwd = 2, bty = "n")

# --- PLOT 2: Value surfaces (expected cost-to-go) ---
test_grid <- seq(0, 1, length.out = 100)
linear_q_profile <- sapply(test_grid, function(x)
  max(as.numeric(t(sarsa_results$Weights) %*% get_dense_features(x))))
profiles <- list(linear_q_profile)
if (has_torch) {
  dqn_q_profile <- sapply(test_grid, function(x)
    with_no_grad(as.numeric(torch_max(dqn_results$model(as_state_tensor(x))))))
  profiles[[2]] <- dqn_q_profile
}
plot(test_grid, linear_q_profile, type = "l", col = col_sarsa, lwd = 2.5,
     xlab = "Fraction of order executed (x)", ylab = "Estimated max Q(x, a)  [bps]",
     main = "Value Surface (cost-to-go)", ylim = range(unlist(profiles)))
if (has_torch) lines(test_grid, dqn_q_profile, col = col_dqn, lwd = 2.5)
grid(col = "gray80")
abline(v = 1.0, col = "darkgreen", lty = 2)

# --- PLOT 3: Learned execution schedule (greedy action vs inventory) ---
plot(test_grid, sapply(test_grid, policies[["Linear SARSA"]]), type = "s",
     col = col_sarsa, lwd = 2.5, yaxt = "n", ylim = c(0.8, 3.2),
     xlab = "Fraction of order executed (x)", ylab = "", main = "Greedy Execution Policy")
axis(2, at = 1:3, labels = c("Passive", "Moderate", "Aggr."), las = 1, cex.axis = 0.8)
if (has_torch) lines(test_grid, sapply(test_grid, policies[["Deep DQN"]]) + 0.05,
                     type = "s", col = col_dqn, lwd = 2.5, lty = 2)
grid(col = "gray80")

# --- PLOT 4: Implementation-shortfall distribution (risk) ---
boxplot(lapply(evaluations, `[[`, "shortfall_bps"), horizontal = TRUE, las = 1,
        cex.axis = 0.7, col = "gray90", outline = FALSE,
        xlab = "Realised implementation shortfall (bps)", main = "Execution Cost Risk")
points(risk_table[, "CVaR95_bps"], seq_len(nrow(risk_table)), pch = 4, col = "red", lwd = 2)

par(mfrow = c(1, 1))

# ------------------------------------------------------------------------------
# 7. TODO (Task 4): Portfolio-management extension
# ------------------------------------------------------------------------------
# Implement a rebalancing environment with the SAME interface as step_env() so
# the SARSA / DQN agents can be reused unchanged. See ASSIGNMENT.md, Task 4.
#
# step_env_portfolio <- function(state, action, t, mkt) {
#   # state  : current equity weight w in [0, 1] (rest held in cash/bonds)
#   # action : 1 = de-risk (w - 0.10), 2 = hold, 3 = re-risk (w + 0.10)
#   # reward : w * r_equity + (1 - w) * r_cash - cost * |dw| - lambda * w^2 * sigma^2
#   # done   : t >= horizon
# }
