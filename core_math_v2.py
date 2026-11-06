# ==============================================================================
# MODULE: core_math_v2.py
# PERPETUAL RUIN ENGINE - ADVANCED STOCHASTIC RISK & MARGIN ANALYTICS
# ==============================================================================
# Description: Production-grade quantitative finance library for perpetual swap
#              liquidation modeling, continuous-time barrier ruin probabilities,
#              discrete sampling corrections, Merton jump-diffusions, Heston
#              stochastic volatility, regime-switching stochastic funding,
#              inverse (coin-margined) dynamic collateral decay, and risk-constrained
#              Kelly capital allocation.
# ==============================================================================

import math
import numpy as np


# ==============================================================================
# SECTION 1: STANDARD NORMAL PROBABILITY DISTRIBUTION FUNCTIONS
# ==============================================================================

def norm_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function Phi(x).
    
    Mathematical Principles:
    - [Equation 1]: Integral representation of Standard Normal CDF:
      Phi(x) = (1 / sqrt(2 * pi)) * integral_{-inf}^{x} exp(-u^2 / 2) du
    - [Equation 2]: Gauss Error Function Definition:
      erf(z) = (2 / sqrt(pi)) * integral_{0}^{z} exp(-t^2) dt
    - [Equation 3]: Analytical Formulation via Gauss Error Function:
      Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ==============================================================================
# SECTION 2: TIERED MAINTENANCE MARGIN RATE (MMR) RESOLUTION
# ==============================================================================

def resolve_tiered_mmr(
    notional_value: float,
    custom_tiers: list = None
) -> float:
    """
    Computes tiered maintenance margin rate based on total position notional value.
    
    Mathematical Principles:
    - [Equation 5]: Piecewise Constant Maintenance Margin Schedule MMR(V0):
      MMR(V0) = 0.005 (0.50%) if |V0| <= 50,000
              = 0.010 (1.00%) if 50,000 < |V0| <= 250,000
              = 0.020 (2.00%) if 250,000 < |V0| <= 1,000,000
              = 0.050 (5.00%) if |V0| > 1,000,000
    
    Parameters:
    - notional_value: float, Total position notional size (S0 * Quantity) in USD.
    - custom_tiers: list of tuples (threshold, rate), Optional custom bracket tiering.
    """
    tiers = custom_tiers or [
        (50_000.0, 0.005),      # Retail tier up to 50k USD (0.50% MMR)
        (250_000.0, 0.010),     # Mid-tier up to 250k USD (1.00% MMR)
        (1_000_000.0, 0.020),   # High notional up to 1M USD (2.00% MMR)
        (float("inf"), 0.050)   # Institutional whale tier > 1M USD (5.00% MMR)
    ]

    abs_notional = abs(float(notional_value))
    for threshold, rate in tiers:
        if abs_notional <= threshold:
            return float(rate)

    return float(tiers[-1][1])


# ==============================================================================
# SECTION 3: EXCHANGE MECHANICAL LIQUIDATION PRICE
# ==============================================================================

def calculate_liquidation_price(
    entry_price: float,
    leverage: float,
    position_type: str = "long",
    maintenance_margin_rate: float = 0.005,
    margin_mode: str = "linear",
    liquidation_fee_rate: float = 0.001,
    slippage_rate: float = 0.0,
    collateral: float = None
) -> float:
    """
    Calculates exchange mechanical liquidation price across linear and inverse margin modes.
    
    Mathematical Principles:
    - [Equation 4]: Position Notional Value: V0 = Collateral * Leverage
    - [Equation 6]: Effective Maintenance Margin: MMR_adj = MMR(V0) + f_liq
    - [Equation 7]: Initial Margin Rate: IMR0 = 1 / Leverage
    - [Equation 10]: Linear Long Liquidation: P_liq = P0 * (1 - (1 / L) + MMR_adj)
    - [Equation 11]: Linear Short Liquidation: P_liq = P0 * (1 + (1 / L) - MMR_adj)
    - [Equation 13]: Inverse Long Liquidation: P_liq = P0 / (1 + (1 / L) - MMR_adj)
    - [Equation 14]: Inverse Short Liquidation: P_liq = P0 / (1 - (1 / L) + MMR_adj)
    - [Equation 15-16]: Slippage Adjustment:
      P_liq,eff = P_liq * (1 + f_slip) for long
      P_liq,eff = P_liq * max(1e-6, 1 - f_slip) for short
    """
    if leverage <= 1.0:
        raise ValueError("Leverage must be strictly greater than 1.0.")
    if entry_price <= 0.0:
        raise ValueError("Entry price must be strictly positive.")

    mode = margin_mode.lower()
    pos = position_type.lower()
    if pos not in ("long", "short"):
        raise ValueError("position_type must be either 'long' or 'short'.")
    if mode not in ("linear", "inverse"):
        raise ValueError("margin_mode must be either 'linear' or 'inverse'.")

    # Evaluate tiered MMR if collateral is supplied
    mmr = float(maintenance_margin_rate)
    if collateral is not None and collateral > 0.0:
        notional_value = float(collateral) * float(leverage)
        mmr = resolve_tiered_mmr(notional_value)

    initial_margin_rate = 1.0 / leverage
    f_liq = float(liquidation_fee_rate)
    mmr_adj = mmr + f_liq

    if mode == "linear":
        if pos == "long":
            liq_price = entry_price * (1.0 - initial_margin_rate + mmr_adj)
        else:
            liq_price = entry_price * (1.0 + initial_margin_rate - mmr_adj)
    else:  # mode == "inverse"
        if pos == "long":
            denom = 1.0 + initial_margin_rate - mmr_adj
            liq_price = entry_price / max(1e-6, denom)
        else:
            denom = 1.0 - initial_margin_rate + mmr_adj
            liq_price = entry_price / max(1e-6, denom)

    # Adverse execution slippage adjustment
    if slippage_rate > 0.0:
        if pos == "long":
            liq_price *= (1.0 + slippage_rate)
        else:
            liq_price *= max(1e-6, (1.0 - slippage_rate))

    return max(0.0, float(liq_price))


# ==============================================================================
# SECTION 4: BROADIE-GLASSERMAN-KOU DISCRETE SAMPLING BARRIER CORRECTION
# ==============================================================================

def calculate_bgk_barrier_shift(
    barrier: float,
    sigma: float,
    dt: float,
    position_type: str = "long"
) -> float:
    """
    Shifts continuous absorbing barrier away from S0 to account for discrete monitoring.
    
    Mathematical Principles:
    - [Equation 18]: Riemann Zeta Constant Beta:
      beta = -zeta(1/2) / sqrt(2 * pi) approx 0.5825971579
    - [Equation 20]: Shifted Barrier for Long Positions (shifted downward, away from S0):
      B_adj^long = B * exp(-beta * sigma * sqrt(dt))
    - [Equation 21]: Shifted Barrier for Short Positions (shifted upward, away from S0):
      B_adj^short = B * exp(beta * sigma * sqrt(dt))
    """
    if barrier <= 0.0 or sigma <= 0.0 or dt <= 0.0:
        return barrier

    BGK_BETA = 0.5825971579317323
    vol_step = sigma * math.sqrt(dt)

    if position_type.lower() == "long":
        shifted_barrier = barrier * math.exp(-BGK_BETA * vol_step)
    else:
        shifted_barrier = barrier * math.exp(BGK_BETA * vol_step)

    return float(shifted_barrier)


# ==============================================================================
# SECTION 5: ANALYTICAL FIRST-PASSAGE TIME RUIN PROBABILITY
# ==============================================================================

def analytical_ruin_probability(
    S0: float,
    B: float,
    sigma: float,
    T_days: float,
    position_type: str = "long",
    mu: float = 0.0,
    steps_per_day: int = None
) -> float:
    """
    Computes closed-form first-passage time ruin probability via Bachelier-Harrison reflection principle.
    
    Mathematical Principles:
    - [Equation 17]: Log Drift: m = mu - 0.5 * sigma^2
    - [Equation 22-25]: Bachelier-Harrison Reflection Ruin Probability:
      P(tau <= T) = Phi(d1) + exp(scale) * Phi(d2)
    """
    if B <= 0.0 or S0 <= 0.0 or sigma <= 0.0 or T_days <= 0.0:
        return 0.0

    pos = position_type.lower()
    if pos == "long" and S0 <= B:
        return 1.0
    if pos == "short" and S0 >= B:
        return 1.0

    effective_B = B
    if steps_per_day is not None and steps_per_day > 0:
        total_steps = max(1, int(T_days * steps_per_day))
        dt = (T_days / 365.0) / total_steps
        effective_B = calculate_bgk_barrier_shift(B, sigma, dt, position_type=pos)
        if pos == "long" and S0 <= effective_B:
            return 1.0
        if pos == "short" and S0 >= effective_B:
            return 1.0

    T = T_days / 365.0
    sqrt_T = math.sqrt(T)
    m = mu - 0.5 * (sigma ** 2)

    if pos == "long":
        h = math.log(effective_B / S0)
        d1 = (h - m * T) / (sigma * sqrt_T)
        d2 = (h + m * T) / (sigma * sqrt_T)
        scale = 2.0 * m * h / (sigma ** 2)
    else:
        h = math.log(S0 / effective_B)
        d1 = (h + m * T) / (sigma * sqrt_T)
        d2 = (h - m * T) / (sigma * sqrt_T)
        scale = -2.0 * m * h / (sigma ** 2)

    term1 = norm_cdf(d1)
    term2 = 0.0 if scale > 700.0 else math.exp(scale) * norm_cdf(d2)

    prob = term1 + term2
    return max(0.0, min(1.0, float(prob)))


# ==============================================================================
# SECTION 6: BROWNIAN BRIDGE CONTINUOUS CROSSING PROBABILITY
# ==============================================================================

def brownian_bridge_crossing_prob(
    S_start: np.ndarray,
    S_end: np.ndarray,
    barrier: np.ndarray,
    sigma: np.ndarray,
    dt: float,
    position_type: str = "long"
) -> np.ndarray:
    """
    Computes exact continuous barrier breach probability within discrete step (t, t + dt).
    
    Mathematical Principles:
    - [Equation 26]: Lower Barrier (Long): P_cross = exp(-2 * ln(Sk/B) * ln(S_{k+1}/B) / (sigma_k^2 * dt))
    - [Equation 27]: Upper Barrier (Short): P_cross = exp(-2 * ln(B/Sk) * ln(B/S_{k+1}) / (sigma_k^2 * dt))
    """
    denom = (sigma ** 2) * dt
    pos = position_type.lower()
    prob = np.zeros_like(S_start, dtype=float)

    if pos == "long":
        mask = (S_start > barrier) & (S_end > barrier) & (denom > 1e-12)
        if np.any(mask):
            log_start = np.log(S_start[mask] / barrier[mask])
            log_end = np.log(S_end[mask] / barrier[mask])
            cur_denom = denom[mask] if isinstance(denom, np.ndarray) else denom
            exponent = -2.0 * log_start * log_end / cur_denom
            prob[mask] = np.exp(np.clip(exponent, -700.0, 0.0))
    else:
        mask = (S_start < barrier) & (S_end < barrier) & (denom > 1e-12)
        if np.any(mask):
            log_start = np.log(barrier[mask] / S_start[mask])
            log_end = np.log(barrier[mask] / S_end[mask])
            cur_denom = denom[mask] if isinstance(denom, np.ndarray) else denom
            exponent = -2.0 * log_start * log_end / cur_denom
            prob[mask] = np.exp(np.clip(exponent, -700.0, 0.0))

    return prob


# ==============================================================================
# SECTION 7: STOCHASTIC FUNDING, HESTON VOLATILITY & JUMP-DIFFUSION SIMULATION
# ==============================================================================

def simulate_merton_jump_diffusion_paths(
    S0: float,
    collateral: float,
    leverage: float,
    sigma: float,
    T_days: float,
    position_type: str = "long",
    mu: float = 0.0,
    annual_funding_rate: float = 0.0,
    maintenance_margin_rate: float = 0.005,
    margin_mode: str = "linear",
    liquidation_fee_rate: float = 0.001,
    slippage_rate: float = 0.001,
    num_paths: int = 5000,
    steps_per_day: int = 24,
    enable_jumps: bool = True,
    jump_intensity_lambda: float = 3.0,
    jump_mean_mu: float = -0.06,
    jump_std_sigma: float = 0.12,
    enable_stochastic_funding: bool = True,
    funding_ou_theta: float = 5.0,
    funding_ou_sigma: float = 0.04,
    funding_price_corr: float = 0.65,
    enable_regime_switching: bool = True,
    funding_bear_rate: float = -0.05,
    enable_stochastic_vol: bool = True,
    heston_kappa: float = 3.0,
    heston_vol_of_vol: float = 0.35,
    heston_price_vol_corr: float = -0.60,
    enable_brownian_bridge: bool = True
) -> dict:
    """
    Monte Carlo engine simulating price paths with Heston stochastic volatility,
    regime-switching OU funding rates, Merton jumps, and inverse coin-margined collateral decay.
    
    Mathematical Principles:
    - [Equation 28-30]: Heston CIR process with Full-Truncation Euler scheme
    - [Equation 31-33]: Merton jump-diffusion with discrete log-Euler stepping
    - [Equation 34-35]: Regime-switching OU stochastic funding rates
    - [Equation 36-41]: Dynamic liquidation barrier evolution across linear and inverse margin modes
    """
    T = T_days / 365.0
    total_steps = max(1, int(T_days * steps_per_day))
    dt = T / total_steps
    dt_days = T_days / total_steps
    sqrt_dt = math.sqrt(dt)

    position_size = collateral * leverage
    pos = position_type.lower()
    mode = margin_mode.lower()

    # Generate correlated Brownian shocks
    z_price = np.random.normal(0.0, 1.0, size=(num_paths, total_steps))
    z_vol_ortho = np.random.normal(0.0, 1.0, size=(num_paths, total_steps))
    z_fund_ortho = np.random.normal(0.0, 1.0, size=(num_paths, total_steps))

    rho_sv = np.clip(heston_price_vol_corr, -0.99, 0.99)
    z_vol = rho_sv * z_price + math.sqrt(1.0 - (rho_sv ** 2)) * z_vol_ortho

    rho_sf = np.clip(funding_price_corr, -0.99, 0.99)
    z_funding = rho_sf * z_price + math.sqrt(1.0 - (rho_sf ** 2)) * z_fund_ortho

    # 1. Heston CIR Stochastic Volatility
    variance_paths = np.zeros((num_paths, total_steps + 1))
    init_var = sigma ** 2
    variance_paths[:, 0] = init_var

    if enable_stochastic_vol:
        for step in range(total_steps):
            v_curr = np.maximum(0.0, variance_paths[:, step])
            sqrt_v = np.sqrt(v_curr)
            dv = (
                heston_kappa * (init_var - v_curr) * dt
                + heston_vol_of_vol * sqrt_v * sqrt_dt * z_vol[:, step]
            )
            variance_paths[:, step + 1] = np.maximum(1e-6, variance_paths[:, step] + dv)
    else:
        variance_paths[:, :] = init_var

    local_sigmas = np.sqrt(variance_paths[:, :-1])

    # 2. Merton Jump-Diffusion
    if enable_jumps and jump_intensity_lambda > 0.0:
        kappa = math.exp(jump_mean_mu + 0.5 * (jump_std_sigma ** 2)) - 1.0
        effective_lambda = jump_intensity_lambda
    else:
        kappa = 0.0
        effective_lambda = 0.0

    drift_steps = (mu - 0.5 * (local_sigmas ** 2) - effective_lambda * kappa) * dt
    continuous_increments = drift_steps + local_sigmas * sqrt_dt * z_price

    if enable_jumps and effective_lambda > 0.0:
        jump_counts = np.random.poisson(effective_lambda * dt, size=(num_paths, total_steps))
        jump_shocks = (
            jump_counts * jump_mean_mu
            + np.sqrt(jump_counts) * jump_std_sigma * np.random.normal(0.0, 1.0, size=(num_paths, total_steps))
        )
        total_log_increments = continuous_increments + jump_shocks
    else:
        total_log_increments = continuous_increments

    cum_log = np.cumsum(total_log_increments, axis=1)
    paths = S0 * np.exp(cum_log)
    full_paths = np.hstack([np.full((num_paths, 1), S0), paths])
    time_grid = np.linspace(0.0, T_days, total_steps + 1)

    # 3. Regime-Switching Stochastic Funding
    funding_rates = np.zeros((num_paths, total_steps + 1))
    funding_rates[:, 0] = annual_funding_rate

    if enable_stochastic_funding:
        for step in range(total_steps):
            if enable_regime_switching:
                price_ratio = full_paths[:, step] / S0
                is_bear = price_ratio < 0.90
                target_r = np.where(is_bear, funding_bear_rate, annual_funding_rate)
            else:
                target_r = annual_funding_rate

            dr = (
                funding_ou_theta * (target_r - funding_rates[:, step]) * dt
                + funding_ou_sigma * sqrt_dt * z_funding[:, step]
            )
            funding_rates[:, step + 1] = funding_rates[:, step] + dr
    else:
        funding_rates[:, :] = annual_funding_rate

    # 4. Collateral Bleed & Dynamic Barrier Evolution
    collateral_matrix_usd = np.zeros((num_paths, total_steps + 1))
    dynamic_barriers = np.zeros((num_paths, total_steps + 1))

    collateral_matrix_usd[:, 0] = collateral
    initial_coin_collateral = collateral / S0 if mode == "inverse" else 0.0
    coin_collateral_matrix = np.full((num_paths, total_steps + 1), initial_coin_collateral)

    # Resolve MMR bracket based on initial position notional
    base_mmr = (
        resolve_tiered_mmr(position_size)
        if (collateral is not None and collateral > 0.0)
        else float(maintenance_margin_rate)
    )
    mmr_adj = base_mmr + float(liquidation_fee_rate)

    dynamic_barriers[:, 0] = calculate_liquidation_price(
        entry_price=S0,
        leverage=leverage,
        position_type=pos,
        maintenance_margin_rate=base_mmr,
        margin_mode=mode,
        liquidation_fee_rate=liquidation_fee_rate,
        slippage_rate=slippage_rate,
        collateral=collateral
    )

    dt_years_step = dt_days / 365.0

    for step in range(1, total_steps + 1):
        s_curr = full_paths[:, step]
        current_rates = funding_rates[:, step - 1]

        if mode == "linear":
            # Linear contract USD collateral bleed
            if pos == "long":
                collateral_matrix_usd[:, step] = (
                    collateral_matrix_usd[:, step - 1] - position_size * current_rates * dt_years_step
                )
            else:
                collateral_matrix_usd[:, step] = (
                    collateral_matrix_usd[:, step - 1] + position_size * current_rates * dt_years_step
                )
            collateral_matrix_usd[:, step] = np.maximum(0.0, collateral_matrix_usd[:, step])
            current_imr = np.maximum(1e-4, collateral_matrix_usd[:, step] / position_size)

            if pos == "long":
                dynamic_barriers[:, step] = S0 * (1.0 - current_imr + mmr_adj) * (1.0 + slippage_rate)
            else:
                dynamic_barriers[:, step] = S0 * (1.0 + current_imr - mmr_adj) * max(1e-6, 1.0 - slippage_rate)

        else:  # mode == "inverse" (Coin-Margined)
            # Funding fee deducted in underlying cryptocurrency coin
            coin_funding_delta = (position_size / np.maximum(1e-4, s_curr)) * current_rates * dt_years_step
            if pos == "long":
                coin_collateral_matrix[:, step] = coin_collateral_matrix[:, step - 1] - coin_funding_delta
            else:
                coin_collateral_matrix[:, step] = coin_collateral_matrix[:, step - 1] + coin_funding_delta

            coin_collateral_matrix[:, step] = np.maximum(0.0, coin_collateral_matrix[:, step])
            collateral_matrix_usd[:, step] = np.maximum(0.0, coin_collateral_matrix[:, step] * s_curr)
            
            # Anchor IMR to S0 to prevent double-counting mark-to-market drops in the barrier
            current_imr = np.maximum(1e-4, (coin_collateral_matrix[:, step] * S0) / position_size)

            if pos == "long":
                denom = np.maximum(1e-6, 1.0 + current_imr - mmr_adj)
                dynamic_barriers[:, step] = (S0 / denom) * (1.0 + slippage_rate)
            else:
                denom = np.maximum(1e-6, 1.0 - current_imr + mmr_adj)
                dynamic_barriers[:, step] = (S0 / denom) * max(1e-6, 1.0 - slippage_rate)

        dynamic_barriers[:, step] = np.maximum(0.0, dynamic_barriers[:, step])

    # 5. Ruin Detection via Discrete Check & Continuous Brownian Bridge
    if pos == "long":
        discrete_breached = full_paths <= dynamic_barriers
    else:
        discrete_breached = full_paths >= dynamic_barriers

    ever_breached = np.any(discrete_breached, axis=1)

    if enable_brownian_bridge:
        for step in range(total_steps):
            alive_mask = ~ever_breached
            if not np.any(alive_mask):
                break

            s_start = full_paths[alive_mask, step]
            s_end = full_paths[alive_mask, step + 1]
            b_step = dynamic_barriers[alive_mask, step + 1]
            sig_step = local_sigmas[alive_mask, step]

            p_cross = brownian_bridge_crossing_prob(
                S_start=s_start,
                S_end=s_end,
                barrier=b_step,
                sigma=sig_step,
                dt=dt,
                position_type=pos
            )

            uniform_draw = np.random.uniform(0.0, 1.0, size=len(p_cross))
            bridge_breach = uniform_draw < p_cross
            ever_breached[alive_mask] = bridge_breach

    empirical_ruin_prob = float(np.mean(ever_breached))
    avg_collateral_series = np.mean(collateral_matrix_usd, axis=0)
    avg_barrier_series = np.mean(dynamic_barriers, axis=0)

    return {
        "empirical_prob": empirical_ruin_prob,
        "liquidated_count": int(np.sum(ever_breached)),
        "total_paths": num_paths,
        "sample_paths": full_paths[:50],
        "sample_paths_breached": ever_breached[:50].tolist(),
        "time_grid": time_grid,
        "dynamic_barriers": avg_barrier_series,
        "final_collateral": float(avg_collateral_series[-1]),
        "collateral_series": avg_collateral_series,
        "average_funding_rates": np.mean(funding_rates, axis=0),
        "average_volatility": float(np.mean(np.sqrt(variance_paths))),
    }


# ==============================================================================
# SECTION 8: 2D PARAMETRIC RUIN PROBABILITY HEATMAP MATRIX
# ==============================================================================

def generate_ruin_heatmap_matrix(
    entry_price: float,
    sigma_min: float = 0.20,
    sigma_max: float = 1.40,
    leverage_min: float = 2.0,
    leverage_max: float = 50.0,
    duration_days: float = 7.0,
    position_type: str = "long",
    mu: float = 0.0,
    grid_size: int = 25,
    margin_mode: str = "linear",
    liquidation_fee_rate: float = 0.001,
    slippage_rate: float = 0.0,
    steps_per_day: int = 24
) -> tuple:
    """
    Constructs a 2D matrix of ruin probabilities across leverage and volatility dimensions.
    
    Mathematical Principles:
    - [Equation 42-44]: Uniform Cartesian grid evaluation of analytical ruin surface
    """
    sigmas = np.linspace(sigma_min, sigma_max, grid_size)
    leverages = np.linspace(leverage_min, leverage_max, grid_size)
    prob_matrix = np.zeros((grid_size, grid_size))

    for i, sig in enumerate(sigmas):
        for j, lev in enumerate(leverages):
            liq = calculate_liquidation_price(
                entry_price=entry_price,
                leverage=lev,
                position_type=position_type,
                margin_mode=margin_mode,
                liquidation_fee_rate=liquidation_fee_rate,
                slippage_rate=slippage_rate
            )
            prob_matrix[i, j] = analytical_ruin_probability(
                S0=entry_price,
                B=liq,
                sigma=sig,
                T_days=duration_days,
                position_type=position_type,
                mu=mu,
                steps_per_day=steps_per_day
            )

    return leverages, sigmas, prob_matrix


# ==============================================================================
# SECTION 9: RISK-CONSTRAINED HALF-KELLY LEVERAGE OPTIMIZATION
# ==============================================================================

def calculate_kelly_constrained_leverage(
    mu: float,
    sigma: float,
    entry_price: float,
    duration_days: float,
    position_type: str = "long",
    max_ruin_prob: float = 0.01,
    max_allowed_leverage: float = 50.0,
    margin_mode: str = "linear",
    liquidation_fee_rate: float = 0.001,
    slippage_rate: float = 0.0,
    steps_per_day: int = 24,
    use_half_kelly: bool = True,
    apply_shrinkage: bool = False,
    sample_size_days: float = 180.0
) -> dict:
    """
    Computes growth-optimal Kelly leverage bounded by a maximum permissible ruin probability.
    
    Mathematical Principles:
    - [Equation 45]: Directional Drift Alignment: mu_dir = +mu (long), -mu (short)
    - [Equation 46]: Optional Merton-Bayes Shrinkage: mu_eff = w * mu_dir
    - [Equation 47]: Fractional Half-Kelly Leverage: f_half^* = max(1.0, 0.5 * mu_eff / sigma^2)
    - [Equation 48-49]: Bisection search for max safe leverage under ruin constraint
    """
    pos = position_type.lower()
    mu_dir = float(mu) if pos == "long" else -float(mu)

    if mu_dir > 0.0 and sigma > 0.0:
        if apply_shrinkage and sample_size_days > 0.0:
            sample_variance_mu = (sigma ** 2) / (sample_size_days / 365.0)
            prior_variance = 0.15 ** 2
            shrinkage_weight = prior_variance / (prior_variance + sample_variance_mu)
            effective_mu = mu_dir * shrinkage_weight
        else:
            effective_mu = mu_dir

        unconstrained_kelly = effective_mu / (sigma ** 2)
        if use_half_kelly:
            unconstrained_kelly *= 0.5
        unconstrained_kelly = max(1.0, unconstrained_kelly)
    else:
        unconstrained_kelly = 1.0

    low, high = 1.01, float(max_allowed_leverage)
    safe_leverage = 1.01

    # 40-iteration bisection search
    for _ in range(40):
        mid = (low + high) / 2.0
        test_liq = calculate_liquidation_price(
            entry_price=entry_price,
            leverage=mid,
            position_type=position_type,
            margin_mode=margin_mode,
            liquidation_fee_rate=liquidation_fee_rate,
            slippage_rate=slippage_rate
        )
        prob = analytical_ruin_probability(
            S0=entry_price,
            B=test_liq,
            sigma=sigma,
            T_days=duration_days,
            position_type=position_type,
            mu=0.0,
            steps_per_day=steps_per_day
        )

        if prob > max_ruin_prob:
            high = mid
        else:
            low = mid
            safe_leverage = mid

    recommended_leverage = min(unconstrained_kelly, safe_leverage)

    return {
        "unconstrained_kelly": float(unconstrained_kelly),
        "ruin_constrained_leverage": float(safe_leverage),
        "recommended_leverage": float(recommended_leverage),
        "target_max_ruin_prob": float(max_ruin_prob)
    }


# ==============================================================================
# SECTION 10: HISTORICAL BLACK SWAN & SHORT SQUEEZE REPLAY ENGINE
# ==============================================================================

def replay_historical_black_swan(
    scenario_key: str,
    entry_price: float,
    leverage: float,
    position_type: str = "long",
    maintenance_margin_rate: float = 0.005,
    margin_mode: str = "linear",
    liquidation_fee_rate: float = 0.001,
    slippage_rate: float = 0.0
) -> dict:
    """
    Stress-tests trade parameters against empirical market crash and short-squeeze trajectories.
    
    Mathematical Principles:
    - [Equation 50-53]: Historical multiplier trajectory replay and maximum adverse excursion
    """
    scenarios = {
        "black_thursday_2020": {
            "name": "March 2020 (Black Thursday Liquidity Collapse)",
            "hours": 36,
            "multipliers": [
                1.00, 0.98, 0.95, 0.92, 0.88, 0.85, 0.82, 0.80, 0.77, 0.74,
                0.71, 0.67, 0.61, 0.55, 0.51, 0.49, 0.52, 0.56, 0.53, 0.50,
                0.54, 0.58, 0.60, 0.62, 0.61, 0.64, 0.66, 0.65, 0.68, 0.67,
                0.69, 0.71, 0.70, 0.72, 0.71, 0.73
            ]
        },
        "china_ban_2021": {
            "name": "May 2021 (China Mining Ban Flash Crash)",
            "hours": 24,
            "multipliers": [
                1.00, 0.97, 0.95, 0.93, 0.91, 0.88, 0.85, 0.81, 0.78, 0.74,
                0.71, 0.68, 0.69, 0.73, 0.76, 0.79, 0.81, 0.80, 0.82, 0.83,
                0.81, 0.82, 0.84, 0.85
            ]
        },
        "ftx_collapse_2022": {
            "name": "November 2022 (FTX Insolvency Cascade)",
            "hours": 48,
            "multipliers": [
                1.00, 0.99, 0.97, 0.96, 0.95, 0.93, 0.91, 0.90, 0.88, 0.87,
                0.84, 0.81, 0.79, 0.78, 0.76, 0.75, 0.77, 0.76, 0.75, 0.74,
                0.73, 0.74, 0.75, 0.76, 0.75, 0.74, 0.73, 0.72, 0.74, 0.75,
                0.76, 0.75, 0.74, 0.73, 0.74, 0.75, 0.76, 0.75, 0.74, 0.75,
                0.76, 0.77, 0.76, 0.75, 0.76, 0.77, 0.78, 0.79
            ]
        },
        "short_squeeze_2021": {
            "name": "July 2021 (Amazon Rumor Short Squeeze Spike)",
            "hours": 24,
            "multipliers": [
                1.00, 1.02, 1.03, 1.05, 1.08, 1.12, 1.15, 1.19, 1.23, 1.28,
                1.32, 1.35, 1.31, 1.27, 1.25, 1.23, 1.21, 1.20, 1.18, 1.19,
                1.17, 1.16, 1.15, 1.14
            ]
        }
    }

    selected = scenarios.get(scenario_key, scenarios["black_thursday_2020"])
    pos = position_type.lower()

    liq_price = calculate_liquidation_price(
        entry_price=entry_price,
        leverage=leverage,
        position_type=pos,
        maintenance_margin_rate=maintenance_margin_rate,
        margin_mode=margin_mode,
        liquidation_fee_rate=liquidation_fee_rate,
        slippage_rate=slippage_rate
    )

    if pos == "long":
        trajectory = np.array(selected["multipliers"]) * entry_price
        breaches = np.where(trajectory <= liq_price)[0]
    else:
        if scenario_key == "short_squeeze_2021":
            trajectory = np.array(selected["multipliers"]) * entry_price
        else:
            trajectory = entry_price * (2.0 - np.array(selected["multipliers"]))
        breaches = np.where(trajectory >= liq_price)[0]

    liquidated = len(breaches) > 0
    liquidation_hour = int(breaches[0]) if liquidated else None

    if pos == "long":
        max_drawdown = float(np.min(trajectory) / entry_price - 1.0)
    else:
        max_drawdown = float(np.max(trajectory) / entry_price - 1.0)

    return {
        "name": selected["name"],
        "timeline_hours": list(range(len(trajectory))),
        "trajectory": trajectory.tolist(),
        "liq_price": float(liq_price),
        "liquidated": bool(liquidated),
        "liquidation_hour": liquidation_hour,
        "max_drawdown_pct": max_drawdown
    }


if __name__ == "__main__":
    print("Testing core_math_v2 module...")
    p_liq = calculate_liquidation_price(60000.0, 20.0, "long", collateral=30000.0, slippage_rate=0.002)
    print(f"Long Liq Price: {p_liq:.2f}")
    p_ruin = analytical_ruin_probability(60000.0, 57000.0, 0.65, 14.0, "long", steps_per_day=24)
    print(f"Analytical Ruin Prob (BGK shifted): {p_ruin:.4f}")
    sim_res = simulate_merton_jump_diffusion_paths(
        S0=60000.0, collateral=3000.0, leverage=10.0, sigma=0.60, T_days=7.0,
        position_type="long", margin_mode="inverse", num_paths=1000
    )
    print(f"Inverse Monte Carlo Ruin Prob: {sim_res['empirical_prob']:.4f}")
    kelly_res = calculate_kelly_constrained_leverage(0.15, 0.70, 60000.0, 7.0, position_type="long")
    print(f"Recommended Kelly Leverage: {kelly_res['recommended_leverage']:.2f}")
    print("All core_math_v2 tests passed successfully!")