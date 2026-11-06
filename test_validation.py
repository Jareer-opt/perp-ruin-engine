# ==============================================================================
# MODULE: test_validation.py
# PERPETUAL RUIN ENGINE - MATHEMATICAL VALIDATION & PERFORMANCE BENCHMARK SUITE
# ==============================================================================
# Description: Zero-dependency unit test and latency benchmarking suite using
#              Python's built-in `unittest` and `time` modules. Validates closed-
#              form boundary conditions, BGK discrete corrections, Monte Carlo
#              convergence to Bachelier-Harrison reflection formulas, inverse vs.
#              linear convexity, volatility estimators, and execution speed.
# ==============================================================================

import math
import time
import unittest
import numpy as np

from core_math_v2 import (
    norm_cdf,
    resolve_tiered_mmr,
    calculate_liquidation_price,
    calculate_bgk_barrier_shift,
    analytical_ruin_probability,
    brownian_bridge_crossing_prob,
    simulate_merton_jump_diffusion_paths,
    generate_ruin_heatmap_matrix,
    calculate_kelly_constrained_leverage,
    replay_historical_black_swan,
)
from market_data_v2 import (
    calculate_parkinson_volatility,
    calculate_rogers_satchell_volatility,
    calculate_yang_zhang_volatility,
    calculate_bayesian_drift,
    fetch_deribit_dvol,
)


class TestCoreMathEngine(unittest.TestCase):
    """Unit tests verifying mathematical invariants and financial logic."""

    def test_01_norm_cdf_invariants(self):
        """Verify standard normal CDF symmetry and known quantiles."""
        self.assertAlmostEqual(norm_cdf(0.0), 0.5, places=12)
        self.assertAlmostEqual(norm_cdf(1.959963984540054), 0.975, places=6)
        self.assertAlmostEqual(norm_cdf(-1.959963984540054), 0.025, places=6)
        for x in [0.5, 1.2, 2.5, 4.0]:
            self.assertAlmostEqual(norm_cdf(x) + norm_cdf(-x), 1.0, places=12)

    def test_02_tiered_mmr_schedule(self):
        """Verify institutional maintenance margin tier brackets."""
        self.assertEqual(resolve_tiered_mmr(10_000.0), 0.005)
        self.assertEqual(resolve_tiered_mmr(50_000.0), 0.005)
        self.assertEqual(resolve_tiered_mmr(100_000.0), 0.010)
        self.assertEqual(resolve_tiered_mmr(500_000.0), 0.020)
        self.assertEqual(resolve_tiered_mmr(2_500_000.0), 0.050)

    def test_03_linear_vs_inverse_liquidation_convexity(self):
        """
        Verify exact linear and inverse mechanical liquidation formulas and
        confirm that inverse long liquidation triggers earlier (higher price)
        than linear long liquidation due to collateral devaluation convexity.
        """
        S0 = 60_000.0
        L = 10.0
        mmr = 0.005
        f_liq = 0.001

        # Linear Long: 60000 * (1 - 0.10 + 0.006) = 54,360.0
        liq_lin_long = calculate_liquidation_price(
            S0, L, "long", mmr, "linear", f_liq, slippage_rate=0.0
        )
        self.assertAlmostEqual(liq_lin_long, 54_360.0, places=4)

        # Inverse Long: 60000 / (1 + 0.10 - 0.006) = 60000 / 1.094 = 54,844.6069
        liq_inv_long = calculate_liquidation_price(
            S0, L, "long", mmr, "inverse", f_liq, slippage_rate=0.0
        )
        self.assertAlmostEqual(liq_inv_long, 60_000.0 / 1.094, places=4)

        # Inverse long barrier must be closer to entry price than linear long
        self.assertGreater(liq_inv_long, liq_lin_long)

    def test_04_bgk_barrier_shift_direction(self):
        """
        Verify Broadie-Glasserman-Kou discrete monitoring shifts the barrier
        away from spot S0, strictly reducing discrete ruin probability vs continuous.
        """
        S0 = 60_000.0
        B_long = 55_000.0
        B_short = 65_000.0
        sigma = 0.65
        dt = 1.0 / (365.0 * 24.0)

        shifted_long = calculate_bgk_barrier_shift(B_long, sigma, dt, "long")
        shifted_short = calculate_bgk_barrier_shift(B_short, sigma, dt, "short")

        self.assertLess(shifted_long, B_long)
        self.assertGreater(shifted_short, B_short)

        p_cont = analytical_ruin_probability(S0, B_long, sigma, 7.0, "long", steps_per_day=None)
        p_disc = analytical_ruin_probability(S0, B_long, sigma, 7.0, "long", steps_per_day=24)
        self.assertLess(p_disc, p_cont)

    def test_05_monte_carlo_convergence_to_analytical(self):
        """
        Verify pure GBM Monte Carlo + Brownian Bridge converges within statistical
        tolerance of the exact closed-form Bachelier-Harrison reflection formula.
        """
        np.random.seed(12345)
        S0 = 60_000.0
        collateral = 1_000.0
        leverage = 15.0
        sigma = 0.60
        T_days = 7.0

        B = calculate_liquidation_price(
            S0, leverage, "long", 0.005, "linear", 0.001, slippage_rate=0.0, collateral=collateral
        )
        exact_continuous_prob = analytical_ruin_probability(
            S0=S0, B=B, sigma=sigma, T_days=T_days, position_type="long", mu=0.0, steps_per_day=None
        )

        sim = simulate_merton_jump_diffusion_paths(
            S0=S0,
            collateral=collateral,
            leverage=leverage,
            sigma=sigma,
            T_days=T_days,
            position_type="long",
            mu=0.0,
            annual_funding_rate=0.0,
            maintenance_margin_rate=0.005,
            margin_mode="linear",
            liquidation_fee_rate=0.001,
            slippage_rate=0.0,
            num_paths=15_000,
            steps_per_day=24,
            enable_jumps=False,
            enable_stochastic_funding=False,
            enable_regime_switching=False,
            enable_stochastic_vol=False,
            enable_brownian_bridge=True,
        )

        # Standard error for M=15,000 paths is <= 0.5 / sqrt(15000) approx 0.0041
        diff = abs(sim["empirical_prob"] - exact_continuous_prob)
        self.assertLess(diff, 0.015, f"MC prob {sim['empirical_prob']:.4f} diverged from exact {exact_continuous_prob:.4f}")

    def test_06_directional_kelly_and_ruin_cap(self):
        """
        Verify Half-Kelly responds properly to both positive drift (Long) and
        negative drift (Short), while enforcing the ruin probability ceiling.
        """
        # Long with positive drift
        res_long = calculate_kelly_constrained_leverage(
            mu=0.40, sigma=0.60, entry_price=60_000.0, duration_days=7.0,
            position_type="long", max_ruin_prob=0.01, apply_shrinkage=False
        )
        # Short with negative drift (-0.40 is favorable for shorts)
        res_short = calculate_kelly_constrained_leverage(
            mu=-0.40, sigma=0.60, entry_price=60_000.0, duration_days=7.0,
            position_type="short", max_ruin_prob=0.01, apply_shrinkage=False
        )
        self.assertAlmostEqual(res_long["unconstrained_kelly"], res_short["unconstrained_kelly"], places=4)
        self.assertLessEqual(res_long["recommended_leverage"], res_long["ruin_constrained_leverage"])

        # Verify that recommended_leverage strictly obeys max_ruin_prob
        liq = calculate_liquidation_price(60_000.0, res_long["ruin_constrained_leverage"], "long")
        prob_at_bound = analytical_ruin_probability(60_000.0, liq, 0.60, 7.0, "long", steps_per_day=24)
        self.assertLessEqual(prob_at_bound, 0.01 + 1e-4)

    def test_07_volatility_estimators_and_bayesian_shrinkage(self):
        """Verify range volatility estimators and Merton-Bayes drift shrinkage."""
        np.random.seed(42)
        true_daily_vol = 0.03
        n_days = 180
        returns = np.random.normal(0.002, true_daily_vol, n_days)
        closes = 60_000.0 * np.exp(np.cumsum(returns))
        opens = np.roll(closes, 1)
        opens[0] = 60_000.0
        highs = np.maximum(opens, closes) * 1.012
        lows = np.minimum(opens, closes) * 0.988

        v_park = calculate_parkinson_volatility(highs, lows)
        v_rs = calculate_rogers_satchell_volatility(opens, highs, lows, closes)
        v_yz = calculate_yang_zhang_volatility(opens, highs, lows, closes)

        self.assertGreater(v_park, 0.10)
        self.assertGreater(v_rs, 0.10)
        self.assertGreater(v_yz, 0.10)

        drift_info = calculate_bayesian_drift(closes, sigma=v_yz, prior_mean=0.0, prior_std=0.15)
        self.assertGreater(drift_info["shrinkage_weight"], 0.0)
        self.assertLess(drift_info["shrinkage_weight"], 1.0)
        self.assertLess(abs(drift_info["shrunk_drift"]), abs(drift_info["naive_drift"]))

    def test_08_dvol_whitelist_guard(self):
        """Verify unsupported altcoins fast-fail in < 5ms without network timeout."""
        t0 = time.perf_counter()
        res = fetch_deribit_dvol("UNSUPPORTED_ALTCOIN_TICKER")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertIsNone(res)
        self.assertLess(elapsed_ms, 5.0)

    def test_09_historical_black_swan_replay(self):
        """Verify high-leverage positions liquidate on Black Thursday and survive low leverage."""
        crash_20x = replay_historical_black_swan("black_thursday_2020", 60_000.0, 20.0, "long")
        safe_1_5x = replay_historical_black_swan("black_thursday_2020", 60_000.0, 1.5, "long")
        self.assertTrue(crash_20x["liquidated"])
        self.assertIsNotNone(crash_20x["liquidation_hour"])
        self.assertFalse(safe_1_5x["liquidated"])


def run_performance_benchmarks():
    """Executes latency and throughput benchmarks across all core computational functions."""
    print("\n" + "=" * 78)
    print("⚡ PERPETUAL RUIN ENGINE v2 - PERFORMANCE & LATENCY BENCHMARK SUITE")
    print("=" * 78)

    benchmarks = []

    # 1. Analytical First-Passage Ruin (10,000 evaluations)
    t0 = time.perf_counter()
    for _ in range(10_000):
        analytical_ruin_probability(60_000.0, 55_000.0, 0.65, 7.0, "long", steps_per_day=24)
    t_analytical = (time.perf_counter() - t0) * 1000.0 / 10_000.0
    benchmarks.append(("Analytical First-Passage Ruin (per call)", f"{t_analytical * 1000:.2f} µs", "< 50 µs"))

    # 2. Bisection Half-Kelly Optimizer (40 iterations per call, averaged over 100 calls)
    t0 = time.perf_counter()
    for _ in range(100):
        calculate_kelly_constrained_leverage(0.30, 0.65, 60_000.0, 7.0, "long")
    t_kelly = (time.perf_counter() - t0) * 1000.0 / 100.0
    benchmarks.append(("Bisection Half-Kelly Solver (40 iters)", f"{t_kelly:.2f} ms", "< 5 ms"))

    # 3. 2D Ruin Heatmap Surface (25x25 = 625 grid evaluations)
    t0 = time.perf_counter()
    generate_ruin_heatmap_matrix(60_000.0, grid_size=25)
    t_heatmap = (time.perf_counter() - t0) * 1000.0
    benchmarks.append(("2D Ruin Risk Surface (25x25 = 625 pts)", f"{t_heatmap:.2f} ms", "< 25 ms"))

    # 4. Monte Carlo Engine (2,000 paths x 168 hourly steps - Full Heston + Merton + Bridge)
    t0 = time.perf_counter()
    simulate_merton_jump_diffusion_paths(
        S0=60_000.0, collateral=1_000.0, leverage=15.0, sigma=0.65, T_days=7.0,
        num_paths=2_000, steps_per_day=24, enable_jumps=True,
        enable_stochastic_vol=True, enable_regime_switching=True, enable_brownian_bridge=True
    )
    t_mc_2k = (time.perf_counter() - t0) * 1000.0
    benchmarks.append(("Monte Carlo Full Stack (2,000 paths, 7d)", f"{t_mc_2k:.1f} ms", "< 250 ms"))

    # 5. Monte Carlo Engine (5,000 paths x 168 hourly steps - Institutional Default)
    t0 = time.perf_counter()
    simulate_merton_jump_diffusion_paths(
        S0=60_000.0, collateral=1_000.0, leverage=15.0, sigma=0.65, T_days=7.0,
        num_paths=5_000, steps_per_day=24, enable_jumps=True,
        enable_stochastic_vol=True, enable_regime_switching=True, enable_brownian_bridge=True
    )
    t_mc_5k = (time.perf_counter() - t0) * 1000.0
    benchmarks.append(("Monte Carlo Full Stack (5,000 paths, 7d)", f"{t_mc_5k:.1f} ms", "< 500 ms"))

    print(f"{'Benchmark Component':<44} | {'Measured Latency':<16} | {'Target SLA'}")
    print("-" * 78)
    for name, latency, target in benchmarks:
        print(f"{name:<44} | {latency:<16} | {target}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    print("Running Mathematical & Invariant Unit Tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCoreMathEngine)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        run_performance_benchmarks()
    else:
        raise SystemExit(1)