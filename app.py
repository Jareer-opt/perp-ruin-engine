# Purpose: Upgraded quantitative risk dashboard connecting core_math_v2 and market_data_v2.
# Features: Heston Stochastic Volatility, Regime-Switching Funding, Inverse Margin Mode,
#           Parkinson/Rogers-Satchell Estimators, Half-Kelly Optimization, and Short Squeeze Replay.

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from core_math_v2 import (
    calculate_liquidation_price,
    analytical_ruin_probability,
    simulate_merton_jump_diffusion_paths,
    generate_ruin_heatmap_matrix,
    calculate_kelly_constrained_leverage,
    replay_historical_black_swan,
    norm_cdf
)
from market_data_v2 import get_market_data

st.set_page_config(
    page_title="Institutional Perpetual Risk Engine v2",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stMetric {
        background-color: #0e1117;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #1f2430;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Quantitative Margin Ruin & Liquidation Engine (v2)")
st.caption("Institutional-grade framework evaluating Heston stochastic volatility, Merton jumps, rolling funding bleed, inverse coin margins, and tail risk.")

# ----------------- Sidebar Controls -----------------
st.sidebar.header("⚙️ Market & Asset Setup")
pair = st.sidebar.selectbox("Trading Pair", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])

@st.cache_data(ttl=60)
def fetch_cached_market(symbol):
    return get_market_data(symbol, timeframe="1d", limit=90)

with st.sidebar:
    with st.spinner("Connecting to exchanges & Deribit DVOL..."):
        try:
            m_data = fetch_cached_market(pair)
            live_price = m_data["current_price"]
            yz_vol = m_data["yang_zhang_volatility"]
            rs_vol = m_data["rogers_satchell_volatility"]
            park_vol = m_data["parkinson_volatility"]
            dvol = m_data["implied_volatility"]
            f_rate_spot = m_data["spot_funding_rate_8h"]
            f_rate_smoothed = m_data["smoothed_funding_rate_8h"]
            drift = m_data["annualized_drift"]
        except Exception:
            live_price = 65000.0 if "BTC" in pair else (2500.0 if "ETH" in pair else 100.0)
            yz_vol = 0.58
            rs_vol = 0.55
            park_vol = 0.54
            dvol = 0.62
            f_rate_spot = 0.0001
            f_rate_smoothed = 0.0001
            drift = 0.0

    col_side1, col_side2 = st.columns(2)
    with col_side1:
        position_type = st.radio("Direction", ["long", "short"], horizontal=True)
    with col_side2:
        margin_mode = st.radio("Margin Mode", ["linear", "inverse"], horizontal=True, 
                               help="Linear = USDT/USDC margined; Inverse = Coin-margined (e.g. BTC)")

    entry_price = st.number_input("Entry Price ($)", value=float(live_price), step=10.0)
    leverage = st.slider("Leverage Multiple", min_value=2, max_value=50, value=15, step=1)
    collateral = st.number_input("Margin Collateral ($)", min_value=50.0, value=1000.0, step=50.0)
    duration_days = st.slider("Horizon (Days)", min_value=1, max_value=30, value=7, step=1)

    st.sidebar.subheader("🔬 Volatility & Funding Dynamics")
    
    vol_options = [
        "Deribit Implied Vol (DVOL)" if dvol else "DVOL (Unavailable)",
        "Yang-Zhang (Overnight Adjusted)",
        "Rogers-Satchell (24/7 Continuous)",
        "Parkinson (High-Low)"
    ]
    vol_mode = st.selectbox("Volatility Estimator", vol_options)
    
    if "Deribit" in vol_mode and dvol:
        selected_sigma = dvol
    elif "Rogers-Satchell" in vol_mode:
        selected_sigma = rs_vol
    elif "Parkinson" in vol_mode:
        selected_sigma = park_vol
    else:
        selected_sigma = yz_vol

    sigma = st.slider("Active Sigma (Annualized)", min_value=0.10, max_value=2.00, value=float(selected_sigma), step=0.01)

    enable_funding = st.checkbox("Account for Funding Drag", value=True)
    funding_type = st.radio("Funding Baseline", ["Rolling 7d Smoothed", "Spot Instantaneous"], horizontal=True)
    base_f_rate = f_rate_smoothed if "Smoothed" in funding_type else f_rate_spot

    custom_funding_8h = st.number_input(
        "8h Funding Rate (%)", value=float(base_f_rate * 100.0), step=0.005, format="%.4f"
    ) / 100.0

    slippage_buffer = st.slider("Adverse Slippage Buffer (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05) / 100.0

# ----------------- Top KPI Cards -----------------
annual_funding = (custom_funding_8h * 3.0 * 365.0) if enable_funding else 0.0
position_size = collateral * leverage
initial_liq = calculate_liquidation_price(
    entry_price=entry_price, 
    leverage=leverage, 
    position_type=position_type,
    margin_mode=margin_mode,
    slippage_rate=slippage_buffer,
    collateral=collateral
)
analytical_prob = analytical_ruin_probability(
    S0=entry_price, 
    B=initial_liq, 
    sigma=sigma, 
    T_days=duration_days, 
    position_type=position_type, 
    mu=0.0,
    steps_per_day=24
)

col1, col2, col3, col4 = st.columns(4)
pct_to_liq = ((initial_liq - entry_price) / entry_price) * 100.0
col1.metric("Effective Liquidation Price", f"${initial_liq:,.2f}", f"{pct_to_liq:.2f}% distance", delta_color="inverse")
col2.metric("Total Position Exposure", f"${position_size:,.2f}", f"Collateral: ${collateral:,.0f} ({margin_mode.capitalize()})")
col3.metric("BGK Corrected Ruin Prob", f"{analytical_prob * 100:.2f}%", f"{duration_days}d Horizon")
col4.metric("Annualized Funding Drag", f"{annual_funding * 100:.2f}%", f"{custom_funding_8h * 100:.4f}% / 8h")

st.markdown("---")

# ----------------- Dashboard Tabs -----------------
tab1, tab2, tab3 = st.tabs([
    "📈 Stochastic Diffusion & Bleed Simulator", 
    "🗺️ 2D Ruin Heatmap & Half-Kelly Solver", 
    "⏳ Historical Black Swan & Squeeze Replay"
])

# ----------------- Tab 1: Stochastic Simulation -----------------
with tab1:
    col_ctrl, col_graph = st.columns([1, 2.5])
    with col_ctrl:
        st.subheader("Advanced Process Controls")
        
        enable_jumps = st.checkbox("Enable Merton Jumps (Tail Drops)", value=True)
        jump_lambda = st.slider("Jump Intensity (λ/year)", 0.5, 12.0, 3.0, 0.5)
        jump_mu = st.slider("Mean Jump Size (%)", -25.0, 5.0, -8.0, 1.0) / 100.0
        jump_sig = st.slider("Jump Volatility (%)", 2.0, 25.0, 12.0, 1.0) / 100.0

        st.markdown("---")
        enable_heston = st.checkbox("Enable Heston Volatility Clustering", value=True)
        heston_kappa = st.slider("Vol Mean-Reversion (κ)", 0.5, 10.0, 3.0, 0.5)
        heston_vol_vol = st.slider("Vol-of-Vol (σ_v)", 0.05, 1.0, 0.35, 0.05)
        heston_corr = st.slider("Price-Vol Correlation (ρ_Sv)", -0.95, 0.0, -0.60, 0.05)

        st.markdown("---")
        enable_regime = st.checkbox("Enable Funding Regime Switching", value=True)
        bear_funding = st.slider("Bear Regime Funding (%/8h)", -0.10, 0.0, -0.02, 0.005) / 100.0 * 3.0 * 365.0
        enable_bridge = st.checkbox("Brownian Bridge Sub-step Breach", value=True)

    with col_graph:
        with st.spinner("Running Monte Carlo simulation (Heston + Merton + Dynamic Barrier)..."):
            sim = simulate_merton_jump_diffusion_paths(
                S0=entry_price,
                collateral=collateral,
                leverage=leverage,
                sigma=sigma,
                T_days=duration_days,
                position_type=position_type,
                mu=0.0,
                annual_funding_rate=annual_funding,
                margin_mode=margin_mode,
                slippage_rate=slippage_buffer,
                num_paths=2000,
                steps_per_day=24,
                enable_jumps=enable_jumps,
                jump_intensity_lambda=jump_lambda,
                jump_mean_mu=jump_mu,
                jump_std_sigma=jump_sig,
                enable_stochastic_funding=True,
                enable_regime_switching=enable_regime,
                funding_bear_rate=bear_funding,
                enable_stochastic_vol=enable_heston,
                heston_kappa=heston_kappa,
                heston_vol_of_vol=heston_vol_vol,
                heston_price_vol_corr=heston_corr,
                enable_brownian_bridge=enable_bridge
            )

        st.subheader(f"Simulated Trajectories (Empirical Ruin: **{sim['empirical_prob'] * 100:.2f}%** | Avg Vol: **{sim['average_volatility']*100:.1f}%**)")
        
        fig = go.Figure()
        # Plot sample paths with exact breach tracking
        breached_flags = sim.get("sample_paths_breached", [False] * len(sim["sample_paths"]))
        for i, path in enumerate(sim["sample_paths"][:35]):
            is_breached = breached_flags[i] if i < len(breached_flags) else False
            color = "rgba(239, 83, 80, 0.2)" if is_breached else "rgba(100, 181, 246, 0.25)"
            fig.add_trace(go.Scatter(x=sim["time_grid"], y=path, mode="lines", line=dict(color=color, width=1), showlegend=False, hoverinfo="skip"))

        fig.add_trace(go.Scatter(x=[0, duration_days], y=[entry_price, entry_price], mode="lines", name="Entry Price", line=dict(color="#29b6f6", width=2, dash="dash")))

        fig.add_trace(go.Scatter(
            x=sim["time_grid"], y=sim["dynamic_barriers"], 
            mode="lines", name=f"Dynamic Liq Barrier ({margin_mode.capitalize()} Bleed)", 
            line=dict(color="#f44336", width=2.5)
        ))

        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Days",
            yaxis_title="Price ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

# ----------------- Tab 2: 2D Heatmap & Half-Kelly -----------------
with tab2:
    col_h1, col_h2 = st.columns([2, 1])

    with col_h2:
        st.subheader("Optimal Sizing Engine")
        expected_drift = st.slider("Expected Annual Asset Drift (μ)", -0.50, 1.50, float(drift) if drift != 0.0 else 0.20, 0.05)
        risk_cap = st.slider("Max Acceptable Ruin Probability (%)", 0.1, 5.0, 1.0, 0.1) / 100.0
        use_half_kelly = st.checkbox("Apply Half-Kelly Conservatism (Recommended)", value=True)

        # apply_shrinkage=False because drift from market_data_v2 is already Bayesian-shrunk
        kelly_results = calculate_kelly_constrained_leverage(
            mu=expected_drift,
            sigma=sigma,
            entry_price=entry_price,
            duration_days=duration_days,
            position_type=position_type,
            max_ruin_prob=risk_cap,
            margin_mode=margin_mode,
            slippage_rate=slippage_buffer,
            use_half_kelly=use_half_kelly,
            apply_shrinkage=False
        )

        st.metric("Bayesian Shrunk Kelly Leverage", f"{kelly_results['unconstrained_kelly']:.2f}x")
        st.metric("Ruin-Constrained Safe Bound", f"{kelly_results['ruin_constrained_leverage']:.2f}x")
        st.success(f"💡 **Recommended Sizing:** **{kelly_results['recommended_leverage']:.1f}x**")

        st.caption("Bayesian shrinkage regularizes sample drift error, while ruin constraints prevent geometric growth from crossing the liquidation absorbing barrier.")

    with col_h1:
        st.subheader(f"2D Ruin Probability Surface ({margin_mode.capitalize()} Mode)")
        levs, sigs, z_matrix = generate_ruin_heatmap_matrix(
            entry_price=entry_price,
            duration_days=duration_days,
            position_type=position_type,
            margin_mode=margin_mode,
            slippage_rate=slippage_buffer,
            grid_size=20
        )

        fig_heat = go.Figure(data=go.Heatmap(
            z=z_matrix * 100.0,
            x=levs,
            y=sigs * 100.0,
            colorscale="Viridis",
            colorbar=dict(title="Ruin %")
        ))

        fig_heat.add_trace(go.Scatter(
            x=[leverage], y=[sigma * 100.0],
            mode="markers",
            marker=dict(color="red", size=14, symbol="cross", line=dict(color="white", width=2)),
            name="Your Position"
        ))

        fig_heat.update_layout(
            template="plotly_dark",
            xaxis_title="Leverage Multiple",
            yaxis_title="Annualized Volatility (%)",
            margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ----------------- Tab 3: Historical Black Swan & Squeeze Replay -----------------
with tab3:
    st.subheader("Historical Liquidity Cascade & Short Squeeze Replay")
    st.write("Replays your exact leverage, margin regime, and liquidation barrier against empirical market extremes.")

    scenario = st.selectbox(
        "Select Historical Shock Event", 
        [
            ("black_thursday_2020", "March 12, 2020: Black Thursday (-50% Liquidity Cascade)"),
            ("china_ban_2021", "May 19, 2021: China Mining Ban Flash Dump (-32%)"),
            ("ftx_collapse_2022", "November 8, 2022: FTX Balance Sheet Insolvency (-25%)"),
            ("short_squeeze_2021", "July 26, 2021: Amazon Rumor Short Squeeze Spike (+35%)")
        ],
        format_func=lambda x: x[1]
    )[0]

    replay_data = replay_historical_black_swan(
        scenario_key=scenario,
        entry_price=entry_price,
        leverage=leverage,
        position_type=position_type,
        margin_mode=margin_mode,
        slippage_rate=slippage_buffer
    )

    if replay_data["liquidated"]:
        st.error(f"💀 **Position Liquidated!** Under this scenario, your position would have been forcefully liquidated at **Hour {replay_data['liquidation_hour']}**.")
    else:
        st.success(f"🛡️ **Position Survived!** Maximum adverse excursion during the event was **{replay_data['max_drawdown_pct'] * 100:.2f}%**, remaining clear of your liquidation barrier.")

    fig_replay = go.Figure()
    fig_replay.add_trace(go.Scatter(
        x=replay_data["timeline_hours"], y=replay_data["trajectory"],
        mode="lines", name="Historical Replay Path",
        line=dict(color="#ffa726", width=2.5)
    ))
    fig_replay.add_trace(go.Scatter(
        x=[0, replay_data["timeline_hours"][-1]], y=[replay_data["liq_price"], replay_data["liq_price"]],
        mode="lines", name="Effective Liquidation Barrier",
        line=dict(color="#f44336", width=2, dash="dash")
    ))

    fig_replay.update_layout(
        template="plotly_dark",
        xaxis_title="Event Timeline (Elapsed Hours)",
        yaxis_title="Asset Price ($)",
        margin=dict(l=10, r=10, t=20, b=10)
    )
    st.plotly_chart(fig_replay, use_container_width=True)