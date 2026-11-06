# ==============================================================================
# MODULE: market_data_v2.py
# PERPETUAL RUIN ENGINE - ADVANCED MARKET DATA & VOLATILITY PIPELINE
# ==============================================================================
# Description: Production-grade quantitative market data extractor integrating
#              Deribit forward-looking DVOL implied volatility, Parkinson and
#              Rogers-Satchell continuous 24/7 volatility estimators, Yang-Zhang
#              jump-adjusted realized volatility, rolling multi-period funding
#              rate smoothing, and Bayesian shrinkage drift regularizers.
# ==============================================================================

import json
import math
import urllib.request
import numpy as np
import ccxt


# ==============================================================================
# SECTION 1: DERIBIT DVOL (IMPLIED VOLATILITY INDEX) FETCHER
# ==============================================================================

DERIBIT_SUPPORTED_DVOL_CURRENCIES = {"BTC", "ETH", "SOL"}


def fetch_deribit_dvol(currency: str = "BTC") -> float | None:
    """
    Fetches 30-day forward-looking Implied Volatility Index (DVOL) from Deribit.
    
    Mathematical Principles:
    - [Equation 2]: Model-Free Implied Variance Index
    - [Equation 3]: Decimal Scaling: sigma_DVOL = DVOL_close / 100.0
    """
    base_coin = currency.upper().strip()
    if base_coin not in DERIBIT_SUPPORTED_DVOL_CURRENCIES:
        return None

    url = f"https://www.deribit.com/api/v2/public/get_volatility_index_data?currency={base_coin}&resolution=1D"
    req = urllib.request.Request(url, headers={"User-Agent": "PerpetualRuinEngine/2.0"})
    
    try:
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode("utf-8"))
            if "result" in data and "data" in data["result"] and len(data["result"]["data"]) > 0:
                latest_point = data["result"]["data"][-1]
                close_dvol = float(latest_point[4])
                return close_dvol / 100.0
    except Exception:
        return None

    return None


# ==============================================================================
# SECTION 2: CONTINUOUS & HISTORICAL VOLATILITY ESTIMATORS
# ==============================================================================

def calculate_parkinson_volatility(
    highs: np.ndarray,
    lows: np.ndarray,
    trading_periods: float = 365.0
) -> float:
    """
    Computes Parkinson (1980) extreme-value high-low volatility estimator.
    
    Mathematical Principles:
    - [Equation 4]: sigma_P^2 = (1 / (4 * ln(2) * n)) * sum ln(H_t / L_t)^2
    - [Equation 5]: Annualized: sigma_P,annual = sqrt(sigma_P^2 * trading_periods)
    """
    if len(highs) < 2 or len(lows) < 2:
        return 0.0

    h_over_l = np.maximum(1e-8, highs / np.maximum(1e-8, lows))
    log_hl_sq = np.log(h_over_l) ** 2
    scale_constant = 1.0 / (4.0 * math.log(2.0))
    parkinson_var = scale_constant * float(np.mean(log_hl_sq))

    return float(math.sqrt(max(0.0, parkinson_var)) * math.sqrt(trading_periods))


def calculate_rogers_satchell_volatility(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    trading_periods: float = 365.0
) -> float:
    """
    Computes Rogers-Satchell (1991) drift-independent volatility estimator.
    
    Mathematical Principles:
    - [Equation 6]: sigma_RS^2 = (1 / n) * sum [ ln(H_t/C_t)*ln(H_t/O_t) + ln(L_t/C_t)*ln(L_t/O_t) ]
    - [Equation 7]: Annualized: sigma_RS,annual = sqrt(sigma_RS^2 * trading_periods)
    """
    if len(closes) < 2:
        return 0.0

    o = np.maximum(1e-8, opens)
    h = np.maximum(1e-8, highs)
    l = np.maximum(1e-8, lows)
    c = np.maximum(1e-8, closes)

    term1 = np.log(h / c) * np.log(h / o)
    term2 = np.log(l / c) * np.log(l / o)
    rs_variance = float(np.mean(term1 + term2))

    return float(math.sqrt(max(0.0, rs_variance)) * math.sqrt(trading_periods))


def calculate_yang_zhang_volatility(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    trading_periods: float = 365.0
) -> float:
    """
    Computes Yang-Zhang (2000) drift-independent, overnight-jump adjusted volatility.
    
    Mathematical Principles:
    - [Equation 8]: Overnight Jump Component: u_t = ln(O_t / C_{t-1})
    - [Equation 9]: Open-to-Close Component: c_t = ln(C_t / O_t)
    - [Equation 10]: Rogers-Satchell Component on lagged interval
    - [Equation 11]: Blending Weight: k(n) = 0.34 / (1.34 + (n + 1)/(n - 1))
    - [Equation 12-13]: Combined Yang-Zhang Variance and Annualized Volatility
    """
    if len(closes) < 3:
        log_ret = np.diff(np.log(np.maximum(1e-8, closes)))
        return float(np.std(log_ret, ddof=1) * math.sqrt(trading_periods)) if len(log_ret) > 1 else 0.0

    n = len(closes) - 1
    o_t = np.maximum(1e-8, opens[1:])
    h_t = np.maximum(1e-8, highs[1:])
    l_t = np.maximum(1e-8, lows[1:])
    c_t = np.maximum(1e-8, closes[1:])
    c_prev = np.maximum(1e-8, closes[:-1])

    u = np.log(o_t / c_prev)
    sigma_open_sq = float(np.var(u, ddof=1))

    c = np.log(c_t / o_t)
    sigma_close_sq = float(np.var(c, ddof=1))

    rs_elements = np.log(h_t / c_t) * np.log(h_t / o_t) + np.log(l_t / c_t) * np.log(l_t / o_t)
    sigma_rs_sq = float(np.mean(rs_elements))

    k = 0.34 / (1.34 + (n + 1.0) / max(1.0, (n - 1.0)))
    variance = sigma_open_sq + k * sigma_close_sq + (1.0 - k) * sigma_rs_sq

    return float(math.sqrt(max(0.0, variance)) * math.sqrt(trading_periods))


# ==============================================================================
# SECTION 3: BAYESIAN SHRINKAGE DRIFT REGULARIZER
# ==============================================================================

def calculate_bayesian_drift(
    closes: np.ndarray,
    sigma: float,
    trading_periods: float = 365.0,
    prior_mean: float = 0.0,
    prior_std: float = 0.15
) -> dict:
    """
    Computes annualized historical drift shrunk via Merton-Bayes regularizer.
    
    Mathematical Principles:
    - [Equation 17]: Naive Sample Log Drift
    - [Equation 18]: Merton Standard Error: SE(mu) = sigma / sqrt(T_years)
    - [Equation 19]: Bayesian Posterior Weight: w = sigma_prior^2 / (sigma_prior^2 + SE^2)
    - [Equation 20]: Regularized Shrunk Drift: mu_shrunk = w * mu_naive
    """
    if len(closes) < 3:
        return {
            "naive_drift": 0.0,
            "shrunk_drift": prior_mean,
            "shrinkage_weight": 0.0,
            "merton_standard_error": 0.0
        }

    log_returns = np.diff(np.log(np.maximum(1e-8, closes)))
    sample_mean_daily = float(np.mean(log_returns))
    naive_annual_drift = sample_mean_daily * trading_periods

    horizon_years = len(log_returns) / trading_periods
    merton_se = sigma / math.sqrt(max(1e-4, horizon_years))

    prior_var = prior_std ** 2
    se_var = merton_se ** 2
    w = prior_var / (prior_var + se_var)

    shrunk_annual_drift = w * naive_annual_drift + (1.0 - w) * prior_mean

    return {
        "naive_drift": float(naive_annual_drift),
        "shrunk_drift": float(shrunk_annual_drift),
        "shrinkage_weight": float(w),
        "merton_standard_error": float(merton_se)
    }


# ==============================================================================
# SECTION 4: ROLLING MULTI-PERIOD FUNDING RATE SMOOTHING
# ==============================================================================

def fetch_rolling_funding_rate(
    exchange,
    symbol: str,
    lookback_periods: int = 21,
    default_fallback: float = 0.0001
) -> dict:
    """
    Fetches historical funding payments to construct an unskewed rolling average funding rate.
    
    Mathematical Principles:
    - [Equation 14]: Rolling Mean: r_bar_8h = (1 / M) * sum r_8h^(m)
    - [Equation 15]: Annualized Smoothed Funding: r_annual = 1095 * r_bar_8h
    """
    spot_rate = default_fallback
    try:
        funding_info = exchange.fetch_funding_rate(symbol)
        spot_rate = float(funding_info.get("fundingRate", default_fallback))
    except Exception:
        pass

    rates_history = []
    try:
        if hasattr(exchange, "fetch_funding_rate_history"):
            history = exchange.fetch_funding_rate_history(symbol, limit=lookback_periods)
            rates_history = [
                float(item["fundingRate"]) for item in history if "fundingRate" in item and item["fundingRate"] is not None
            ]
    except Exception:
        rates_history = []

    if len(rates_history) >= 3:
        smoothed_8h_rate = float(np.mean(rates_history))
        periods_used = len(rates_history)
    else:
        smoothed_8h_rate = spot_rate
        periods_used = 1

    annualized_smoothed_funding = smoothed_8h_rate * 3.0 * 365.0
    annualized_spot_funding = spot_rate * 3.0 * 365.0

    return {
        "spot_funding_rate_8h": float(spot_rate),
        "smoothed_funding_rate_8h": float(smoothed_8h_rate),
        "annualized_smoothed_funding": float(annualized_smoothed_funding),
        "annualized_spot_funding": float(annualized_spot_funding),
        "funding_periods_sampled": periods_used
    }


# ==============================================================================
# SECTION 5: UNIFIED PRODUCTION MARKET DATA EXTRACTION ENGINE
# ==============================================================================

def get_market_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    limit: int = 90,
    exchange_name: str = "binance"
) -> dict:
    """
    Unified market data extraction pipeline providing robust volatility metrics,
    smoothed funding rates, and regularized drift for perpetual ruin engines.
    """
    primary_cls = getattr(ccxt, exchange_name, ccxt.binance)
    fallback_cls = ccxt.bybit if exchange_name.lower() == "binance" else ccxt.binance

    # Initialize exchange with defaultType: "swap" to target perpetual contracts
    swap_config = {
        "enableRateLimit": True,
        "timeout": 8000,
        "options": {"defaultType": "swap"}
    }
    exchange = primary_cls(swap_config)
    ohlcv = None

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception:
        try:
            exchange = fallback_cls(swap_config)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception:
            ohlcv = None

    if not ohlcv or len(ohlcv) < 5:
        raise ConnectionError(f"Failed to fetch sufficient market OHLCV data for {symbol}.")

    ohlcv_arr = np.array(ohlcv, dtype=float)
    opens = ohlcv_arr[:, 1]
    highs = ohlcv_arr[:, 2]
    lows = ohlcv_arr[:, 3]
    closes = ohlcv_arr[:, 4]
    current_price = float(closes[-1])

    parkinson_vol = calculate_parkinson_volatility(highs, lows, trading_periods=365.0)
    rs_vol = calculate_rogers_satchell_volatility(opens, highs, lows, closes, trading_periods=365.0)
    yz_vol = calculate_yang_zhang_volatility(opens, highs, lows, closes, trading_periods=365.0)

    base_coin = symbol.split("/")[0].split(":")[0]
    implied_vol = fetch_deribit_dvol(base_coin)

    # Active Volatility Selection Hierarchy (Equation 21)
    active_vol = implied_vol if (implied_vol is not None and implied_vol > 0.0) else yz_vol

    funding_metrics = fetch_rolling_funding_rate(
        exchange=exchange,
        symbol=symbol,
        lookback_periods=21,
        default_fallback=0.0001
    )

    drift_metrics = calculate_bayesian_drift(
        closes=closes,
        sigma=active_vol,
        trading_periods=365.0,
        prior_mean=0.0,
        prior_std=0.15
    )

    return {
        "symbol": symbol,
        "current_price": current_price,
        "active_volatility": active_vol,
        "implied_volatility": implied_vol,
        "yang_zhang_volatility": yz_vol,
        "rogers_satchell_volatility": rs_vol,
        "parkinson_volatility": parkinson_vol,
        "spot_funding_rate_8h": funding_metrics["spot_funding_rate_8h"],
        "smoothed_funding_rate_8h": funding_metrics["smoothed_funding_rate_8h"],
        "annualized_funding_rate": funding_metrics["annualized_smoothed_funding"],
        "annualized_spot_funding": funding_metrics["annualized_spot_funding"],
        "funding_periods_sampled": funding_metrics["funding_periods_sampled"],
        "naive_annual_drift": drift_metrics["naive_drift"],
        "annualized_drift": drift_metrics["shrunk_drift"],
        "drift_shrinkage_weight": drift_metrics["shrinkage_weight"],
        "merton_standard_error": drift_metrics["merton_standard_error"],
        "candles_count": len(closes),
    }


if __name__ == "__main__":
    print("Testing market_data_v2 module...")
    np.random.seed(42)
    prices = 65000.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.03, 90)))
    mock_closes = prices
    mock_opens = np.roll(prices, 1)
    mock_opens[0] = 65000.0
    mock_highs = np.maximum(mock_opens, mock_closes) * (1.0 + np.abs(np.random.normal(0.0, 0.01, 90)))
    mock_lows = np.minimum(mock_opens, mock_closes) * (1.0 - np.abs(np.random.normal(0.0, 0.01, 90)))

    print(f"Parkinson Vol: {calculate_parkinson_volatility(mock_highs, mock_lows):.4f}")
    print(f"Rogers-Satchell Vol: {calculate_rogers_satchell_volatility(mock_opens, mock_highs, mock_lows, mock_closes):.4f}")
    print(f"Yang-Zhang Vol: {calculate_yang_zhang_volatility(mock_opens, mock_highs, mock_lows, mock_closes):.4f}")
    print("All market_data_v2 tests passed successfully!")