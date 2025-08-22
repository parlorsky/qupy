"""
Momentum and oscillator indicators.

Fast implementations of momentum-based technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple

# Optional numba import for performance
try:
    from numba import jit
    HAS_NUMBA = True
except (ImportError, SystemError):
    HAS_NUMBA = False
    # Create dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .core import ArrayLike, rolling_mean, rolling_sum, rolling_max, rolling_min, diff, pct_change


# ============================================================================
# RSI (Relative Strength Index)
# ============================================================================

@jit(nopython=True)
def _rsi_calc(close: np.ndarray, n: int) -> np.ndarray:
    """Numba-optimized RSI calculation using Wilder's smoothing."""
    deltas = np.diff(close)
    seed = deltas[:n]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    
    rs = np.zeros_like(close)
    rsi = np.zeros_like(close)
    rs[:n] = np.nan
    rsi[:n] = np.nan
    
    for i in range(n, len(close)):
        delta = deltas[i-1]
        
        if delta > 0:
            up_val = delta
            down_val = 0.0
        else:
            up_val = 0.0
            down_val = -delta
        
        up = (up * (n - 1) + up_val) / n
        down = (down * (n - 1) + down_val) / n
        
        if down != 0:
            rs[i] = up / down
            rsi[i] = 100.0 - (100.0 / (1.0 + rs[i]))
        else:
            rsi[i] = 100.0
    
    return rsi


def rsi(close: ArrayLike, n: int = 14) -> ArrayLike:
    """
    Relative Strength Index.
    
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    if isinstance(close, pd.Series):
        values = close.values
        result = _rsi_calc(values, n)
        return pd.Series(result, index=close.index)
    else:
        return _rsi_calc(np.asarray(close), n)


# ============================================================================
# Stochastic Oscillator
# ============================================================================

def stoch_k(high: ArrayLike, low: ArrayLike, close: ArrayLike, n: int = 14) -> ArrayLike:
    """
    Stochastic %K.
    
    %K = 100 * (Close - Low(n)) / (High(n) - Low(n))
    """
    lowest = rolling_min(low, n)
    highest = rolling_max(high, n)
    
    k = 100 * (close - lowest) / (highest - lowest)
    
    # Handle division by zero
    if isinstance(k, pd.Series):
        k = k.replace([np.inf, -np.inf], np.nan)
    else:
        k[np.isinf(k)] = np.nan
    
    return k


def stoch_d(k: ArrayLike, n: int = 3) -> ArrayLike:
    """
    Stochastic %D (smoothed %K).
    
    %D = SMA(%K, n)
    """
    return rolling_mean(k, n)


def stoch_full(high: ArrayLike, low: ArrayLike, close: ArrayLike,
               k_period: int = 14, d_period: int = 3, 
               smooth_k: int = 3) -> Tuple[ArrayLike, ArrayLike]:
    """
    Full Stochastic Oscillator with smoothing.
    
    Returns:
    --------
    k : ArrayLike
        Smoothed %K
    d : ArrayLike
        %D (smoothed %K)
    """
    # Calculate raw %K
    raw_k = stoch_k(high, low, close, k_period)
    
    # Smooth %K
    k = rolling_mean(raw_k, smooth_k)
    
    # Calculate %D
    d = rolling_mean(k, d_period)
    
    return k, d


def stochastic(high: ArrayLike, low: ArrayLike, close: ArrayLike,
               k_period: int = 14, d_period: int = 3) -> dict:
    """
    Simple Stochastic Oscillator wrapper.
    
    Returns:
    --------
    dict with keys: '%K', '%D'
    """
    k = stoch_k(high, low, close, k_period)
    d = stoch_d(k, d_period)
    
    return {
        '%K': k,
        '%D': d
    }


# ============================================================================
# Williams %R
# ============================================================================

def williams_r(high: ArrayLike, low: ArrayLike, close: ArrayLike, n: int = 14) -> ArrayLike:
    """
    Williams %R.
    
    %R = -100 * (High(n) - Close) / (High(n) - Low(n))
    """
    highest = rolling_max(high, n)
    lowest = rolling_min(low, n)
    
    wr = -100 * (highest - close) / (highest - lowest)
    
    # Handle division by zero
    if isinstance(wr, pd.Series):
        wr = wr.replace([np.inf, -np.inf], np.nan)
    else:
        wr[np.isinf(wr)] = np.nan
    
    return wr


# ============================================================================
# Rate of Change and Momentum
# ============================================================================

def roc(close: ArrayLike, n: int = 10) -> ArrayLike:
    """
    Rate of Change.
    
    ROC = 100 * (Close - Close[n]) / Close[n]
    """
    return 100 * pct_change(close, n)


def mom(close: ArrayLike, n: int = 10) -> ArrayLike:
    """
    Momentum.
    
    MOM = Close - Close[n]
    """
    return diff(close, n)


def price_rate_of_change(close: ArrayLike, n: int = 10) -> ArrayLike:
    """Price Rate of Change (alias for ROC)."""
    return roc(close, n)


# ============================================================================
# MACD (Moving Average Convergence Divergence)
# ============================================================================

def macd(close: ArrayLike, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    MACD indicator.
    
    Returns:
    --------
    dict with keys: 'macd', 'signal', 'histogram'
    """
    from .trend import ema
    
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def macd_signal(close: ArrayLike, fast: int = 12, slow: int = 26, signal: int = 9) -> ArrayLike:
    """MACD signal line only."""
    _, signal_line, _ = macd(close, fast, slow, signal)
    return signal_line


def macd_hist(close: ArrayLike, fast: int = 12, slow: int = 26, signal: int = 9) -> ArrayLike:
    """MACD histogram only."""
    _, _, histogram = macd(close, fast, slow, signal)
    return histogram


# ============================================================================
# PPO (Percentage Price Oscillator)
# ============================================================================

def ppo(close: ArrayLike, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Percentage Price Oscillator.
    
    Similar to MACD but expressed as percentage.
    """
    from .trend import ema
    
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    
    ppo_line = 100 * (ema_fast - ema_slow) / ema_slow
    signal_line = ema(ppo_line, signal)
    histogram = ppo_line - signal_line
    
    return ppo_line, signal_line, histogram


# ============================================================================
# CCI (Commodity Channel Index)
# ============================================================================

def cci(high: ArrayLike, low: ArrayLike, close: ArrayLike, n: int = 20) -> ArrayLike:
    """
    Commodity Channel Index.
    
    CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
    """
    from .core import hlc3
    
    tp = hlc3(high, low, close)
    sma_tp = rolling_mean(tp, n)
    
    # Calculate mean deviation
    if isinstance(tp, pd.Series):
        mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    else:
        tp_series = pd.Series(tp)
        mad = tp_series.rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    
    cci_val = (tp - sma_tp) / (0.015 * mad)
    
    # Handle division by zero
    if isinstance(cci_val, pd.Series):
        cci_val = cci_val.replace([np.inf, -np.inf], np.nan)
    else:
        cci_val[np.isinf(cci_val)] = np.nan
    
    return cci_val


# ============================================================================
# TSI (True Strength Index)
# ============================================================================

def tsi(close: ArrayLike, r: int = 25, s: int = 13) -> ArrayLike:
    """
    True Strength Index.
    
    Double-smoothed momentum oscillator.
    """
    from .trend import ema
    
    momentum = diff(close, 1)
    
    # Double smooth momentum
    ema_momentum = ema(ema(momentum, r), s)
    
    # Double smooth absolute momentum
    ema_abs_momentum = ema(ema(np.abs(momentum), r), s)
    
    tsi_val = 100 * ema_momentum / ema_abs_momentum
    
    # Handle division by zero
    if isinstance(tsi_val, pd.Series):
        tsi_val = tsi_val.replace([np.inf, -np.inf], np.nan)
    else:
        tsi_val[np.isinf(tsi_val)] = np.nan
    
    return tsi_val


# ============================================================================
# Awesome Oscillator
# ============================================================================

def awesome_oscillator(high: ArrayLike, low: ArrayLike, fast: int = 5, slow: int = 34) -> ArrayLike:
    """
    Awesome Oscillator.
    
    AO = SMA(HL2, fast) - SMA(HL2, slow)
    """
    from .core import hl2
    from .trend import sma
    
    hl_avg = hl2(high, low)
    
    sma_fast = sma(hl_avg, fast)
    sma_slow = sma(hl_avg, slow)
    
    return sma_fast - sma_slow


# ============================================================================
# Ultimate Oscillator
# ============================================================================

def ultimate_oscillator(high: ArrayLike, low: ArrayLike, close: ArrayLike,
                        period1: int = 7, period2: int = 14, period3: int = 28) -> ArrayLike:
    """
    Ultimate Oscillator.
    
    Combines short, medium, and long-term momentum.
    """
    from .volatility import true_range
    
    # Calculate buying pressure
    if isinstance(close, pd.Series):
        prev_close = close.shift(1)
    else:
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
    
    min_low_pc = np.minimum(low, prev_close)
    bp = close - min_low_pc
    
    # Calculate true range
    tr = true_range(high, low, prev_close)
    
    # Calculate average for each period
    avg1 = rolling_sum(bp, period1) / rolling_sum(tr, period1)
    avg2 = rolling_sum(bp, period2) / rolling_sum(tr, period2)
    avg3 = rolling_sum(bp, period3) / rolling_sum(tr, period3)
    
    # Weighted average
    uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
    
    return uo


# ============================================================================
# Money Flow Index
# ============================================================================

def mfi(high: ArrayLike, low: ArrayLike, close: ArrayLike, volume: ArrayLike, n: int = 14) -> ArrayLike:
    """
    Money Flow Index.
    
    Volume-weighted RSI.
    """
    from .core import hlc3
    
    tp = hlc3(high, low, close)
    raw_money_flow = tp * volume
    
    # Calculate money flow direction
    if isinstance(tp, pd.Series):
        tp_change = tp.diff()
    else:
        tp_change = diff(tp, 1)
    
    # Positive and negative money flow
    positive_flow = np.where(tp_change > 0, raw_money_flow, 0)
    negative_flow = np.where(tp_change < 0, raw_money_flow, 0)
    
    # Sum over period
    positive_sum = rolling_sum(positive_flow, n)
    negative_sum = rolling_sum(negative_flow, n)
    
    # Money flow ratio and index
    mfr = positive_sum / negative_sum
    mfi_val = 100 - (100 / (1 + mfr))
    
    # Handle division by zero
    if isinstance(mfi_val, pd.Series):
        mfi_val = mfi_val.replace([np.inf, -np.inf], np.nan)
        mfi_val[negative_sum == 0] = 100
    else:
        mfi_val[np.isinf(mfi_val)] = np.nan
        mfi_val[negative_sum == 0] = 100
    
    return mfi_val


def money_flow_index(high: ArrayLike, low: ArrayLike, close: ArrayLike, volume: ArrayLike, n: int = 14) -> ArrayLike:
    """Money Flow Index (alias)."""
    return mfi(high, low, close, volume, n)


# ============================================================================
# Force Index
# ============================================================================

def force_index(close: ArrayLike, volume: ArrayLike, n: int = 13) -> ArrayLike:
    """
    Force Index.
    
    FI = EMA((Close - Close[1]) * Volume, n)
    """
    from .trend import ema
    
    price_change = diff(close, 1)
    raw_force = price_change * volume
    
    return ema(raw_force, n)


# ============================================================================
# Chande Momentum Oscillator
# ============================================================================

def chande_momentum_oscillator(close: ArrayLike, n: int = 14) -> ArrayLike:
    """
    Chande Momentum Oscillator.
    
    CMO = 100 * (Sum(Up) - Sum(Down)) / (Sum(Up) + Sum(Down))
    """
    price_change = diff(close, 1)
    
    up_sum = rolling_sum(np.where(price_change > 0, price_change, 0), n)
    down_sum = rolling_sum(np.where(price_change < 0, -price_change, 0), n)
    
    cmo = 100 * (up_sum - down_sum) / (up_sum + down_sum)
    
    # Handle division by zero
    if isinstance(cmo, pd.Series):
        cmo = cmo.replace([np.inf, -np.inf], np.nan)
    else:
        cmo[np.isinf(cmo)] = np.nan
    
    return cmo