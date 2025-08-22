"""
Volatility estimators and range-based indicators.

Optimized implementations of volatility measures using various estimators.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional

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

from .core import ArrayLike, rolling_mean, rolling_std, rolling_max, rolling_min


# ============================================================================
# True Range and ATR
# ============================================================================

def true_range(high: ArrayLike, low: ArrayLike, prev_close: ArrayLike) -> ArrayLike:
    """
    True Range.
    
    TR = max(High - Low, abs(High - PrevClose), abs(Low - PrevClose))
    """
    hl = high - low
    hc = np.abs(high - prev_close)
    lc = np.abs(low - prev_close)
    
    if isinstance(high, pd.Series):
        return pd.concat([hl, hc, lc], axis=1).max(axis=1)
    else:
        return np.maximum(hl, np.maximum(hc, lc))


@jit(nopython=True)
def _atr_calc(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    """Numba-optimized ATR calculation using Wilder's smoothing."""
    length = len(close)
    tr = np.zeros(length)
    atr = np.zeros(length)
    
    # Calculate True Range
    for i in range(1, length):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    # Initialize ATR
    atr[n] = np.mean(tr[1:n+1])
    
    # Wilder's smoothing
    for i in range(n+1, length):
        atr[i] = (atr[i-1] * (n-1) + tr[i]) / n
    
    # Fill initial values with NaN
    atr[:n] = np.nan
    
    return atr


def atr(high: ArrayLike, low: ArrayLike, close: ArrayLike, n: int = 14) -> ArrayLike:
    """
    Average True Range.
    
    Wilder's smoothed average of True Range.
    """
    if isinstance(high, pd.Series):
        values = _atr_calc(high.values, low.values, close.values, n)
        return pd.Series(values, index=high.index)
    else:
        return _atr_calc(np.asarray(high), np.asarray(low), np.asarray(close), n)


def natr(high: ArrayLike, low: ArrayLike, close: ArrayLike, n: int = 14) -> ArrayLike:
    """
    Normalized Average True Range.
    
    NATR = 100 * ATR / Close
    """
    atr_val = atr(high, low, close, n)
    return 100 * atr_val / close


# ============================================================================
# Historical Volatility
# ============================================================================

def historical_vol(close: ArrayLike, n: int = 20, periods_per_year: int = 252, 
                   method: str = "log") -> ArrayLike:
    """
    Historical volatility from close prices.
    
    Annualized standard deviation of returns.
    """
    from .core import close_to_close_ret
    
    returns = close_to_close_ret(close, method=method)
    vol = rolling_std(returns, n)
    
    # Annualize
    return vol * np.sqrt(periods_per_year)


def historical_vol_from_close(close: ArrayLike, n: int = 20, method: str = "log") -> ArrayLike:
    """Historical volatility (non-annualized)."""
    from .core import close_to_close_ret
    
    returns = close_to_close_ret(close, method=method)
    return rolling_std(returns, n)


# ============================================================================
# Range-Based Volatility Estimators
# ============================================================================

def parkinson_vol(high: ArrayLike, low: ArrayLike, n: int = 20, 
                  periods_per_year: int = 252) -> ArrayLike:
    """
    Parkinson volatility estimator.
    
    Uses high-low range, more efficient than close-to-close.
    """
    hl_ratio = np.log(high / low)
    
    # Parkinson constant
    constant = 1.0 / (4.0 * np.log(2.0))
    
    if isinstance(high, pd.Series):
        vol = np.sqrt(constant * hl_ratio.pow(2).rolling(n).mean())
    else:
        hl_ratio_sq = hl_ratio ** 2
        vol = np.sqrt(constant * rolling_mean(hl_ratio_sq, n))
    
    # Annualize
    return vol * np.sqrt(periods_per_year)


def garman_klass_vol(open: ArrayLike, high: ArrayLike, low: ArrayLike, 
                     close: ArrayLike, n: int = 20, 
                     periods_per_year: int = 252) -> ArrayLike:
    """
    Garman-Klass volatility estimator.
    
    Incorporates OHLC information for better efficiency.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open)
    
    rs = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
    
    if isinstance(high, pd.Series):
        vol = np.sqrt(rs.rolling(n).mean())
    else:
        vol = np.sqrt(rolling_mean(rs, n))
    
    # Annualize
    return vol * np.sqrt(periods_per_year)


def rogers_satchell_vol(open: ArrayLike, high: ArrayLike, low: ArrayLike,
                        close: ArrayLike, n: int = 20,
                        periods_per_year: int = 252) -> ArrayLike:
    """
    Rogers-Satchell volatility estimator.
    
    Handles non-zero drift better than Garman-Klass.
    """
    log_ho = np.log(high / open)
    log_hc = np.log(high / close)
    log_lo = np.log(low / open)
    log_lc = np.log(low / close)
    
    rs = log_ho * log_hc + log_lo * log_lc
    
    if isinstance(high, pd.Series):
        vol = np.sqrt(rs.rolling(n).mean())
    else:
        vol = np.sqrt(rolling_mean(rs, n))
    
    # Annualize
    return vol * np.sqrt(periods_per_year)


def yang_zhang_vol(open: ArrayLike, high: ArrayLike, low: ArrayLike,
                   close: ArrayLike, n: int = 20,
                   periods_per_year: int = 252) -> ArrayLike:
    """
    Yang-Zhang volatility estimator.
    
    Most efficient estimator, combines overnight and intraday components.
    """
    if isinstance(close, pd.Series):
        prev_close = close.shift(1)
    else:
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
    
    # Overnight volatility
    log_oc = np.log(open / prev_close)
    overnight_var = rolling_mean(log_oc**2, n) - rolling_mean(log_oc, n)**2
    
    # Open-to-close volatility  
    log_co = np.log(close / open)
    open_to_close_var = rolling_mean(log_co**2, n) - rolling_mean(log_co, n)**2
    
    # Rogers-Satchell volatility
    rs_vol = rogers_satchell_vol(open, high, low, close, n, 1)  # Don't annualize yet
    
    # Combine components
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    
    vol = np.sqrt(overnight_var + k * open_to_close_var + (1 - k) * rs_vol**2)
    
    # Annualize
    return vol * np.sqrt(periods_per_year)


# ============================================================================
# Realized Volatility
# ============================================================================

def realized_var_from_close(close: ArrayLike, n: int = 20) -> ArrayLike:
    """
    Realized variance from close prices.
    
    Sum of squared returns over n periods.
    """
    from .core import close_to_close_ret
    
    returns = close_to_close_ret(close, method="log")
    
    if isinstance(returns, pd.Series):
        return returns.pow(2).rolling(n).sum()
    else:
        return rolling_sum(returns**2, n)


def realized_vol_from_close(close: ArrayLike, n: int = 20, 
                            periods_per_year: int = 252) -> ArrayLike:
    """
    Realized volatility from close prices.
    
    Square root of realized variance, annualized.
    """
    var = realized_var_from_close(close, n)
    return np.sqrt(var * periods_per_year / n)


# ============================================================================
# Other Volatility Measures
# ============================================================================

def rolling_hl_range(high: ArrayLike, low: ArrayLike, n: int = 20) -> ArrayLike:
    """
    Rolling high-low range.
    
    Max(High) - Min(Low) over n periods.
    """
    return rolling_max(high, n) - rolling_min(low, n)


def choppiness_index(high: ArrayLike, low: ArrayLike, close: ArrayLike, n: int = 14) -> ArrayLike:
    """
    Choppiness Index.
    
    Measures market choppiness vs trending (0-100).
    100 = choppy, 0 = trending.
    """
    atr_sum = rolling_sum(atr(high, low, close, 1), n)
    hl_range = rolling_max(high, n) - rolling_min(low, n)
    
    ci = 100 * np.log10(atr_sum / hl_range) / np.log10(n)
    
    return ci


def ulcer_index(close: ArrayLike, n: int = 14) -> ArrayLike:
    """
    Ulcer Index.
    
    Measures downside volatility/drawdown risk.
    """
    # Calculate rolling maximum
    roll_max = rolling_max(close, n)
    
    # Percentage drawdown
    dd_pct = 100 * (close - roll_max) / roll_max
    
    # Square and average
    if isinstance(close, pd.Series):
        ui = np.sqrt((dd_pct**2).rolling(n).mean())
    else:
        ui = np.sqrt(rolling_mean(dd_pct**2, n))
    
    return ui


# ============================================================================
# Bands and Channels
# ============================================================================

def bollinger_bands(close: ArrayLike, n: int = 20, k: float = 2.0) -> dict:
    """
    Bollinger Bands.
    
    Returns:
    --------
    dict with keys:
        - upper: Upper band
        - middle: Middle band (SMA)
        - lower: Lower band
        - width: Band width
        - pctb: Percent B (position within bands)
    """
    middle = rolling_mean(close, n)
    std = rolling_std(close, n)
    
    upper = middle + k * std
    lower = middle - k * std
    width = upper - lower
    pctb = (close - lower) / width
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'width': width,
        'pctb': pctb
    }


def keltner_channels(high: ArrayLike, low: ArrayLike, close: ArrayLike,
                     ema_n: int = 20, atr_n: int = 10, k: float = 2.0) -> dict:
    """
    Keltner Channels.
    
    Returns:
    --------
    dict with keys:
        - upper: Upper channel
        - middle: Middle line (EMA)
        - lower: Lower channel
        - width: Channel width
    """
    from .trend import ema
    
    middle = ema(close, ema_n)
    atr_val = atr(high, low, close, atr_n)
    
    upper = middle + k * atr_val
    lower = middle - k * atr_val
    width = upper - lower
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'width': width
    }


def donchian_channels(high: ArrayLike, low: ArrayLike, n: int = 20) -> dict:
    """
    Donchian Channels.
    
    Returns:
    --------
    dict with keys:
        - upper: Upper channel (highest high)
        - lower: Lower channel (lowest low)
        - middle: Middle line
        - width: Channel width
    """
    upper = rolling_max(high, n)
    lower = rolling_min(low, n)
    middle = (upper + lower) / 2
    width = upper - lower
    
    return {
        'upper': upper,
        'lower': lower,
        'middle': middle,
        'width': width
    }


def chandelier_exit(high: ArrayLike, low: ArrayLike, close: ArrayLike,
                    atr_n: int = 22, k: float = 3.0) -> dict:
    """
    Chandelier Exit.
    
    Trailing stop based on ATR.
    
    Returns:
    --------
    dict with keys:
        - long_stop: Stop for long positions
        - short_stop: Stop for short positions
    """
    atr_val = atr(high, low, close, atr_n)
    
    highest = rolling_max(high, atr_n)
    lowest = rolling_min(low, atr_n)
    
    long_stop = highest - k * atr_val
    short_stop = lowest + k * atr_val
    
    return {
        'long_stop': long_stop,
        'short_stop': short_stop
    }


# ============================================================================
# Position Sizing and Stops
# ============================================================================

def atr_stop(entry_price: float, atr_val: ArrayLike, k: float = 2.0, 
             side: str = "long") -> ArrayLike:
    """
    ATR-based stop loss.
    
    Parameters:
    -----------
    entry_price : float
        Entry price for the position
    atr_val : ArrayLike
        ATR values
    k : float
        ATR multiplier
    side : str
        "long" or "short"
    """
    if side == "long":
        return entry_price - k * atr_val
    else:
        return entry_price + k * atr_val


def chandelier_stop(high: ArrayLike, low: ArrayLike, close: ArrayLike,
                    atr_val: ArrayLike, k: float = 3.0, side: str = "long") -> ArrayLike:
    """
    Chandelier trailing stop.
    
    Trails from highest high (long) or lowest low (short).
    """
    if side == "long":
        highest = rolling_max(high, len(atr_val))
        return highest - k * atr_val
    else:
        lowest = rolling_min(low, len(atr_val))
        return lowest + k * atr_val


def trailing_stop_pct(close: ArrayLike, trail_pct: float, side: str = "long") -> ArrayLike:
    """
    Percentage-based trailing stop.
    
    Parameters:
    -----------
    close : ArrayLike
        Price series
    trail_pct : float
        Trail percentage (e.g., 0.05 for 5%)
    side : str
        "long" or "short"
    """
    if side == "long":
        if isinstance(close, pd.Series):
            return close.expanding().max() * (1 - trail_pct)
        else:
            return pd.Series(close).expanding().max().values * (1 - trail_pct)
    else:
        if isinstance(close, pd.Series):
            return close.expanding().min() * (1 + trail_pct)
        else:
            return pd.Series(close).expanding().min().values * (1 + trail_pct)