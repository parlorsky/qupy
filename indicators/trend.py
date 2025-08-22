"""
Trend indicators and moving averages.

Optimized implementations of trend-following indicators.
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

from .core import ArrayLike, rolling_mean, rolling_max, rolling_min, diff


# ============================================================================
# Moving Averages
# ============================================================================

def sma(close: ArrayLike, n: int, min_periods: Optional[int] = None) -> ArrayLike:
    """Simple Moving Average."""
    return rolling_mean(close, n, min_periods)


def ema(close: ArrayLike, n: int, adjust: bool = False) -> ArrayLike:
    """
    Exponential Moving Average.
    
    Parameters:
    -----------
    close : ArrayLike
        Price series
    n : int
        Period for EMA
    adjust : bool
        Use adjusted calculation (default: False)
    """
    if isinstance(close, pd.Series):
        return close.ewm(span=n, adjust=adjust).mean()
    else:
        close_series = pd.Series(close)
        return close_series.ewm(span=n, adjust=adjust).mean().values


def wma(close: ArrayLike, n: int) -> ArrayLike:
    """
    Weighted Moving Average.
    
    Weights increase linearly from oldest to newest.
    """
    weights = np.arange(1, n + 1)
    
    if isinstance(close, pd.Series):
        return close.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    else:
        close_series = pd.Series(close)
        return close_series.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).values


def hma(close: ArrayLike, n: int) -> ArrayLike:
    """
    Hull Moving Average.
    
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    half_n = n // 2
    sqrt_n = int(np.sqrt(n))
    
    wma_half = wma(close, half_n)
    wma_full = wma(close, n)
    
    raw_hma = 2 * wma_half - wma_full
    return wma(raw_hma, sqrt_n)


def dema(close: ArrayLike, n: int) -> ArrayLike:
    """
    Double Exponential Moving Average.
    
    DEMA = 2 * EMA(n) - EMA(EMA(n), n)
    """
    ema1 = ema(close, n)
    ema2 = ema(ema1, n)
    return 2 * ema1 - ema2


def tema(close: ArrayLike, n: int) -> ArrayLike:
    """
    Triple Exponential Moving Average.
    
    TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    """
    ema1 = ema(close, n)
    ema2 = ema(ema1, n)
    ema3 = ema(ema2, n)
    return 3 * ema1 - 3 * ema2 + ema3


def zlema(close: ArrayLike, n: int) -> ArrayLike:
    """
    Zero-Lag Exponential Moving Average.
    
    Uses a de-lagging technique before applying EMA.
    """
    lag_period = (n - 1) // 2
    
    if isinstance(close, pd.Series):
        lagged = close.shift(lag_period)
        de_lagged = 2 * close - lagged
    else:
        close_series = pd.Series(close)
        lagged = close_series.shift(lag_period)
        de_lagged = 2 * close_series - lagged
        de_lagged = de_lagged.values
    
    return ema(de_lagged, n)


@jit(nopython=True)
def _kama_calc(close: np.ndarray, n: int, fast: int = 2, slow: int = 30) -> np.ndarray:
    """Numba-optimized KAMA calculation."""
    result = np.empty_like(close)
    result[:n] = np.nan
    
    fastest_sc = 2.0 / (fast + 1.0)
    slowest_sc = 2.0 / (slow + 1.0)
    
    result[n-1] = close[n-1]
    
    for i in range(n, len(close)):
        # Calculate efficiency ratio
        direction = abs(close[i] - close[i-n])
        volatility = 0.0
        for j in range(1, n+1):
            volatility += abs(close[i-j+1] - close[i-j])
        
        if volatility != 0:
            er = direction / volatility
        else:
            er = 0.0
        
        # Calculate smoothing constant
        sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
        
        # Calculate KAMA
        result[i] = result[i-1] + sc * (close[i] - result[i-1])
    
    return result


def kama(close: ArrayLike, n: int, fast: int = 2, slow: int = 30) -> ArrayLike:
    """
    Kaufman Adaptive Moving Average.
    
    Adapts between fast and slow EMAs based on market efficiency.
    """
    if isinstance(close, pd.Series):
        values = close.values
        result = _kama_calc(values, n, fast, slow)
        return pd.Series(result, index=close.index)
    else:
        return _kama_calc(np.asarray(close), n, fast, slow)


def t3(close: ArrayLike, n: int, vfactor: float = 0.7) -> ArrayLike:
    """
    T3 Moving Average (Tillson).
    
    Triple-smoothed EMA with volume factor.
    """
    alpha = 2.0 / (n + 1.0)
    
    c1 = -vfactor ** 3
    c2 = 3 * vfactor ** 2 + 3 * vfactor ** 3
    c3 = -6 * vfactor ** 2 - 3 * vfactor - 3 * vfactor ** 3
    c4 = 1 + 3 * vfactor + vfactor ** 3 + 3 * vfactor ** 2
    
    ema1 = ema(close, n)
    ema2 = ema(ema1, n)
    ema3 = ema(ema2, n)
    ema4 = ema(ema3, n)
    ema5 = ema(ema4, n)
    ema6 = ema(ema5, n)
    
    return c1 * ema6 + c2 * ema5 + c3 * ema4 + c4 * ema3


# ============================================================================
# Trend Indicators
# ============================================================================

@jit(nopython=True)
def _adx_calc(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate ADX, +DI, -DI using Numba."""
    length = len(close)
    
    # Initialize arrays
    tr = np.zeros(length)
    dm_plus = np.zeros(length)
    dm_minus = np.zeros(length)
    
    # Calculate True Range and Directional Movement
    for i in range(1, length):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
        
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        
        if up > down and up > 0:
            dm_plus[i] = up
        if down > up and down > 0:
            dm_minus[i] = down
    
    # Smooth using Wilder's method
    atr = np.zeros(length)
    adm_plus = np.zeros(length)
    adm_minus = np.zeros(length)
    
    # Initialize
    atr[n] = np.mean(tr[1:n+1])
    adm_plus[n] = np.mean(dm_plus[1:n+1])
    adm_minus[n] = np.mean(dm_minus[1:n+1])
    
    # Wilder's smoothing
    for i in range(n+1, length):
        atr[i] = (atr[i-1] * (n-1) + tr[i]) / n
        adm_plus[i] = (adm_plus[i-1] * (n-1) + dm_plus[i]) / n
        adm_minus[i] = (adm_minus[i-1] * (n-1) + dm_minus[i]) / n
    
    # Calculate DI
    di_plus = np.zeros(length)
    di_minus = np.zeros(length)
    dx = np.zeros(length)
    
    for i in range(n, length):
        if atr[i] != 0:
            di_plus[i] = 100 * adm_plus[i] / atr[i]
            di_minus[i] = 100 * adm_minus[i] / atr[i]
            
            di_sum = di_plus[i] + di_minus[i]
            if di_sum != 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    # Calculate ADX
    adx = np.zeros(length)
    adx[2*n-1] = np.mean(dx[n:2*n])
    
    for i in range(2*n, length):
        adx[i] = (adx[i-1] * (n-1) + dx[i]) / n
    
    return adx, di_plus, di_minus


def adx(high: ArrayLike, low: ArrayLike, close: ArrayLike, n: int = 14) -> ArrayLike:
    """Average Directional Index."""
    if isinstance(high, pd.Series):
        adx_vals, _, _ = _adx_calc(high.values, low.values, close.values, n)
        return pd.Series(adx_vals, index=high.index)
    else:
        adx_vals, _, _ = _adx_calc(np.asarray(high), np.asarray(low), np.asarray(close), n)
        return adx_vals


def di_plus(high: ArrayLike, low: ArrayLike, close: ArrayLike, n: int = 14) -> ArrayLike:
    """Positive Directional Indicator."""
    if isinstance(high, pd.Series):
        _, di_p, _ = _adx_calc(high.values, low.values, close.values, n)
        return pd.Series(di_p, index=high.index)
    else:
        _, di_p, _ = _adx_calc(np.asarray(high), np.asarray(low), np.asarray(close), n)
        return di_p


def di_minus(high: ArrayLike, low: ArrayLike, close: ArrayLike, n: int = 14) -> ArrayLike:
    """Negative Directional Indicator."""
    if isinstance(high, pd.Series):
        _, _, di_m = _adx_calc(high.values, low.values, close.values, n)
        return pd.Series(di_m, index=high.index)
    else:
        _, _, di_m = _adx_calc(np.asarray(high), np.asarray(low), np.asarray(close), n)
        return di_m


def aroon_up(high: ArrayLike, low: ArrayLike, n: int = 25) -> ArrayLike:
    """Aroon Up indicator."""
    if isinstance(high, pd.Series):
        return high.rolling(n + 1).apply(lambda x: 100 * (n - x.argmax()) / n, raw=True)
    else:
        high_series = pd.Series(high)
        return high_series.rolling(n + 1).apply(lambda x: 100 * (n - x.argmax()) / n, raw=True).values


def aroon_down(high: ArrayLike, low: ArrayLike, n: int = 25) -> ArrayLike:
    """Aroon Down indicator."""
    if isinstance(low, pd.Series):
        return low.rolling(n + 1).apply(lambda x: 100 * (n - x.argmin()) / n, raw=True)
    else:
        low_series = pd.Series(low)
        return low_series.rolling(n + 1).apply(lambda x: 100 * (n - x.argmin()) / n, raw=True).values


def aroon_oscillator(high: ArrayLike, low: ArrayLike, n: int = 25) -> ArrayLike:
    """Aroon Oscillator."""
    return aroon_up(high, low, n) - aroon_down(high, low, n)


def supertrend(high: ArrayLike, low: ArrayLike, close: ArrayLike, 
               atr_n: int = 10, multiplier: float = 3.0) -> Tuple[ArrayLike, ArrayLike]:
    """
    Supertrend indicator.
    
    Returns:
    --------
    supertrend : ArrayLike
        Supertrend line
    direction : ArrayLike
        Direction (1 for uptrend, -1 for downtrend)
    """
    from .volatility import atr
    from .core import hl2
    
    atr_val = atr(high, low, close, atr_n)
    hl_avg = hl2(high, low)
    
    # Calculate basic bands
    upper_band = hl_avg + multiplier * atr_val
    lower_band = hl_avg - multiplier * atr_val
    
    # Initialize arrays
    if isinstance(close, pd.Series):
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=float)
        
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(close)):
            if close.iloc[i] <= upper_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
            
            if supertrend.iloc[i] <= close.iloc[i]:
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = -1
            
            # Adjust based on previous direction
            if direction.iloc[i] == direction.iloc[i-1]:
                if direction.iloc[i] == 1:
                    supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
                else:
                    supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
    else:
        close_arr = np.asarray(close)
        upper_arr = np.asarray(upper_band)
        lower_arr = np.asarray(lower_band)
        
        supertrend = np.zeros_like(close_arr)
        direction = np.zeros_like(close_arr)
        
        supertrend[0] = upper_arr[0]
        direction[0] = 1
        
        for i in range(1, len(close_arr)):
            if close_arr[i] <= upper_arr[i]:
                supertrend[i] = upper_arr[i]
            else:
                supertrend[i] = lower_arr[i]
            
            if supertrend[i] <= close_arr[i]:
                direction[i] = 1
            else:
                direction[i] = -1
            
            if direction[i] == direction[i-1]:
                if direction[i] == 1:
                    supertrend[i] = max(lower_arr[i], supertrend[i-1])
                else:
                    supertrend[i] = min(upper_arr[i], supertrend[i-1])
    
    return supertrend, direction


def ichimoku(high: ArrayLike, low: ArrayLike, close: ArrayLike,
             conv: int = 9, base: int = 26, span: int = 52) -> dict:
    """
    Ichimoku Cloud indicator.
    
    Returns:
    --------
    dict with keys:
        - tenkan: Conversion line
        - kijun: Base line
        - senkou_a: Leading span A
        - senkou_b: Leading span B
        - chikou: Lagging span
    """
    # Tenkan-sen (Conversion Line)
    tenkan = (rolling_max(high, conv) + rolling_min(low, conv)) / 2
    
    # Kijun-sen (Base Line)
    kijun = (rolling_max(high, base) + rolling_min(low, base)) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B)
    senkou_b = (rolling_max(high, span) + rolling_min(low, span)) / 2
    
    # Chikou Span (Lagging Span)
    if isinstance(close, pd.Series):
        chikou = close.shift(-base)
        senkou_a = senkou_a.shift(base)
        senkou_b = senkou_b.shift(base)
    else:
        chikou = np.roll(close, -base)
        senkou_a = np.roll(senkou_a, base)
        senkou_b = np.roll(senkou_b, base)
    
    return {
        'tenkan': tenkan,
        'kijun': kijun,
        'senkou_a': senkou_a,
        'senkou_b': senkou_b,
        'chikou': chikou
    }


def moving_average_crossover(fast_ma: ArrayLike, slow_ma: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """
    Detect moving average crossovers.
    
    Returns:
    --------
    bullish_cross : ArrayLike
        True where fast crosses above slow
    bearish_cross : ArrayLike
        True where fast crosses below slow
    """
    if isinstance(fast_ma, pd.Series) and isinstance(slow_ma, pd.Series):
        fast_above = fast_ma > slow_ma
        bullish = fast_above & ~fast_above.shift(1)
        bearish = ~fast_above & fast_above.shift(1)
    else:
        fast_ma = np.asarray(fast_ma)
        slow_ma = np.asarray(slow_ma)
        fast_above = fast_ma > slow_ma
        bullish = fast_above[1:] & ~fast_above[:-1]
        bearish = ~fast_above[1:] & fast_above[:-1]
        
        # Pad to match original length
        bullish = np.concatenate([[False], bullish])
        bearish = np.concatenate([[False], bearish])
    
    return bullish, bearish


def price_above_ma(close: ArrayLike, ma: ArrayLike) -> ArrayLike:
    """Check if price is above moving average."""
    return close > ma


def price_below_ma(close: ArrayLike, ma: ArrayLike) -> ArrayLike:
    """Check if price is below moving average."""
    return close < ma


def rolling_linreg_slope(close: ArrayLike, n: int) -> ArrayLike:
    """Rolling linear regression slope."""
    if isinstance(close, pd.Series):
        x = np.arange(n)
        return close.rolling(n).apply(
            lambda y: np.polyfit(x, y, 1)[0] if len(y) == n else np.nan,
            raw=True
        )
    else:
        close_series = pd.Series(close)
        x = np.arange(n)
        return close_series.rolling(n).apply(
            lambda y: np.polyfit(x, y, 1)[0] if len(y) == n else np.nan,
            raw=True
        ).values


def rolling_linreg_intercept(close: ArrayLike, n: int) -> ArrayLike:
    """Rolling linear regression intercept."""
    if isinstance(close, pd.Series):
        x = np.arange(n)
        return close.rolling(n).apply(
            lambda y: np.polyfit(x, y, 1)[1] if len(y) == n else np.nan,
            raw=True
        )
    else:
        close_series = pd.Series(close)
        x = np.arange(n)
        return close_series.rolling(n).apply(
            lambda y: np.polyfit(x, y, 1)[1] if len(y) == n else np.nan,
            raw=True
        ).values