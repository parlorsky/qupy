"""
Core OHLCV transforms and composites.

Fast, vectorized implementations of fundamental price transformations.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple

# Optional numba import for performance
try:
    from numba import jit, float64, int64
    HAS_NUMBA = True
except (ImportError, SystemError):
    HAS_NUMBA = False
    # Create dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# Type aliases
ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]
SeriesLike = Union[np.ndarray, pd.Series]


# ============================================================================
# OHLCV Composites
# ============================================================================

def hl2(high: ArrayLike, low: ArrayLike) -> ArrayLike:
    """Median price (high + low) / 2."""
    return (high + low) / 2


def hlc3(high: ArrayLike, low: ArrayLike, close: ArrayLike) -> ArrayLike:
    """Typical price (high + low + close) / 3."""
    return (high + low + close) / 3


def ohlc4(open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike) -> ArrayLike:
    """Average price (open + high + low + close) / 4."""
    return (open + high + low + close) / 4


def tp(high: ArrayLike, low: ArrayLike, close: ArrayLike) -> ArrayLike:
    """Typical price alias for hlc3."""
    return hlc3(high, low, close)


def median_price(high: ArrayLike, low: ArrayLike) -> ArrayLike:
    """Median price alias for hl2."""
    return hl2(high, low)


def log_price(close: ArrayLike) -> ArrayLike:
    """Natural logarithm of price."""
    return np.log(close)


# ============================================================================
# Basic Transforms
# ============================================================================

def lag(x: ArrayLike, k: int = 1) -> ArrayLike:
    """Lag series by k periods."""
    if isinstance(x, pd.Series):
        return x.shift(k)
    elif isinstance(x, pd.DataFrame):
        return x.shift(k)
    else:
        result = np.empty_like(x)
        result[:k] = np.nan
        result[k:] = x[:-k]
        return result


def diff(x: ArrayLike, k: int = 1) -> ArrayLike:
    """Difference of series with k-period lag."""
    if isinstance(x, pd.Series):
        return x.diff(k)
    elif isinstance(x, pd.DataFrame):
        return x.diff(k)
    else:
        return x - lag(x, k)


def pct_change(x: ArrayLike, k: int = 1, log: bool = False) -> ArrayLike:
    """Percentage change over k periods."""
    if log:
        return diff(np.log(x), k)
    else:
        if isinstance(x, (pd.Series, pd.DataFrame)):
            return x.pct_change(k)
        else:
            lagged = lag(x, k)
            return (x - lagged) / lagged


# ============================================================================
# Rolling Statistics (Vectorized)
# ============================================================================

@jit(nopython=True)
def _rolling_window_1d(arr: np.ndarray, window: int, min_periods: int) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to get rolling window indices and valid mask."""
    n = len(arr)
    valid = np.zeros(n, dtype=np.bool_)
    
    for i in range(n):
        start = max(0, i - window + 1)
        end = i + 1
        if end - start >= min_periods:
            valid[i] = True
    
    return valid


def rolling_sum(x: ArrayLike, n: int, min_periods: Optional[int] = None) -> ArrayLike:
    """Rolling sum over n periods."""
    min_periods = min_periods or n
    
    if isinstance(x, pd.Series):
        return x.rolling(n, min_periods=min_periods).sum()
    elif isinstance(x, pd.DataFrame):
        return x.rolling(n, min_periods=min_periods).sum()
    else:
        return pd.Series(x).rolling(n, min_periods=min_periods).sum().values


def rolling_mean(x: ArrayLike, n: int, min_periods: Optional[int] = None) -> ArrayLike:
    """Rolling mean over n periods."""
    min_periods = min_periods or n
    
    if isinstance(x, pd.Series):
        return x.rolling(n, min_periods=min_periods).mean()
    elif isinstance(x, pd.DataFrame):
        return x.rolling(n, min_periods=min_periods).mean()
    else:
        return pd.Series(x).rolling(n, min_periods=min_periods).mean().values


def rolling_std(x: ArrayLike, n: int, min_periods: Optional[int] = None, ddof: int = 1) -> ArrayLike:
    """Rolling standard deviation over n periods."""
    min_periods = min_periods or n
    
    if isinstance(x, pd.Series):
        return x.rolling(n, min_periods=min_periods).std(ddof=ddof)
    elif isinstance(x, pd.DataFrame):
        return x.rolling(n, min_periods=min_periods).std(ddof=ddof)
    else:
        return pd.Series(x).rolling(n, min_periods=min_periods).std(ddof=ddof).values


def rolling_min(x: ArrayLike, n: int, min_periods: Optional[int] = None) -> ArrayLike:
    """Rolling minimum over n periods."""
    min_periods = min_periods or n
    
    if isinstance(x, pd.Series):
        return x.rolling(n, min_periods=min_periods).min()
    elif isinstance(x, pd.DataFrame):
        return x.rolling(n, min_periods=min_periods).min()
    else:
        return pd.Series(x).rolling(n, min_periods=min_periods).min().values


def rolling_max(x: ArrayLike, n: int, min_periods: Optional[int] = None) -> ArrayLike:
    """Rolling maximum over n periods."""
    min_periods = min_periods or n
    
    if isinstance(x, pd.Series):
        return x.rolling(n, min_periods=min_periods).max()
    elif isinstance(x, pd.DataFrame):
        return x.rolling(n, min_periods=min_periods).max()
    else:
        return pd.Series(x).rolling(n, min_periods=min_periods).max().values


def rolling_median(x: ArrayLike, n: int, min_periods: Optional[int] = None) -> ArrayLike:
    """Rolling median over n periods."""
    min_periods = min_periods or n
    
    if isinstance(x, pd.Series):
        return x.rolling(n, min_periods=min_periods).median()
    elif isinstance(x, pd.DataFrame):
        return x.rolling(n, min_periods=min_periods).median()
    else:
        return pd.Series(x).rolling(n, min_periods=min_periods).median().values


def rolling_quantile(x: ArrayLike, n: int, q: float, min_periods: Optional[int] = None) -> ArrayLike:
    """Rolling quantile over n periods."""
    min_periods = min_periods or n
    
    if isinstance(x, pd.Series):
        return x.rolling(n, min_periods=min_periods).quantile(q)
    elif isinstance(x, pd.DataFrame):
        return x.rolling(n, min_periods=min_periods).quantile(q)
    else:
        return pd.Series(x).rolling(n, min_periods=min_periods).quantile(q).values


def rolling_corr(x: ArrayLike, y: ArrayLike, n: int, min_periods: Optional[int] = None) -> ArrayLike:
    """Rolling correlation between two series."""
    min_periods = min_periods or n
    
    if isinstance(x, pd.Series) and isinstance(y, pd.Series):
        return x.rolling(n, min_periods=min_periods).corr(y)
    else:
        x_series = pd.Series(x) if not isinstance(x, pd.Series) else x
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        return x_series.rolling(n, min_periods=min_periods).corr(y_series).values


def rolling_cov(x: ArrayLike, y: ArrayLike, n: int, min_periods: Optional[int] = None) -> ArrayLike:
    """Rolling covariance between two series."""
    min_periods = min_periods or n
    
    if isinstance(x, pd.Series) and isinstance(y, pd.Series):
        return x.rolling(n, min_periods=min_periods).cov(y)
    else:
        x_series = pd.Series(x) if not isinstance(x, pd.Series) else x
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        return x_series.rolling(n, min_periods=min_periods).cov(y_series).values


# ============================================================================
# Normalization and Scoring
# ============================================================================

def zscore(x: ArrayLike, n: int, min_periods: Optional[int] = None) -> ArrayLike:
    """Rolling Z-score normalization."""
    mean = rolling_mean(x, n, min_periods)
    std = rolling_std(x, n, min_periods)
    return (x - mean) / std


def robust_zscore_mad(x: ArrayLike, n: int, min_periods: Optional[int] = None) -> ArrayLike:
    """Robust Z-score using median absolute deviation."""
    min_periods = min_periods or n
    median = rolling_median(x, n, min_periods)
    
    # Calculate MAD
    if isinstance(x, pd.Series):
        mad = (x - median).abs().rolling(n, min_periods=min_periods).median()
    else:
        x_series = pd.Series(x)
        median_series = pd.Series(median)
        mad = (x_series - median_series).abs().rolling(n, min_periods=min_periods).median().values
    
    # Scale factor for normal distribution
    mad_scaled = mad * 1.4826
    return (x - median) / mad_scaled


# ============================================================================
# Return Series
# ============================================================================

def close_to_close_ret(close: ArrayLike, method: str = "log") -> ArrayLike:
    """Close-to-close returns."""
    if method == "log":
        return diff(np.log(close), 1)
    else:
        return pct_change(close, 1)


def open_to_close_ret(open: ArrayLike, close: ArrayLike, method: str = "log") -> ArrayLike:
    """Intraday returns from open to close."""
    if method == "log":
        return np.log(close) - np.log(open)
    else:
        return (close - open) / open


def close_to_open_ret(prev_close: ArrayLike, open: ArrayLike, method: str = "log") -> ArrayLike:
    """Overnight/gap returns from previous close to current open."""
    if method == "log":
        return np.log(open) - np.log(prev_close)
    else:
        return (open - prev_close) / prev_close


def cumulative_return(returns: ArrayLike, method: str = "simple") -> ArrayLike:
    """Cumulative returns from period returns."""
    if method == "log":
        if isinstance(returns, pd.Series):
            return returns.cumsum()
        else:
            return np.nancumsum(returns)
    else:
        if isinstance(returns, pd.Series):
            return (1 + returns).cumprod() - 1
        else:
            return np.nancumprod(1 + returns) - 1


def rolling_return(close: ArrayLike, n: int, method: str = "log") -> ArrayLike:
    """Rolling n-period returns."""
    if method == "log":
        return diff(np.log(close), n)
    else:
        return pct_change(close, n)


def gap_pct(open: ArrayLike, prev_close: ArrayLike) -> ArrayLike:
    """Gap percentage from previous close to current open."""
    return (open - prev_close) / prev_close


def overnight_return(prev_close: ArrayLike, open: ArrayLike) -> ArrayLike:
    """Overnight returns (alias for close_to_open_ret)."""
    return close_to_open_ret(prev_close, open, method="simple")


def intraday_return(open: ArrayLike, close: ArrayLike) -> ArrayLike:
    """Intraday returns (alias for open_to_close_ret)."""
    return open_to_close_ret(open, close, method="simple")


def high_low_intrabar_range_pct(high: ArrayLike, low: ArrayLike) -> ArrayLike:
    """Intrabar range as percentage of low price."""
    return (high - low) / low


# ============================================================================
# Resampling
# ============================================================================

def resample_ohlcv(df: pd.DataFrame, rule: str, 
                   columns: Optional[dict] = None) -> pd.DataFrame:
    """
    Resample OHLCV data to different timeframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index and OHLCV columns
    rule : str
        Pandas resampling rule (e.g., '5T', '1H', '1D')
    columns : dict, optional
        Column name mapping, default: {'open': 'open', 'high': 'high', ...}
    
    Returns:
    --------
    pd.DataFrame
        Resampled OHLCV data
    """
    columns = columns or {
        'open': 'open',
        'high': 'high', 
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    }
    
    agg_dict = {
        columns['open']: 'first',
        columns['high']: 'max',
        columns['low']: 'min',
        columns['close']: 'last',
        columns['volume']: 'sum'
    }
    
    return df.resample(rule).agg(agg_dict)


# ============================================================================
# Signal Processing
# ============================================================================

def signal_smooth(signal: ArrayLike, method: str = "ema", n: int = 3) -> ArrayLike:
    """Smooth a signal using moving average."""
    if method == "ema":
        from .trend import ema
        return ema(signal, n)
    else:
        return rolling_mean(signal, n)


def signal_clip(signal: ArrayLike, lo: float, hi: float) -> ArrayLike:
    """Clip signal values to range [lo, hi]."""
    if isinstance(signal, pd.Series):
        return signal.clip(lo, hi)
    else:
        return np.clip(signal, lo, hi)


def signal_winsorize(signal: ArrayLike, p: float = 0.01) -> ArrayLike:
    """Winsorize signal at p and 1-p percentiles."""
    if isinstance(signal, pd.Series):
        lower = signal.quantile(p)
        upper = signal.quantile(1 - p)
        return signal.clip(lower, upper)
    else:
        lower = np.nanpercentile(signal, p * 100)
        upper = np.nanpercentile(signal, (1 - p) * 100)
        return np.clip(signal, lower, upper)


def signal_zscore_normalize(signal: ArrayLike, n: Optional[int] = None) -> ArrayLike:
    """Z-score normalize signal (full sample or rolling)."""
    if n is None:
        # Full sample normalization
        if isinstance(signal, pd.Series):
            return (signal - signal.mean()) / signal.std()
        else:
            return (signal - np.nanmean(signal)) / np.nanstd(signal)
    else:
        # Rolling normalization
        return zscore(signal, n)