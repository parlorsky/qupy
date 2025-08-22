"""
Technical indicators library for qupy backtesting framework.

Optimized, vectorized implementations of technical indicators using NumPy.
All functions are designed to be:
- Vectorized for performance
- Rolling/streaming safe
- NaN-aware with proper handling
- Aligned to input index

Standard signature: fn(series_or_df, n=..., min_periods=None, **params)
"""

from .core import *
from .trend import *
from .momentum import *
from .volatility import *

__all__ = [
    # Core transforms
    'hl2', 'hlc3', 'ohlc4', 'tp', 'median_price',
    'log_price', 'lag', 'diff', 'pct_change',
    'rolling_sum', 'rolling_mean', 'rolling_std',
    'rolling_min', 'rolling_max', 'rolling_median',
    'rolling_quantile', 'rolling_corr', 'rolling_cov',
    'zscore', 'robust_zscore_mad',
    
    # Returns
    'close_to_close_ret', 'open_to_close_ret', 
    'close_to_open_ret', 'cumulative_return',
    'rolling_return', 'gap_pct', 'overnight_return',
    'intraday_return',
    
    # Moving averages
    'sma', 'ema', 'wma', 'hma', 'dema', 'tema',
    'zlema', 'kama', 't3',
    
    # Trend indicators
    'adx', 'di_plus', 'di_minus', 'aroon_up', 'aroon_down',
    'aroon_oscillator', 'supertrend', 'parabolic_sar',
    'ichimoku', 'moving_average_crossover',
    
    # Momentum
    'rsi', 'stoch_k', 'stoch_d', 'williams_r', 'roc', 'mom',
    'macd', 'macd_signal', 'macd_hist', 'ppo', 'cci', 'tsi',
    'awesome_oscillator', 'ultimate_oscillator', 'mfi',
    
    # Volatility
    'true_range', 'atr', 'natr', 'historical_vol',
    'parkinson_vol', 'garman_klass_vol', 'rogers_satchell_vol',
    'yang_zhang_vol', 'realized_vol',
    
    # Bands
    'bollinger_bands', 'keltner_channels', 'donchian_channels',
    'chandelier_exit', 'choppiness_index',
    
    # Volume
    'vwap', 'obv', 'acc_dist_line', 'chaikin_money_flow',
    'ease_of_movement', 'price_volume_trend', 'money_flow_index',
    
    # Patterns
    'pivot_points', 'swing_highs_lows', 'higher_highs_lows',
    'engulfing_pattern', 'hammer', 'doji',
    
    # Position sizing
    'fixed_notional_size', 'fixed_risk_size', 'atr_position_size',
    'inverse_vol_position_size',
    
    # Stops
    'atr_stop', 'chandelier_stop', 'trailing_stop_pct',
    'profit_target_pct', 'stop_loss_pct',
]