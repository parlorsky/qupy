"""
Data ingestion module for backtesting engine.
Handles loading and preprocessing CSV klines data.
"""

import pandas as pd
from datetime import datetime
from typing import Tuple, Optional
import numpy as np


def load_klines_csv(path: str, use_polars: bool = False) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Load Binance-style klines CSV data.
    
    Expected columns:
    - open_time, close_time: epoch ms
    - open, high, low, close: float prices
    - volume: base volume
    - quote_volume: quote volume
    - count, taker_buy_volume, taker_buy_quote_volume, ignore: additional fields
    
    Args:
        path: Path to CSV file
        use_polars: Whether to use Polars (fallback to Pandas if not available)
        
    Returns:
        Tuple of (DataFrame, frequency_hint)
        DataFrame with columns:
        - dt_open, dt_close: UTC timestamps
        - open, high, low, close, volume, quote_volume: numeric data
        frequency_hint: estimated frequency (e.g., '15min', '1h')
    """
    if use_polars:
        try:
            import polars as pl
            return _load_with_polars(path)
        except ImportError:
            print("Polars not available, falling back to Pandas")
    
    return _load_with_pandas(path)


def _load_with_pandas(path: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """Load data using Pandas."""
    df = pd.read_csv(path)
    
    # Convert timestamps from epoch ms to datetime
    df['dt_open'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['dt_close'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
    
    # Ensure numeric columns are float
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
    for col in numeric_cols:
        df[col] = df[col].astype(float)
    
    # Select and reorder columns
    result_df = df[['dt_open', 'dt_close', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']].copy()
    
    # Estimate frequency
    freq_hint = _estimate_frequency(result_df['dt_open'])
    
    return result_df, freq_hint


def _load_with_polars(path: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """Load data using Polars (converted to Pandas for compatibility)."""
    import polars as pl
    
    df = pl.read_csv(path)
    
    # Convert timestamps and ensure proper types
    df = df.with_columns([
        pl.from_epoch(pl.col('open_time'), time_unit='ms').dt.replace_time_zone('UTC').alias('dt_open'),
        pl.from_epoch(pl.col('close_time'), time_unit='ms').dt.replace_time_zone('UTC').alias('dt_close'),
        pl.col('open').cast(pl.Float64),
        pl.col('high').cast(pl.Float64),
        pl.col('low').cast(pl.Float64),
        pl.col('close').cast(pl.Float64),
        pl.col('volume').cast(pl.Float64),
        pl.col('quote_volume').cast(pl.Float64),
    ])
    
    # Select columns and convert to Pandas
    result_df = df.select(['dt_open', 'dt_close', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']).to_pandas()
    
    # Estimate frequency
    freq_hint = _estimate_frequency(result_df['dt_open'])
    
    return result_df, freq_hint


def _estimate_frequency(dt_series: pd.Series) -> Optional[str]:
    """Estimate the frequency of the datetime series."""
    if len(dt_series) < 2:
        return None
    
    # Calculate median time difference
    time_diffs = dt_series.diff().dropna()
    median_diff = time_diffs.median()
    
    # Convert to common frequency strings
    total_seconds = median_diff.total_seconds()
    
    if total_seconds == 60:
        return '1min'
    elif total_seconds == 300:
        return '5min'
    elif total_seconds == 900:
        return '15min'
    elif total_seconds == 1800:
        return '30min'
    elif total_seconds == 3600:
        return '1h'
    elif total_seconds == 14400:
        return '4h'
    elif total_seconds == 86400:
        return '1d'
    else:
        # Return in minutes or hours
        if total_seconds < 3600:
            return f'{int(total_seconds/60)}min'
        elif total_seconds < 86400:
            return f'{int(total_seconds/3600)}h'
        else:
            return f'{int(total_seconds/86400)}d'


def validate_klines_data(df: pd.DataFrame) -> bool:
    """
    Validate that the klines data is properly formatted.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_cols = ['dt_open', 'dt_close', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']
    
    # Check required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for NaN values in critical columns
    critical_cols = ['open', 'high', 'low', 'close']
    for col in critical_cols:
        if df[col].isna().any():
            raise ValueError(f"NaN values found in {col} column")
    
    # Check OHLC logic
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    )
    
    if invalid_ohlc.any():
        raise ValueError("Invalid OHLC data: high < low or prices outside high/low range")
    
    # Check timestamp ordering
    if not df['dt_open'].is_monotonic_increasing:
        raise ValueError("Timestamps are not in ascending order")
    
    # Check that close_time > open_time
    if (df['dt_close'] <= df['dt_open']).any():
        raise ValueError("Invalid timestamps: close_time <= open_time")
    
    return True