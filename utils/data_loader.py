"""
Data loading utilities for qupy framework.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union
from pathlib import Path


def load_binance_data(filepath: Union[str, Path], 
                     resample: Optional[str] = None) -> pd.DataFrame:
    """
    Load Binance historical data from CSV.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file
    resample : str, optional
        Resample frequency (e.g., '1H', '4H', '1D')
    
    Returns
    -------
    pd.DataFrame
        OHLCV data with datetime index
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Convert timestamp (Binance uses milliseconds)
    if 'open_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    elif 'timestamp' in df.columns:
        # Check if it's in milliseconds
        if df['timestamp'].iloc[0] > 1e10:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        raise ValueError("No timestamp column found")
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # Select OHLCV columns
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df[ohlcv_cols]
    
    # Convert to float
    for col in ohlcv_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Resample if requested
    if resample:
        df = df.resample(resample).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    
    return df


def load_sample_data(symbol: str = 'BTCUSDT', 
                    period: str = '1d',
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load sample data for testing.
    
    Parameters
    ----------
    symbol : str
        Symbol to load
    period : str
        Time period ('1h', '4h', '1d')
    start_date : str, optional
        Start date (YYYY-MM-DD)
    end_date : str, optional
        End date (YYYY-MM-DD)
    
    Returns
    -------
    pd.DataFrame
        OHLCV data
    """
    # Try to load from data directory
    data_dir = Path(__file__).parent.parent / 'data'
    
    # Look for matching files
    for file in data_dir.glob(f'{symbol}*.csv'):
        df = load_binance_data(file)
        
        # Filter by date if specified
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        return df
    
    # If no file found, generate synthetic data
    return generate_synthetic_data(n_days=1000)


def generate_synthetic_data(n_days: int = 1000,
                           start_price: float = 100.0,
                           volatility: float = 0.02,
                           trend: float = 0.0001) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing.
    
    Parameters
    ----------
    n_days : int
        Number of days to generate
    start_price : float
        Starting price
    volatility : float
        Daily volatility
    trend : float
        Daily trend (drift)
    
    Returns
    -------
    pd.DataFrame
        Synthetic OHLCV data
    """
    # Generate dates
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='D')
    
    # Generate returns with trend and volatility
    returns = np.random.normal(trend, volatility, n_days)
    
    # Add volatility clustering
    garch_factor = np.zeros(n_days)
    garch_factor[0] = 1.0
    for i in range(1, n_days):
        garch_factor[i] = 0.9 * garch_factor[i-1] + 0.1 * abs(returns[i-1])
    
    returns = returns * (1 + garch_factor)
    
    # Generate close prices
    close = start_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.randn(n_days) * volatility * 0.5))
    low = close * (1 - np.abs(np.random.randn(n_days) * volatility * 0.5))
    
    # Open is previous close with gap
    open_prices = np.roll(close, 1) * (1 + np.random.randn(n_days) * volatility * 0.3)
    open_prices[0] = close[0]
    
    # Adjust high/low to include open
    high = np.maximum(high, open_prices)
    low = np.minimum(low, open_prices)
    
    # Generate volume (correlated with volatility)
    base_volume = 1000000
    volume = base_volume * (1 + np.abs(returns) * 20) * np.random.uniform(0.8, 1.2, n_days)
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume.astype(int)
    }, index=dates)
    
    return df


def prepare_features(df: pd.DataFrame, 
                    indicators: list = None,
                    lookback: int = 20) -> pd.DataFrame:
    """
    Add technical indicators as features.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    indicators : list, optional
        List of indicators to add
    lookback : int
        Lookback period for indicators
    
    Returns
    -------
    pd.DataFrame
        Data with added features
    """
    from indicators.trend import sma, ema
    from indicators.momentum import rsi
    from indicators.volatility import atr
    
    result = df.copy()
    
    if indicators is None:
        indicators = ['sma', 'ema', 'rsi', 'atr']
    
    # Add indicators
    if 'sma' in indicators:
        result[f'sma_{lookback}'] = sma(df['close'], lookback)
    
    if 'ema' in indicators:
        result[f'ema_{lookback}'] = ema(df['close'], lookback)
    
    if 'rsi' in indicators:
        result['rsi'] = rsi(df['close'].values, 14)
    
    if 'atr' in indicators:
        result['atr'] = atr(df['high'].values, df['low'].values, df['close'].values, 14)
    
    # Add returns
    result['returns'] = df['close'].pct_change()
    result['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    return result


def split_train_test(df: pd.DataFrame, 
                    test_size: float = 0.2,
                    gap: int = 0) -> tuple:
    """
    Split data into train and test sets with optional gap.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to split
    test_size : float
        Proportion of data for testing
    gap : int
        Number of periods to gap between train and test
    
    Returns
    -------
    tuple
        (train_df, test_df)
    """
    n = len(df)
    test_n = int(n * test_size)
    train_n = n - test_n - gap
    
    train_df = df.iloc[:train_n]
    test_df = df.iloc[train_n + gap:]
    
    return train_df, test_df