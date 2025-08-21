"""
Utility functions for the backtesting engine.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json


def format_currency(value: float, currency: str = "USD", decimals: int = 2) -> str:
    """
    Format a number as currency.

    Args:
        value: Numeric value to format
        currency: Currency label
        decimals: Number of decimal places

    Returns:
        Formatted currency string
    """
    if abs(value) >= 1000000:
        return f"${value/1000000:.{decimals-1}f}M {currency}"
    elif abs(value) >= 1000:
        return f"${value/1000:.{decimals}f}K {currency}"
    else:
        return f"${value:.{decimals}f} {currency}"


def format_percentage(value: float, decimals: int = 2, include_sign: bool = True) -> str:
    """
    Format a decimal as percentage.

    Args:
        value: Decimal value (e.g., 0.05 for 5%)
        decimals: Number of decimal places
        include_sign: Whether to include +/- sign

    Returns:
        Formatted percentage string
    """
    pct = value * 100
    sign = "+" if pct > 0 and include_sign else ""
    return f"{sign}{pct:.{decimals}f}%"


def calculate_periods_per_year(frequency: str) -> int:
    """
    Calculate periods per year from frequency string.

    Args:
        frequency: Frequency string (e.g., '1min', '5min', '1h', '1d')

    Returns:
        Number of periods per year
    """
    freq_map = {
        '1min': 525600,    # 60*24*365
        '5min': 105120,    # 12*24*365
        '15min': 35040,    # 4*24*365
        '30min': 17520,    # 2*24*365
        '1h': 8760,        # 24*365
        '4h': 2190,        # 6*365
        '1d': 365,         # 365
        '1w': 52,          # 52
    }

    return freq_map.get(frequency, 252)  # Default to daily trading days


def resample_ohlcv(data: pd.DataFrame, target_freq: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a different frequency.

    Args:
        data: OHLCV DataFrame with datetime index
        target_freq: Target frequency (pandas frequency string)

    Returns:
        Resampled DataFrame
    """
    if 'dt_open' in data.columns:
        data = data.set_index('dt_open')

    resampled = data.resample(target_freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum'
    }).dropna()

    # Add dt_open and dt_close columns
    resampled['dt_open'] = resampled.index
    resampled['dt_close'] = resampled.index + pd.Timedelta(target_freq) - pd.Timedelta('1s')

    return resampled.reset_index(drop=True)


def validate_strategy_params(params: Dict[str, Any], required_params: List[str],
                           param_types: Dict[str, type] = None) -> bool:
    """
    Validate strategy parameters.

    Args:
        params: Parameter dictionary
        required_params: List of required parameter names
        param_types: Optional type validation dict

    Returns:
        True if valid, raises ValueError if invalid
    """
    # Check required parameters
    missing = [p for p in required_params if p not in params]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")

    # Check parameter types
    if param_types:
        for param, expected_type in param_types.items():
            if param in params and not isinstance(params[param], expected_type):
                raise ValueError(f"Parameter '{param}' must be of type {expected_type.__name__}")

    return True


def calculate_rolling_stats(series: pd.Series, window: int) -> Dict[str, pd.Series]:
    """
    Calculate rolling statistics for a time series.

    Args:
        series: Input time series
        window: Rolling window size

    Returns:
        Dictionary of rolling statistics
    """
    return {
        'mean': series.rolling(window).mean(),
        'std': series.rolling(window).std(),
        'min': series.rolling(window).min(),
        'max': series.rolling(window).max(),
        'median': series.rolling(window).median()
    }


def detect_outliers(series: pd.Series, method: str = 'iqr',
                   threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in a time series.

    Args:
        series: Input time series
        method: Method for outlier detection ('iqr', 'zscore')
        threshold: Threshold for outlier detection

    Returns:
        Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)

    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold

    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")


def export_backtest_results(result, filepath: str, format: str = 'json') -> bool:
    """
    Export backtest results to file.

    Args:
        result: BacktestResult object
        filepath: Output file path
        format: Export format ('json', 'csv')

    Returns:
        True if successful, False otherwise
    """
    try:
        if format == 'json':
            export_data = {
                'symbol': result.symbol,
                'initial_cash': result.initial_cash,
                'final_equity': result.final_equity,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat(),
                'total_trades': len(result.trades),
                'open_trades': len(result.open_trades),
                'trades': [
                    {
                        'trade_id': t.trade_id,
                        'direction': t.direction,
                        'entry_time': t.entry_time.isoformat(),
                        'entry_price': t.entry_price,
                        'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                        'exit_price': t.exit_price,
                        'qty': t.qty,
                        'pnl_abs': t.pnl_abs,
                        'pnl_pct': t.pnl_pct,
                        'bars_in_trade': t.bars_in_trade,
                        'mfe_abs': t.mfe_abs,
                        'mae_abs': t.mae_abs
                    }
                    for t in result.trades
                ],
                'equity_curve': result.equity_curve.to_dict()
            }

            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

        elif format == 'csv':
            # Export trades to CSV
            if result.trades:
                trades_df = pd.DataFrame([
                    {
                        'trade_id': t.trade_id,
                        'direction': t.direction,
                        'entry_time': t.entry_time,
                        'entry_price': t.entry_price,
                        'exit_time': t.exit_time,
                        'exit_price': t.exit_price,
                        'qty': t.qty,
                        'pnl_abs': t.pnl_abs,
                        'pnl_pct': t.pnl_pct,
                        'bars_in_trade': t.bars_in_trade
                    }
                    for t in result.trades
                ])
                trades_df.to_csv(filepath, index=False)

        return True

    except Exception as e:
        print(f"Error exporting results: {e}")
        return False


def load_multiple_symbols(data_dir: str, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Load data for multiple symbols.

    Args:
        data_dir: Directory containing CSV files
        symbols: List of symbol names

    Returns:
        Dictionary mapping symbol names to DataFrames
    """
    from .data import load_klines_csv

    data_dict = {}

    for symbol in symbols:
        filepath = f"{data_dir}/{symbol}.csv"
        try:
            data, _ = load_klines_csv(filepath)
            data_dict[symbol] = data
            print(f"Loaded {len(data)} bars for {symbol}")
        except Exception as e:
            print(f"Warning: Could not load {symbol}: {e}")

    return data_dict


def align_timestamps(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Align timestamps across multiple DataFrames.

    Args:
        data_dict: Dictionary of DataFrames with timestamp columns

    Returns:
        Dictionary of aligned DataFrames
    """
    if not data_dict:
        return {}

    # Find common timestamp range
    all_starts = [df.iloc[0]['dt_open'] for df in data_dict.values()]
    all_ends = [df.iloc[-1]['dt_close'] for df in data_dict.values()]

    common_start = max(all_starts)
    common_end = min(all_ends)

    # Filter each DataFrame to common range
    aligned_dict = {}
    for symbol, df in data_dict.items():
        mask = (df['dt_open'] >= common_start) & (df['dt_close'] <= common_end)
        aligned_df = df[mask].reset_index(drop=True)
        aligned_dict[symbol] = aligned_df
        print(f"Aligned {symbol}: {len(aligned_df)} bars from {common_start} to {common_end}")

    return aligned_dict


def create_benchmark_strategy(returns_per_year: float = 0.05):
    """
    Create a simple benchmark strategy that generates constant returns.

    Args:
        returns_per_year: Annual return rate (e.g., 0.05 for 5%)

    Returns:
        Strategy that can be used for comparison
    """
    from .strategy import Strategy, Context
    import pandas as pd

    class BenchmarkStrategy(Strategy):
        def __init__(self, annual_return):
            super().__init__()
            self.annual_return = annual_return

        def on_start(self, context: Context):
            # Buy and hold with the entire balance
            context.buy(context.cash * 0.99, "Benchmark entry", "notional")

        def on_bar(self, context: Context, bar: pd.Series):
            # Do nothing - just hold
            pass

        def on_stop(self, context: Context):
            if not context.position.is_flat:
                context.close("End of benchmark")

    return BenchmarkStrategy(returns_per_year)


def performance_summary_table(results_dict: Dict[str, Any]) -> str:
    """
    Create a formatted performance summary table.

    Args:
        results_dict: Dictionary mapping strategy names to BacktestResults

    Returns:
        Formatted table string
    """
    from .metrics import compute_all_metrics

    if not results_dict:
        return "No results to display"

    # Calculate metrics for each strategy
    metrics_dict = {}
    for name, result in results_dict.items():
        metrics_dict[name] = compute_all_metrics(result)

    # Create table
    headers = ['Strategy', 'Return%', 'MaxDD%', 'Sharpe', 'Trades', 'Win%']
    col_widths = [20, 10, 10, 8, 8, 8]

    # Header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-" * len(header_line)

    table_lines = [header_line, separator]

    # Data rows
    for name, metrics in metrics_dict.items():
        row = [
            name[:col_widths[0]],
            f"{metrics['total_return_pct']:.2f}%",
            f"{metrics['max_drawdown_pct']:.2f}%",
            f"{metrics['sharpe_ratio']:.2f}",
            str(metrics['total_trades']),
            f"{metrics['percent_profitable']:.1f}%"
        ]
        row_line = " | ".join(val.ljust(w) for val, w in zip(row, col_widths))
        table_lines.append(row_line)

    return "\n".join(table_lines)


# Time-related utilities
def market_hours_filter(data: pd.DataFrame, start_hour: int = 9,
                       end_hour: int = 16, timezone: str = 'UTC') -> pd.DataFrame:
    """
    Filter data to market hours only.

    Args:
        data: DataFrame with datetime columns
        start_hour: Market open hour (24h format)
        end_hour: Market close hour (24h format)
        timezone: Timezone for hours

    Returns:
        Filtered DataFrame
    """
    if timezone != 'UTC':
        # Convert to specified timezone
        dt_col = data['dt_open'].dt.tz_convert(timezone)
    else:
        dt_col = data['dt_open']

    hour_mask = (dt_col.dt.hour >= start_hour) & (dt_col.dt.hour < end_hour)
    return data[hour_mask].reset_index(drop=True)


def add_technical_indicators(data: pd.DataFrame, indicators: List[str] = None) -> pd.DataFrame:
    """
    Add basic technical indicators to OHLCV data.

    Args:
        data: OHLCV DataFrame
        indicators: List of indicators to add ('sma', 'ema', 'rsi', 'bb')

    Returns:
        DataFrame with additional indicator columns
    """
    df = data.copy()

    if indicators is None:
        indicators = ['sma_20', 'ema_20', 'rsi_14']

    # Simple Moving Average
    if any('sma' in ind for ind in indicators):
        for ind in indicators:
            if ind.startswith('sma_'):
                period = int(ind.split('_')[1])
                df[ind] = df['close'].rolling(window=period).mean()

    # Exponential Moving Average
    if any('ema' in ind for ind in indicators):
        for ind in indicators:
            if ind.startswith('ema_'):
                period = int(ind.split('_')[1])
                df[ind] = df['close'].ewm(span=period).mean()

    # RSI
    if any('rsi' in ind for ind in indicators):
        for ind in indicators:
            if ind.startswith('rsi_'):
                period = int(ind.split('_')[1])
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[ind] = 100 - (100 / (1 + rs))

    return df