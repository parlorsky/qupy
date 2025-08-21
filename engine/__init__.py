"""
Backtesting Engine for Strategy Development and Testing.

A comprehensive backtesting framework for quantitative trading strategies.
"""

from .data import load_klines_csv, validate_klines_data
from .strategy import Strategy, Context, Position, Order, FixedSizeStrategy, BuyAndHoldStrategy
from .backtest import BacktestEngine, BacktestResult, run_backtest
from .fills import FillConfig, FillEngine, Fill, PositionManager, create_fill_config
from .trades import Trade, TradeTracker, build_trade_table, finalize_trade, create_trade_id
from .metrics import (
    compute_trade_metrics, compute_equity_metrics, compute_sharpe_ratio,
    compute_sortino_ratio, compute_all_metrics, print_metrics_summary
)
from .plot import (
    plot_pnl, plot_dual_axis_pnl, plot_drawdown, plot_trade_analysis,
    save_plot, show_plot, create_quick_plots
)
from .utils import (
    format_currency, format_percentage, calculate_periods_per_year,
    validate_strategy_params, export_backtest_results, performance_summary_table
)

__version__ = "1.0.0"
__author__ = "Backtesting Engine Team"

# Main API functions for easy import
__all__ = [
    # Data handling
    'load_klines_csv',
    'validate_klines_data',
    
    # Strategy development
    'Strategy',
    'FixedSizeStrategy', 
    'BuyAndHoldStrategy',
    'Context',
    'Position',
    'Order',
    
    # Backtesting
    'run_backtest',
    'BacktestEngine',
    'BacktestResult',
    
    # Configuration
    'FillConfig',
    'create_fill_config',
    
    # Trade analysis
    'Trade',
    'TradeTracker',
    'build_trade_table',
    
    # Metrics
    'compute_all_metrics',
    'print_metrics_summary',
    'compute_sharpe_ratio',
    
    # Plotting
    'plot_pnl',
    'plot_trade_analysis',
    'create_quick_plots',
    
    # Utilities
    'format_currency',
    'format_percentage',
    'export_backtest_results',
    'performance_summary_table'
]