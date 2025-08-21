"""
Performance metrics calculation for backtesting results.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from .trades import Trade
from .backtest import BacktestResult


def compute_trade_metrics(trades: List[Trade]) -> Dict[str, Any]:
    """
    Compute trade-based performance metrics.
    
    Args:
        trades: List of closed trades
        
    Returns:
        Dictionary of trade metrics
    """
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'percent_profitable': 0.0,
            'avg_pnl_per_trade': 0.0,
            'avg_winning_trade': 0.0,
            'avg_losing_trade': 0.0,
            'win_loss_ratio': 0.0,
            'largest_winner_abs': 0.0,
            'largest_winner_pct': 0.0,
            'largest_loser_abs': 0.0,
            'largest_loser_pct': 0.0,
            'avg_bars_per_trade': 0.0,
            'avg_bars_winning_trades': 0.0,
            'avg_bars_losing_trades': 0.0,
            'total_pnl_abs': 0.0
        }
    
    # Basic counts
    total_trades = len(trades)
    winning_trades = [t for t in trades if t.pnl_abs > 0]
    losing_trades = [t for t in trades if t.pnl_abs < 0]
    
    num_winners = len(winning_trades)
    num_losers = len(losing_trades)
    
    # P&L metrics
    pnl_values = [t.pnl_abs for t in trades]
    total_pnl = sum(pnl_values)
    avg_pnl_per_trade = total_pnl / total_trades
    
    # Win/loss statistics
    percent_profitable = (num_winners / total_trades) * 100 if total_trades > 0 else 0.0
    
    avg_winning_trade = np.mean([t.pnl_abs for t in winning_trades]) if winning_trades else 0.0
    avg_losing_trade = abs(np.mean([t.pnl_abs for t in losing_trades])) if losing_trades else 0.0
    
    win_loss_ratio = avg_winning_trade / avg_losing_trade if avg_losing_trade > 0 else 0.0
    
    # Largest winners/losers
    largest_winner_abs = max([t.pnl_abs for t in winning_trades]) if winning_trades else 0.0
    largest_winner_pct = max([t.pnl_pct for t in winning_trades]) if winning_trades else 0.0
    
    largest_loser_abs = abs(min([t.pnl_abs for t in losing_trades])) if losing_trades else 0.0
    largest_loser_pct = abs(min([t.pnl_pct for t in losing_trades])) if losing_trades else 0.0
    
    # Trade duration metrics
    avg_bars_per_trade = np.mean([t.bars_in_trade for t in trades])
    avg_bars_winning = np.mean([t.bars_in_trade for t in winning_trades]) if winning_trades else 0.0
    avg_bars_losing = np.mean([t.bars_in_trade for t in losing_trades]) if losing_trades else 0.0
    
    return {
        'total_trades': total_trades,
        'winning_trades': num_winners,
        'losing_trades': num_losers,
        'percent_profitable': percent_profitable,
        'avg_pnl_per_trade': avg_pnl_per_trade,
        'avg_winning_trade': avg_winning_trade,
        'avg_losing_trade': avg_losing_trade,
        'win_loss_ratio': win_loss_ratio,
        'largest_winner_abs': largest_winner_abs,
        'largest_winner_pct': largest_winner_pct,
        'largest_loser_abs': largest_loser_abs,
        'largest_loser_pct': largest_loser_pct,
        'avg_bars_per_trade': avg_bars_per_trade,
        'avg_bars_winning_trades': avg_bars_winning,
        'avg_bars_losing_trades': avg_bars_losing,
        'total_pnl_abs': total_pnl
    }


def compute_equity_metrics(equity_curve: pd.Series, initial_cash: float) -> Dict[str, Any]:
    """
    Compute equity curve based metrics.
    
    Args:
        equity_curve: Time series of equity values
        initial_cash: Initial capital
        
    Returns:
        Dictionary of equity metrics
    """
    if len(equity_curve) == 0:
        return {
            'total_return_pct': 0.0,
            'total_return_abs': 0.0,
            'max_drawdown_pct': 0.0,
            'max_drawdown_abs': 0.0,
            'current_drawdown_pct': 0.0,
            'current_drawdown_abs': 0.0
        }
    
    final_equity = equity_curve.iloc[-1]
    
    # Total return
    total_return_abs = final_equity - initial_cash
    total_return_pct = (final_equity / initial_cash - 1) * 100
    
    # Drawdown calculations
    running_max = equity_curve.expanding().max()
    drawdown_abs = equity_curve - running_max
    drawdown_pct = (equity_curve / running_max - 1) * 100
    
    max_drawdown_abs = drawdown_abs.min()
    max_drawdown_pct = drawdown_pct.min()
    
    current_drawdown_abs = drawdown_abs.iloc[-1]
    current_drawdown_pct = drawdown_pct.iloc[-1]
    
    return {
        'total_return_pct': total_return_pct,
        'total_return_abs': total_return_abs,
        'max_drawdown_pct': max_drawdown_pct,
        'max_drawdown_abs': max_drawdown_abs,
        'current_drawdown_pct': current_drawdown_pct,
        'current_drawdown_abs': current_drawdown_abs
    }


def compute_sharpe_ratio(returns_series: pd.Series, periods_per_year: Optional[int] = None, 
                        risk_free_rate: float = 0.0) -> float:
    """
    Compute annualized Sharpe ratio.
    
    Args:
        returns_series: Series of period returns
        periods_per_year: Number of periods per year for annualization
        risk_free_rate: Risk-free rate (annual)
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns_series) == 0 or returns_series.std() == 0:
        return 0.0
    
    # Auto-detect periods per year if not provided
    if periods_per_year is None:
        periods_per_year = estimate_periods_per_year(returns_series.index)
    
    if periods_per_year == 0:
        return 0.0
    
    # Calculate excess returns
    period_risk_free_rate = risk_free_rate / periods_per_year
    excess_returns = returns_series - period_risk_free_rate
    
    # Sharpe ratio
    mean_excess_return = excess_returns.mean()
    return_std = excess_returns.std()
    
    if return_std == 0:
        return 0.0
    
    # Annualize
    sharpe = (mean_excess_return / return_std) * np.sqrt(periods_per_year)
    
    return sharpe


def estimate_periods_per_year(datetime_index: pd.Index) -> int:
    """
    Estimate periods per year from datetime index.
    
    Args:
        datetime_index: DateTime index
        
    Returns:
        Estimated periods per year
    """
    if len(datetime_index) < 2:
        return 252  # Default to daily
    
    # Calculate median time difference
    time_diffs = pd.Series(datetime_index).diff().dropna()
    median_diff = time_diffs.median()
    
    # Convert to periods per year
    seconds_per_period = median_diff.total_seconds()
    seconds_per_year = 365.25 * 24 * 3600
    
    periods_per_year = int(seconds_per_year / seconds_per_period)
    
    return periods_per_year


def compute_sortino_ratio(returns_series: pd.Series, periods_per_year: Optional[int] = None,
                         target_return: float = 0.0) -> float:
    """
    Compute Sortino ratio (downside deviation version of Sharpe ratio).
    
    Args:
        returns_series: Series of period returns  
        periods_per_year: Number of periods per year
        target_return: Target return (annual)
        
    Returns:
        Annualized Sortino ratio
    """
    if len(returns_series) == 0:
        return 0.0
    
    if periods_per_year is None:
        periods_per_year = estimate_periods_per_year(returns_series.index)
    
    if periods_per_year == 0:
        return 0.0
    
    # Calculate excess returns
    period_target_return = target_return / periods_per_year
    excess_returns = returns_series - period_target_return
    
    # Downside deviation (only negative excess returns)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    downside_std = downside_returns.std()
    if downside_std == 0:
        return 0.0
    
    # Sortino ratio
    mean_excess_return = excess_returns.mean()
    sortino = (mean_excess_return / downside_std) * np.sqrt(periods_per_year)
    
    return sortino


def compute_all_metrics(result: BacktestResult, periods_per_year: Optional[int] = None,
                       risk_free_rate: float = 0.0) -> Dict[str, Any]:
    """
    Compute comprehensive performance metrics from backtest result.
    
    Args:
        result: BacktestResult object
        periods_per_year: Periods per year for annualization
        risk_free_rate: Risk-free rate for Sharpe calculation
        
    Returns:
        Dictionary of all computed metrics
    """
    
    # Trade metrics
    trade_metrics = compute_trade_metrics(result.trades)
    
    # Equity metrics
    equity_metrics = compute_equity_metrics(result.equity_curve, result.initial_cash)
    
    # Risk metrics
    sharpe = compute_sharpe_ratio(result.returns_series, periods_per_year, risk_free_rate)
    sortino = compute_sortino_ratio(result.returns_series, periods_per_year)
    
    # Open trades count
    open_trades_count = len(result.open_trades)
    
    # Combine all metrics
    all_metrics = {
        **trade_metrics,
        **equity_metrics,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'open_trades': open_trades_count,
        'total_fees_paid': result.total_fees_paid,
        'total_slippage_cost': result.total_slippage_cost,
        'final_cash': result.final_cash,
        'final_equity': result.final_equity
    }
    
    return all_metrics


def print_metrics_summary(metrics: Dict[str, Any], currency_label: str = "usp") -> None:
    """
    Print a formatted summary of metrics.
    
    Args:
        metrics: Dictionary of metrics
        currency_label: Currency label for display
    """
    
    print("=" * 60)
    print("BACKTEST PERFORMANCE SUMMARY")
    print("=" * 60)
    
    # Overall Performance
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Total Return:        {metrics['total_return_pct']:>8.2f}% ({metrics['total_return_abs']:>10,.2f} {currency_label})")
    print(f"  Max Drawdown:        {metrics['max_drawdown_pct']:>8.2f}% ({metrics['max_drawdown_abs']:>10,.2f} {currency_label})")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>8.2f}")
    print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>8.2f}")
    
    # Trade Statistics
    print(f"\nTRADE STATISTICS:")
    print(f"  Total Trades:        {metrics['total_trades']:>8d}")
    print(f"  Winning Trades:      {metrics['winning_trades']:>8d}")
    print(f"  Losing Trades:       {metrics['losing_trades']:>8d}")
    print(f"  Open Trades:         {metrics['open_trades']:>8d}")
    print(f"  Percent Profitable:  {metrics['percent_profitable']:>8.1f}%")
    
    # P&L Analysis
    print(f"\nP&L ANALYSIS:")
    print(f"  Avg P&L per Trade:   {metrics['avg_pnl_per_trade']:>10,.2f} {currency_label}")
    print(f"  Avg Winning Trade:   {metrics['avg_winning_trade']:>10,.2f} {currency_label}")
    print(f"  Avg Losing Trade:    {metrics['avg_losing_trade']:>10,.2f} {currency_label}")
    print(f"  Win/Loss Ratio:      {metrics['win_loss_ratio']:>10.2f}")
    print(f"  Largest Winner:      {metrics['largest_winner_abs']:>10,.2f} {currency_label} ({metrics['largest_winner_pct']:>6.2f}%)")
    print(f"  Largest Loser:       {metrics['largest_loser_abs']:>10,.2f} {currency_label} ({metrics['largest_loser_pct']:>6.2f}%)")
    
    # Trade Duration
    print(f"\nTRADE DURATION:")
    print(f"  Avg Bars/Trade:      {metrics['avg_bars_per_trade']:>8.1f}")
    print(f"  Avg Bars (Winners):  {metrics['avg_bars_winning_trades']:>8.1f}")
    print(f"  Avg Bars (Losers):   {metrics['avg_bars_losing_trades']:>8.1f}")
    
    # Costs
    print(f"\nCOSTS:")
    print(f"  Total Fees:          {metrics['total_fees_paid']:>10,.2f} {currency_label}")
    print(f"  Total Slippage:      {metrics['total_slippage_cost']:>10,.2f} {currency_label}")
    
    print("=" * 60)