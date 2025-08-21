"""
Plotting functions for backtesting results visualization.
"""

from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
from .backtest import BacktestResult

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    mdates = None
    Figure = None


def plot_pnl(result: BacktestResult, mode: str = "percent", 
             figsize: Tuple[int, int] = (12, 8), show_drawdown: bool = True,
             title: Optional[str] = None) -> Optional[Figure]:
    """
    Plot P&L curve with optional drawdown overlay.
    
    Args:
        result: BacktestResult object
        mode: "percent" for percentage returns, "absolute" for absolute P&L
        figsize: Figure size tuple
        show_drawdown: Whether to show drawdown as shaded area
        title: Custom title for the plot
        
    Returns:
        Matplotlib Figure object or None if matplotlib not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Please install with: pip install matplotlib")
        return None
    
    if len(result.equity_curve) == 0:
        print("No equity data to plot")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    timestamps = result.equity_curve.index
    initial_cash = result.initial_cash
    
    if mode == "percent":
        # Percentage returns
        pnl_series = (result.equity_curve / initial_cash - 1) * 100
        ylabel = "P&L (%)"
        zero_line = 0
    else:
        # Absolute P&L
        pnl_series = result.equity_curve - initial_cash
        ylabel = f"P&L ({result.currency_label})"
        zero_line = 0
    
    # Plot main P&L line
    ax.plot(timestamps, pnl_series, linewidth=2, color='blue', label='P&L')
    
    # Add zero line
    ax.axhline(y=zero_line, color='gray', linestyle='--', alpha=0.7)
    
    # Add drawdown shading if requested
    if show_drawdown and mode == "percent":
        # Calculate drawdown
        running_max = pnl_series.expanding().max()
        drawdown = pnl_series - running_max
        
        # Fill drawdown area
        ax.fill_between(timestamps, pnl_series, running_max, 
                       where=(drawdown < 0), color='red', alpha=0.3, 
                       label='Drawdown')
    
    # Formatting
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set title
    if title is None:
        final_pnl = pnl_series.iloc[-1]
        if mode == "percent":
            title = f"{result.symbol} Strategy P&L: {final_pnl:.2f}%"
        else:
            title = f"{result.symbol} Strategy P&L: {final_pnl:,.0f} {result.currency_label}"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    return fig


def plot_dual_axis_pnl(result: BacktestResult, figsize: Tuple[int, int] = (12, 8),
                       title: Optional[str] = None) -> Optional[Figure]:
    """
    Plot P&L with dual y-axis showing both percentage and absolute values.
    
    Args:
        result: BacktestResult object
        figsize: Figure size tuple
        title: Custom title for the plot
        
    Returns:
        Matplotlib Figure object or None if matplotlib not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Please install with: pip install matplotlib")
        return None
    
    if len(result.equity_curve) == 0:
        print("No equity data to plot")
        return None
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    timestamps = result.equity_curve.index
    initial_cash = result.initial_cash
    
    # Left y-axis: Percentage
    pnl_pct = (result.equity_curve / initial_cash - 1) * 100
    color1 = 'blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('P&L (%)', color=color1)
    line1 = ax1.plot(timestamps, pnl_pct, color=color1, linewidth=2, label='P&L %')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax1.grid(True, alpha=0.3)
    
    # Right y-axis: Absolute
    ax2 = ax1.twinx()
    pnl_abs = result.equity_curve - initial_cash
    color2 = 'red'
    ax2.set_ylabel(f'P&L ({result.currency_label})', color=color2)
    line2 = ax2.plot(timestamps, pnl_abs, color=color2, linewidth=2, 
                     linestyle='--', alpha=0.7, label=f'P&L {result.currency_label}')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title
    if title is None:
        final_pct = pnl_pct.iloc[-1]
        final_abs = pnl_abs.iloc[-1]
        title = (f"{result.symbol} Strategy P&L: {final_pct:.2f}% "
                f"({final_abs:+,.0f} {result.currency_label})")
    
    ax1.set_title(title, fontsize=14, fontweight='bold')
    
    # Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    return fig


def plot_drawdown(result: BacktestResult, mode: str = "percent", 
                  figsize: Tuple[int, int] = (12, 6)) -> Optional[Figure]:
    """
    Plot drawdown curve.
    
    Args:
        result: BacktestResult object
        mode: "percent" for percentage drawdown, "absolute" for absolute
        figsize: Figure size tuple
        
    Returns:
        Matplotlib Figure object or None if matplotlib not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Please install with: pip install matplotlib")
        return None
    
    if len(result.equity_curve) == 0:
        print("No equity data to plot")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    timestamps = result.equity_curve.index
    
    if mode == "percent":
        # Percentage drawdown
        equity_pct = result.equity_curve / result.initial_cash
        running_max = equity_pct.expanding().max()
        drawdown = (equity_pct / running_max - 1) * 100
        ylabel = "Drawdown (%)"
    else:
        # Absolute drawdown
        running_max = result.equity_curve.expanding().max()
        drawdown = result.equity_curve - running_max
        ylabel = f"Drawdown ({result.currency_label})"
    
    # Plot drawdown
    ax.plot(timestamps, drawdown, color='red', linewidth=2)
    ax.fill_between(timestamps, drawdown, 0, alpha=0.3, color='red')
    
    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
    
    # Formatting
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{result.symbol} Drawdown", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Invert y-axis for drawdown (more negative = worse)
    if mode == "percent":
        ax.invert_yaxis()
        max_dd = drawdown.min()
        ax.text(0.02, 0.95, f'Max Drawdown: {max_dd:.2f}%', 
                transform=ax.transAxes, fontsize=12, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    return fig


def plot_trade_analysis(result: BacktestResult, figsize: Tuple[int, int] = (15, 10)) -> Optional[Figure]:
    """
    Plot comprehensive trade analysis with multiple subplots.
    
    Args:
        result: BacktestResult object
        figsize: Figure size tuple
        
    Returns:
        Matplotlib Figure object or None if matplotlib not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Please install with: pip install matplotlib")
        return None
    
    if not result.trades:
        print("No completed trades to analyze")
        return None
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    trades = result.trades
    pnl_values = [t.pnl_abs for t in trades]
    pnl_pct_values = [t.pnl_pct * 100 for t in trades]
    
    # 1. P&L per trade
    trade_numbers = list(range(1, len(trades) + 1))
    colors = ['green' if pnl > 0 else 'red' for pnl in pnl_values]
    
    ax1.bar(trade_numbers, pnl_values, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.set_title('P&L per Trade (Absolute)')
    ax1.set_xlabel('Trade #')
    ax1.set_ylabel(f'P&L ({result.currency_label})')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative P&L
    cumulative_pnl = np.cumsum(pnl_values)
    ax2.plot(trade_numbers, cumulative_pnl, marker='o', linewidth=2, markersize=4)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_title('Cumulative P&L')
    ax2.set_xlabel('Trade #')
    ax2.set_ylabel(f'Cumulative P&L ({result.currency_label})')
    ax2.grid(True, alpha=0.3)
    
    # 3. P&L distribution histogram
    ax3.hist(pnl_pct_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax3.set_title('P&L Distribution')
    ax3.set_xlabel('P&L (%)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # 4. MFE vs MAE scatter
    mfe_values = [t.mfe_pct * 100 for t in trades]
    mae_values = [t.mae_pct * 100 for t in trades]
    
    scatter_colors = ['green' if pnl > 0 else 'red' for pnl in pnl_values]
    ax4.scatter(mae_values, mfe_values, c=scatter_colors, alpha=0.7)
    ax4.set_title('Max Favorable vs Max Adverse Excursion')
    ax4.set_xlabel('MAE (%)')
    ax4.set_ylabel('MFE (%)')
    ax4.grid(True, alpha=0.3)
    
    # Add diagonal line (MFE = MAE line)
    max_val = max(max(mfe_values) if mfe_values else 0, 
                  max(mae_values) if mae_values else 0)
    ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    
    plt.suptitle(f'{result.symbol} Trade Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def save_plot(fig: Figure, filepath: str, dpi: int = 300, format: str = 'png') -> bool:
    """
    Save a matplotlib figure to file.
    
    Args:
        fig: Matplotlib Figure object
        filepath: Path to save the file
        dpi: Resolution for raster formats
        format: File format ('png', 'pdf', 'svg', etc.)
        
    Returns:
        True if successful, False otherwise
    """
    if fig is None:
        return False
    
    try:
        fig.savefig(filepath, dpi=dpi, format=format, bbox_inches='tight')
        return True
    except Exception as e:
        print(f"Error saving plot: {e}")
        return False


def show_plot(fig: Figure) -> None:
    """
    Display a matplotlib figure.
    
    Args:
        fig: Matplotlib Figure object
    """
    if fig is not None:
        plt.show()
    else:
        print("No figure to display")


def create_quick_plots(result: BacktestResult, save_dir: Optional[str] = None) -> List[Figure]:
    """
    Create a standard set of plots for a backtest result.
    
    Args:
        result: BacktestResult object
        save_dir: Directory to save plots (optional)
        
    Returns:
        List of created Figure objects
    """
    figures = []
    
    # P&L plot (percentage)
    fig1 = plot_pnl(result, mode="percent", show_drawdown=True)
    if fig1:
        figures.append(fig1)
        if save_dir:
            save_plot(fig1, f"{save_dir}/{result.symbol}_pnl_percent.png")
    
    # P&L plot (absolute)
    fig2 = plot_pnl(result, mode="absolute", show_drawdown=False)
    if fig2:
        figures.append(fig2)
        if save_dir:
            save_plot(fig2, f"{save_dir}/{result.symbol}_pnl_absolute.png")
    
    # Drawdown plot
    fig3 = plot_drawdown(result, mode="percent")
    if fig3:
        figures.append(fig3)
        if save_dir:
            save_plot(fig3, f"{save_dir}/{result.symbol}_drawdown.png")
    
    # Trade analysis (if trades exist)
    if result.trades:
        fig4 = plot_trade_analysis(result)
        if fig4:
            figures.append(fig4)
            if save_dir:
                save_plot(fig4, f"{save_dir}/{result.symbol}_trade_analysis.png")
    
    return figures