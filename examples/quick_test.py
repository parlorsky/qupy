"""
Quick test of the backtesting engine with the first 100 bars.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.data import load_klines_csv
from engine.backtest import run_backtest
from engine.metrics import compute_all_metrics, print_metrics_summary
from strategies.barup_example import create_basic_barup


def quick_test():
    """Quick test with limited data."""
    
    print("Quick Backtesting Engine Test")
    print("=" * 40)
    
    # Load first 100 bars only
    data, frequency = load_klines_csv("data/CAKEUSDT.csv")
    data = data.head(100)  # Use only first 100 bars
    
    print(f"Testing with {len(data)} bars of {frequency} data")
    print(f"Date range: {data.iloc[0]['dt_open']} to {data.iloc[-1]['dt_close']}")
    
    # Create simple strategy
    strategy = create_basic_barup(
        size_value=500,  # Smaller trade size
        size_mode="notional",
        max_bars=10,     # Shorter time stop
        min_bars=1
    )
    
    # Run backtest
    result = run_backtest(
        data=data,
        strategy=strategy,
        symbol="CAKEUSDT",
        initial_cash=5000,   # Smaller starting capital
        fee_bps=5,           # 0.05% fees
        slippage_bps=2,      # 0.02% slippage
        currency_label="USDT",
        random_seed=42
    )
    
    print(f"\nBacktest completed: {len(result.trades)} trades executed")
    
    # Calculate and display basic metrics
    metrics = compute_all_metrics(result)
    
    print(f"\nResults Summary:")
    print(f"  Total Return:     {metrics['total_return_pct']:.2f}%")
    print(f"  Final Equity:     ${result.final_equity:,.2f}")
    print(f"  Total Trades:     {metrics['total_trades']}")
    print(f"  Win Rate:         {metrics['percent_profitable']:.1f}%")
    print(f"  Avg P&L/Trade:    {metrics['avg_pnl_per_trade']:.2f}")
    print(f"  Max Drawdown:     {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
    
    # Show first few trades
    if result.trades:
        print(f"\nFirst few trades:")
        for i, trade in enumerate(result.trades[:3]):
            print(f"  Trade {i+1}: {trade.direction} {trade.pnl_abs:.2f} USDT "
                  f"({trade.pnl_pct*100:.2f}%) over {trade.bars_in_trade} bars")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("data/CAKEUSDT.csv"):
        print("Error: Please run this script from the engine/ root directory")
        print("Current directory:", os.getcwd())
        sys.exit(1)
    
    quick_test()