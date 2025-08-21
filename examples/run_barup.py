"""
Example script demonstrating how to run the BarUp strategy backtest.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.data import load_klines_csv
from engine.backtest import run_backtest
from engine.metrics import compute_all_metrics, print_metrics_summary
from engine.plot import plot_pnl, plot_trade_analysis, save_plot, show_plot
from engine.trades import build_trade_table
from strategies.barup_example import create_basic_barup, create_barup_with_stops


def run_barup_example():
    """Run a complete BarUp strategy backtest example."""
    
    print("=" * 60)
    print("BARUP STRATEGY BACKTEST EXAMPLE")
    print("=" * 60)
    
    # 1. Load data
    print("\n1. Loading data...")
    data_path = "data/CAKEUSDT.csv"
    
    try:
        data, frequency = load_klines_csv(data_path)
        print(f"   Loaded {len(data)} bars of {frequency} data")
        print(f"   Date range: {data.iloc[0]['dt_open']} to {data.iloc[-1]['dt_close']}")
    except FileNotFoundError:
        print(f"   Error: Data file not found at {data_path}")
        print("   Please ensure CAKEUSDT.csv is in the data/ directory")
        return
    except Exception as e:
        print(f"   Error loading data: {e}")
        return
    
    # 2. Create strategy
    print("\n2. Creating strategy...")
    strategy = create_basic_barup(
        size_value=1000,  # $1000 per trade
        size_mode="notional",
        max_bars=50,  # Max 50 bars per trade
        min_bars=2    # Min 2 bars before exit
    )
    print(f"   Strategy: BarUp with $1000 trades, max 50 bars, min 2 bars")
    
    # 3. Run backtest
    print("\n3. Running backtest...")
    result = run_backtest(
        data=data,
        strategy=strategy,
        symbol="CAKEUSDT",
        initial_cash=10000,  # $10,000 starting capital
        fee_bps=10,          # 0.1% fees
        slippage_bps=5,      # 0.05% slippage
        currency_label="USDT",
        random_seed=42
    )
    print(f"   Backtest completed: {len(result.trades)} trades executed")
    
    # 4. Calculate metrics
    print("\n4. Computing metrics...")
    metrics = compute_all_metrics(result)
    
    # 5. Display results
    print("\n5. Results Summary:")
    print_metrics_summary(metrics, currency_label="USDT")
    
    # 6. Show trade table (first 10 trades)
    if result.trades:
        print("\n6. Trade Details (First 5 trades):")
        print("-" * 100)
        
        trade_table = build_trade_table(result.trades[:5], None, "USDT")
        
        # Print table headers
        if trade_table:
            headers = trade_table[0].keys()
            header_line = " | ".join(f"{h:>12}" for h in headers)
            print(header_line)
            print("-" * len(header_line))
            
            # Print first few rows
            for i, row in enumerate(trade_table[:15]):  # Show first 5 trades (3 rows each)
                row_line = " | ".join(f"{str(v):>12}" for v in row.values())
                print(row_line)
                
                # Add separator after each trade (every 3 rows)
                if (i + 1) % 3 == 0 and i < len(trade_table) - 1:
                    print("-" * len(header_line))
        
        print(f"\n   Showing first 5 trades. Total trades: {len(result.trades)}")
    
    # 7. Generate plots
    print("\n7. Generating plots...")
    try:
        # P&L plot
        fig1 = plot_pnl(result, mode="percent", show_drawdown=True)
        if fig1:
            print("   - P&L percentage plot created")
            save_plot(fig1, "barup_pnl_percent.png")
            print("   - Saved as barup_pnl_percent.png")
        
        # Trade analysis plot
        if result.trades:
            fig2 = plot_trade_analysis(result)
            if fig2:
                print("   - Trade analysis plot created")
                save_plot(fig2, "barup_trade_analysis.png")
                print("   - Saved as barup_trade_analysis.png")
        
        print("   - Use matplotlib to display plots if needed")
        
    except ImportError:
        print("   - Matplotlib not available, skipping plots")
        print("   - Install with: pip install matplotlib")
    
    # 8. Final summary
    print("\n8. Backtest Complete!")
    print(f"   Final Equity: ${result.final_equity:,.2f}")
    print(f"   Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Total Trades: {metrics['total_trades']}")
    print(f"   Win Rate: {metrics['percent_profitable']:.1f}%")


def compare_strategies():
    """Compare different BarUp strategy variants."""
    
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    
    # Load data once
    data, _ = load_klines_csv("data/CAKEUSDT.csv")
    
    strategies = {
        "Basic BarUp": create_basic_barup(size_value=1000, max_bars=0),
        "BarUp with Time Stop": create_basic_barup(size_value=1000, max_bars=20),
        "BarUp with Stop Loss": create_barup_with_stops(
            size_value=1000, stop_loss_pct=0.03, max_bars=30
        )
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"\nRunning {name}...")
        result = run_backtest(
            data=data,
            strategy=strategy,
            symbol="CAKEUSDT",
            initial_cash=10000,
            fee_bps=10,
            slippage_bps=5,
            currency_label="USDT",
            random_seed=42
        )
        results[name] = result
    
    # Compare results
    print(f"\n{'Strategy':<20} {'Return%':<10} {'Trades':<8} {'Win%':<8} {'Sharpe':<8} {'MaxDD%':<8}")
    print("-" * 70)
    
    for name, result in results.items():
        metrics = compute_all_metrics(result)
        print(f"{name:<20} "
              f"{metrics['total_return_pct']:>8.2f}% "
              f"{metrics['total_trades']:>6d} "
              f"{metrics['percent_profitable']:>6.1f}% "
              f"{metrics['sharpe_ratio']:>6.2f} "
              f"{metrics['max_drawdown_pct']:>7.2f}%")


if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("data/CAKEUSDT.csv"):
        print("Error: Please run this script from the engine/ root directory")
        print("Current directory:", os.getcwd())
        sys.exit(1)
    
    # Run main example
    run_barup_example()
    
    # Optionally run strategy comparison
    try:
        compare_input = input("\nRun strategy comparison? (y/n): ").strip().lower()
        if compare_input in ['y', 'yes']:
            compare_strategies()
    except EOFError:
        # Handle case where input is not available (e.g., automated runs)
        pass
    
    print("\nExample completed!")