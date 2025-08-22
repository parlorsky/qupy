"""
Main backtesting engine and execution loop.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

from .strategy_base import Strategy, Bar
from .context import Context
from .fills import FillEngine, FillConfig, PositionManager, Fill
from .trades import TradeTracker, Trade, finalize_trade
from .data import validate_klines_data


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    
    # Configuration
    symbol: str
    initial_cash: float
    currency_label: str
    start_time: datetime
    end_time: datetime
    
    # Performance data
    equity_curve: pd.Series  # Equity over time
    returns_series: pd.Series  # Bar-by-bar returns
    
    # Trades
    trades: List[Trade]
    open_trades: List[Trade]
    
    # Final state
    final_cash: float
    final_equity: float
    final_position: Any
    
    # Costs
    total_fees_paid: float
    total_slippage_cost: float
    
    # Raw data for analysis
    data: pd.DataFrame
    frequency: str


class BacktestEngine:
    """
    Main backtesting engine that orchestrates strategy execution.
    """
    
    def __init__(self, data: pd.DataFrame, strategy: Strategy, symbol: str = "SYMBOL",
                 initial_cash: float = 100000.0, currency_label: str = "usp",
                 fill_config: Optional[FillConfig] = None):
        
        # Validate data
        validate_klines_data(data)
        
        self.data = data.copy()
        self.strategy = strategy
        self.symbol = symbol
        self.currency_label = currency_label
        
        # Initialize subsystems
        self.position_manager = PositionManager(initial_cash)
        self.fill_engine = FillEngine(fill_config or FillConfig())
        self.trade_tracker = TradeTracker(symbol)
        
        # State tracking
        self.current_idx = 0
        self.current_bar = None
        self.context = Context(self, symbol)
        
        # Performance tracking
        self.equity_curve = []
        self.equity_timestamps = []
        self.returns_series = []
        
        # MFE/MAE tracking for open trades
        self._mfe_mae_trackers = {}  # trade_id -> (max_favorable, max_adverse)
    
    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self.position_manager.cash
    
    @property
    def position(self):
        """Current position."""
        class SimplePosition:
            def __init__(self, qty, avg_price):
                self.qty = qty
                self.avg_price = avg_price
                
            @property
            def direction(self):
                if self.qty > 0:
                    return "long"
                elif self.qty < 0:
                    return "short"
                else:
                    return "flat"
        
        return SimplePosition(
            qty=self.position_manager.position_qty,
            avg_price=self.position_manager.position_avg_price
        )
    
    @property
    def equity(self) -> float:
        """Current equity."""
        if self.current_bar is not None:
            mark_price = self.current_bar['close']
            return self.position_manager.get_equity(mark_price)
        return self.position_manager.cash
    
    def run(self) -> BacktestResult:
        """
        Run the complete backtest.
        
        Returns:
            BacktestResult with all performance metrics and data
        """
        print(f"Starting backtest: {len(self.data)} bars, symbol={self.symbol}")
        
        # Initialize
        self.strategy.on_init(self.context)
        self.strategy.on_start(self.context)
        
        # Main loop
        for idx in range(len(self.data)):
            self.current_idx = idx
            self.current_bar = self.data.iloc[idx]
            
            self._process_bar()
        
        # Finalize
        self.strategy.on_stop(self.context)
        self._finalize_backtest()
        
        return self._build_result()
    
    def _process_bar(self):
        """Process a single bar of data."""
        bar = self.current_bar
        idx = self.current_idx
        
        # 1. Process any pending fills
        fills = self.fill_engine.process_fills(idx, bar)
        self._apply_fills(fills)
        
        # 2. Update equity curve (before strategy to capture fill effects)
        current_equity = self.equity
        self.equity_curve.append(current_equity)
        self.equity_timestamps.append(bar['dt_close'])
        
        # Calculate returns
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]
            if prev_equity > 0:
                ret = (current_equity - prev_equity) / prev_equity
            else:
                ret = 0.0
            self.returns_series.append(ret)
        
        # 3. Update MFE/MAE for open trades
        self._update_mfe_mae()
        
        # 4. Convert to Bar dataclass and call strategy
        bar_obj = Bar(
            dt_open=bar['dt_open'],
            dt_close=bar['dt_close'],
            open=bar['open'],
            high=bar['high'],
            low=bar['low'],
            close=bar['close'],
            volume=bar['volume']
        )
        
        # Update context state
        self.context._update_state(idx, bar_obj)
        
        # Call strategy
        self.strategy.on_bar(self.context, self.symbol, bar_obj)
        
        # 5. Submit new orders
        new_orders = self.context.get_orders()
        if new_orders:
            self.fill_engine.submit_orders(new_orders, idx)
    
    def _apply_fills(self, fills: List[Fill]):
        """Apply fills to position and track trades."""
        for fill in fills:
            # Apply to position
            success, msg = self.position_manager.apply_fill(fill)
            if not success:
                print(f"Warning: Fill rejected - {msg}")
                continue
            
            self._track_trade_from_fill(fill)
    
    def _track_trade_from_fill(self, fill: Fill):
        """Track trades based on fills."""
        side = fill.order.side
        current_position_qty = self.position_manager.position_qty
        
        # Determine if this is opening, closing, or flipping
        if side == "buy":
            if current_position_qty > 0:
                # Opening or adding to long position
                if len(self.trade_tracker.get_open_trades()) == 0:
                    # New long trade
                    trade_id = self.trade_tracker.open_trade(
                        direction="Long",
                        entry_time=fill.fill_time,
                        entry_price=fill.fill_price,
                        qty=fill.fill_qty,
                        entry_signal=fill.order.reason,
                        entry_idx=fill.fill_idx,
                        fees=fill.fees_paid,
                        slippage=fill.slippage_cost
                    )
                    self._mfe_mae_trackers[trade_id] = (0.0, 0.0)
            else:
                # Closing short or flipping to long
                open_trades = self.trade_tracker.get_open_trades()
                if open_trades:
                    # Close existing short
                    trade = open_trades[0]  # Assume single position for now
                    closed_trade = self.trade_tracker.close_trade(
                        trade.trade_id,
                        exit_time=fill.fill_time,
                        exit_price=fill.fill_price,
                        exit_signal=fill.order.reason,
                        exit_idx=fill.fill_idx,
                        fees=fill.fees_paid,
                        slippage=fill.slippage_cost
                    )
                    if closed_trade:
                        self._finalize_closed_trade(closed_trade)
                        del self._mfe_mae_trackers[closed_trade.trade_id]
                
                # If we flipped to long, open new long trade
                if current_position_qty > 0:
                    trade_id = self.trade_tracker.open_trade(
                        direction="Long",
                        entry_time=fill.fill_time,
                        entry_price=fill.fill_price,
                        qty=abs(current_position_qty),
                        entry_signal=fill.order.reason,
                        entry_idx=fill.fill_idx
                    )
                    self._mfe_mae_trackers[trade_id] = (0.0, 0.0)
        
        else:  # sell
            if current_position_qty < 0:
                # Opening or adding to short position
                if len(self.trade_tracker.get_open_trades()) == 0:
                    # New short trade
                    trade_id = self.trade_tracker.open_trade(
                        direction="Short",
                        entry_time=fill.fill_time,
                        entry_price=fill.fill_price,
                        qty=fill.fill_qty,
                        entry_signal=fill.order.reason,
                        entry_idx=fill.fill_idx,
                        fees=fill.fees_paid,
                        slippage=fill.slippage_cost
                    )
                    self._mfe_mae_trackers[trade_id] = (0.0, 0.0)
            else:
                # Closing long or flipping to short
                open_trades = self.trade_tracker.get_open_trades()
                if open_trades:
                    # Close existing long
                    trade = open_trades[0]  # Assume single position for now
                    closed_trade = self.trade_tracker.close_trade(
                        trade.trade_id,
                        exit_time=fill.fill_time,
                        exit_price=fill.fill_price,
                        exit_signal=fill.order.reason,
                        exit_idx=fill.fill_idx,
                        fees=fill.fees_paid,
                        slippage=fill.slippage_cost
                    )
                    if closed_trade:
                        self._finalize_closed_trade(closed_trade)
                        del self._mfe_mae_trackers[closed_trade.trade_id]
                
                # If we flipped to short, open new short trade
                if current_position_qty < 0:
                    trade_id = self.trade_tracker.open_trade(
                        direction="Short",
                        entry_time=fill.fill_time,
                        entry_price=fill.fill_price,
                        qty=abs(current_position_qty),
                        entry_signal=fill.order.reason,
                        entry_idx=fill.fill_idx
                    )
                    self._mfe_mae_trackers[trade_id] = (0.0, 0.0)
    
    def _update_mfe_mae(self):
        """Update MFE/MAE trackers for open trades."""
        if not self._mfe_mae_trackers:
            return
        
        bar = self.current_bar
        high = bar['high']
        low = bar['low']
        
        for trade_id, (current_mfe, current_mae) in self._mfe_mae_trackers.items():
            trade = self.trade_tracker.open_trades.get(trade_id)
            if not trade:
                continue
            
            entry_price = trade.entry_price
            
            if trade.is_long:
                # Long: MFE = max gain (high - entry), MAE = max loss (entry - low)
                mfe = max(current_mfe, high - entry_price)
                mae = max(current_mae, entry_price - low)
            else:
                # Short: MFE = max gain (entry - low), MAE = max loss (high - entry)  
                mfe = max(current_mfe, entry_price - low)
                mae = max(current_mae, high - entry_price)
            
            self._mfe_mae_trackers[trade_id] = (mfe, mae)
    
    def _finalize_closed_trade(self, trade: Trade):
        """Finalize a closed trade with MFE/MAE data."""
        if trade.trade_id in self._mfe_mae_trackers:
            mfe, mae = self._mfe_mae_trackers[trade.trade_id]
            trade.mfe_abs = mfe
            trade.mae_abs = mae
            
            # Calculate percentages
            if trade.entry_price > 0:
                trade.mfe_pct = mfe / trade.entry_price
                trade.mae_pct = mae / trade.entry_price
        
        # Set cumulative P&L
        trade.cumulative_pnl_after_exit = self.position_manager.realized_pnl
    
    def _finalize_backtest(self):
        """Finalize backtest - close any open trades."""
        if self.trade_tracker.open_trades and len(self.data) > 0:
            last_bar = self.data.iloc[-1]
            last_price = last_bar['close']
            closed_trades = self.trade_tracker.force_close_all(
                exit_time=last_bar['dt_close'],
                exit_price=last_price,
                exit_signal="End of backtest",
                exit_idx=len(self.data) - 1
            )
            
            for trade in closed_trades:
                self._finalize_closed_trade(trade)
    
    def _build_result(self) -> BacktestResult:
        """Build final backtest result."""
        
        # Create series
        equity_series = pd.Series(
            self.equity_curve, 
            index=self.equity_timestamps,
            name='equity'
        )
        
        returns_series = pd.Series(
            self.returns_series,
            index=self.equity_timestamps[1:] if len(self.equity_timestamps) > 1 else [],
            name='returns'
        )
        
        return BacktestResult(
            symbol=self.symbol,
            initial_cash=self.position_manager.initial_cash,
            currency_label=self.currency_label,
            start_time=self.data.iloc[0]['dt_open'],
            end_time=self.data.iloc[-1]['dt_close'],
            equity_curve=equity_series,
            returns_series=returns_series,
            trades=self.trade_tracker.get_closed_trades(),
            open_trades=self.trade_tracker.get_open_trades(),
            final_cash=self.position_manager.cash,
            final_equity=self.equity,
            final_position=self.position,
            total_fees_paid=self.position_manager.total_fees_paid,
            total_slippage_cost=self.position_manager.total_slippage_cost,
            data=self.data,
            frequency=""  # Will be set by run_backtest function
        )


def run_backtest(data: pd.DataFrame, strategy: Strategy, symbol: str = "SYMBOL",
                 initial_cash: float = 100000, fee_bps: float = 0, slippage_bps: float = 0,
                 min_fee_abs: float = 0.0, currency_label: str = "usp", 
                 random_seed: Optional[int] = 42, pretty_results: int = 0) -> BacktestResult:
    """
    Convenience function to run a backtest with common parameters.
    
    Args:
        data: OHLCV data
        strategy: Strategy instance
        symbol: Symbol name
        initial_cash: Starting cash
        fee_bps: Fee rate in basis points
        slippage_bps: Slippage in basis points  
        min_fee_abs: Minimum absolute fee
        currency_label: Currency label for display
        random_seed: Random seed for reproducibility
        pretty_results: If 1, print formatted performance report
        
    Returns:
        BacktestResult
    """
    
    # Create fill config
    fill_config = FillConfig(
        fee_rate_bps=fee_bps,
        slippage_bps=slippage_bps,
        min_fee_abs=min_fee_abs,
        random_seed=random_seed
    )
    
    # Run backtest
    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        symbol=symbol,
        initial_cash=initial_cash,
        currency_label=currency_label,
        fill_config=fill_config
    )
    
    result = engine.run()
    
    # Generate comprehensive performance report
    report = generate_performance_report(result, data, fee_bps, slippage_bps)
    
    # Print pretty results if requested
    if pretty_results == 1:
        print_performance_report(report)
    
    return result, report


def generate_performance_report(result: BacktestResult, data: pd.DataFrame, 
                               fee_bps: float = 0, slippage_bps: float = 0) -> dict:
    """
    Generate comprehensive performance report in JSON format.
    
    Args:
        result: BacktestResult from backtest
        data: Original OHLCV data 
        fee_bps: Fee rate used
        slippage_bps: Slippage rate used
        
    Returns:
        Dictionary with comprehensive performance metrics
    """
    
    # Basic metrics
    total_return = (result.final_equity - result.initial_cash) / result.initial_cash
    
    # Trade analysis
    winning_trades = [t for t in result.trades if t.pnl_abs > 0]
    losing_trades = [t for t in result.trades if t.pnl_abs <= 0]
    
    win_rate = len(winning_trades) / len(result.trades) if result.trades else 0
    avg_win = sum(t.pnl_abs for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t.pnl_abs for t in losing_trades) / len(losing_trades) if losing_trades else 0
    
    # Risk metrics
    sharpe_ratio = None
    max_drawdown = None
    volatility = None
    
    if len(result.returns_series) > 1:
        returns_mean = result.returns_series.mean()
        returns_std = result.returns_series.std()
        volatility = returns_std
        
        if returns_std > 0:
            # Determine annualization factor based on data frequency
            time_diff = (data.iloc[1]['dt_close'] - data.iloc[0]['dt_close']).total_seconds() / 60
            if time_diff <= 1:  # 1min
                periods_per_year = 365 * 24 * 60
            elif time_diff <= 5:  # 5min
                periods_per_year = 365 * 24 * 12
            elif time_diff <= 15:  # 15min
                periods_per_year = 365 * 24 * 4
            elif time_diff <= 60:  # 1hour
                periods_per_year = 365 * 24
            elif time_diff <= 1440:  # 1day
                periods_per_year = 365
            else:
                periods_per_year = 252  # Default business days
                
            sharpe_ratio = (returns_mean / returns_std) * np.sqrt(periods_per_year)
        
        # Max drawdown
        equity_curve = result.equity_curve
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
    
    # Profit factor
    gross_profit = sum(t.pnl_abs for t in winning_trades) if winning_trades else 0
    gross_loss = abs(sum(t.pnl_abs for t in losing_trades)) if losing_trades else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    
    # Trade duration analysis
    if result.trades:
        durations = [t.bars_in_trade for t in result.trades if hasattr(t, 'bars_in_trade')]
        avg_duration = float(np.mean(durations)) if durations else None
        max_duration = int(np.max(durations)) if durations else None
        min_duration = int(np.min(durations)) if durations else None
    else:
        avg_duration = max_duration = min_duration = None
    
    # Buy & Hold comparison
    buy_hold_return = (data['close'].iloc[-1] / data['close'].iloc[0]) - 1
    
    # Generate comprehensive report
    report = {
        "strategy_info": {
            "symbol": result.symbol,
            "start_date": result.start_time.isoformat() if result.start_time else None,
            "end_date": result.end_time.isoformat() if result.end_time else None,
            "total_bars": len(data),
            "currency": result.currency_label
        },
        
        "performance": {
            "initial_cash": float(result.initial_cash),
            "final_equity": float(result.final_equity),
            "total_return_pct": round(float(total_return * 100), 2),
            "total_return_abs": float(result.final_equity - result.initial_cash),
            "buy_hold_return_pct": round(float(buy_hold_return * 100), 2),
            "excess_return_pct": round(float((total_return - buy_hold_return) * 100), 2)
        },
        
        "risk_metrics": {
            "sharpe_ratio": round(sharpe_ratio, 3) if sharpe_ratio is not None else None,
            "max_drawdown_pct": round(max_drawdown * 100, 2) if max_drawdown is not None else None,
            "volatility": round(volatility, 4) if volatility is not None else None,
            "var_95": None  # TODO: Implement VaR calculation
        },
        
        "trading_stats": {
            "total_trades": len(result.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate_pct": round(win_rate * 100, 1),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "∞",
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "largest_win": round(max(t.pnl_abs for t in result.trades), 2) if result.trades else 0,
            "largest_loss": round(min(t.pnl_abs for t in result.trades), 2) if result.trades else 0,
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2)
        },
        
        "trade_duration": {
            "avg_bars": round(avg_duration, 1) if avg_duration is not None else None,
            "max_bars": max_duration,
            "min_bars": min_duration
        },
        
        "costs": {
            "total_fees": round(result.total_fees_paid, 2),
            "total_slippage": round(result.total_slippage_cost, 2),
            "fee_rate_bps": fee_bps,
            "slippage_rate_bps": slippage_bps,
            "total_costs": round(result.total_fees_paid + result.total_slippage_cost, 2)
        },
        
        "summary": {
            "status": "Profitable" if total_return > 0 else "Loss",
            "beat_buy_hold": total_return > buy_hold_return,
            "risk_adjusted_performance": "Good" if sharpe_ratio and sharpe_ratio > 1.0 else "Poor" if sharpe_ratio and sharpe_ratio < 0 else "Moderate",
            "total_pnl": round(sum(t.pnl_abs for t in result.trades), 2) if result.trades else 0
        },
        
        # Add plotting data for interactive visualization
        "equity_curve": result.equity_curve.tolist() if hasattr(result.equity_curve, 'tolist') else list(result.equity_curve),
        "equity_timestamps": [str(ts) for ts in result.equity_curve.index] if hasattr(result.equity_curve, 'index') else None,
        "returns_series": result.returns_series.tolist() if hasattr(result.returns_series, 'tolist') else list(result.returns_series),
        "strategy_name": f"{result.symbol} Strategy"
    }
    
    # Convert numpy types to native Python types for JSON serialization
    def make_json_serializable(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    return make_json_serializable(report)


def print_performance_report(report: dict):
    """Print formatted performance report to console."""
    
    print(f"\n📊 STRATEGY PERFORMANCE REPORT")
    print("=" * 60)
    
    # Strategy Info
    print(f"🏷️  Symbol: {report['strategy_info']['symbol']}")
    print(f"📅 Period: {report['strategy_info']['start_date'][:10]} to {report['strategy_info']['end_date'][:10]}")
    print(f"📊 Total Bars: {report['strategy_info']['total_bars']:,}")
    
    # Performance
    print(f"\n💰 PERFORMANCE:")
    print(f"   Initial Cash: ${report['performance']['initial_cash']:,.2f}")
    print(f"   Final Equity: ${report['performance']['final_equity']:,.2f}")
    print(f"   Total Return: {report['performance']['total_return_pct']}%")
    print(f"   Buy & Hold: {report['performance']['buy_hold_return_pct']}%")
    print(f"   Excess Return: {report['performance']['excess_return_pct']}%")
    
    # Risk Metrics
    print(f"\n⚡ RISK METRICS:")
    if report['risk_metrics']['sharpe_ratio']:
        print(f"   Sharpe Ratio: {report['risk_metrics']['sharpe_ratio']}")
    if report['risk_metrics']['max_drawdown_pct']:
        print(f"   Max Drawdown: {report['risk_metrics']['max_drawdown_pct']}%")
    if report['risk_metrics']['volatility']:
        print(f"   Volatility: {report['risk_metrics']['volatility']:.4f}")
    
    # Trading Statistics
    print(f"\n📈 TRADING STATS:")
    print(f"   Total Trades: {report['trading_stats']['total_trades']}")
    if report['trading_stats']['total_trades'] > 0:
        print(f"   Win Rate: {report['trading_stats']['win_rate_pct']}%")
        print(f"   Profit Factor: {report['trading_stats']['profit_factor']}")
        print(f"   Average Win: ${report['trading_stats']['avg_win']}")
        print(f"   Average Loss: ${report['trading_stats']['avg_loss']}")
        print(f"   Largest Win: ${report['trading_stats']['largest_win']}")
        print(f"   Largest Loss: ${report['trading_stats']['largest_loss']}")
    
    # Costs
    print(f"\n💸 COSTS:")
    print(f"   Total Fees: ${report['costs']['total_fees']}")
    print(f"   Total Slippage: ${report['costs']['total_slippage']}")
    print(f"   Total Costs: ${report['costs']['total_costs']}")
    
    # Summary
    print(f"\n🎯 SUMMARY:")
    print(f"   Status: {report['summary']['status']}")
    print(f"   Beat Buy & Hold: {'✅ Yes' if report['summary']['beat_buy_hold'] else '❌ No'}")
    print(f"   Risk-Adjusted Performance: {report['summary']['risk_adjusted_performance']}")
    print(f"   Total P&L: ${report['summary']['total_pnl']}")


def run_backtest_json(data: pd.DataFrame, strategy: Strategy, symbol: str = "SYMBOL",
                     initial_cash: float = 100000, fee_bps: float = 0, slippage_bps: float = 0,
                     min_fee_abs: float = 0.0, currency_label: str = "usp", 
                     random_seed: Optional[int] = 42, pretty_results: int = 0) -> dict:
    """
    Run backtest and return only the JSON performance report.
    
    Args:
        Same as run_backtest()
        
    Returns:
        Performance report dictionary (JSON-serializable) with built-in plotting methods
    """
    result, report = run_backtest(data, strategy, symbol, initial_cash, 
                                 fee_bps, slippage_bps, min_fee_abs, 
                                 currency_label, random_seed, pretty_results)
    
    # Add built-in plotting methods to report
    _add_plotting_methods(report, data)
    
    return report


def _add_plotting_methods(report: dict, data: pd.DataFrame) -> None:
    """Add built-in plotting methods to the report."""
    try:
        from .plots import TradingPlots
        
        # Initialize plotter
        plotter = TradingPlots(theme="plotly_white", width=900, height=600)
        
        # Add equity curve plotting method
        def plot_equity(title: str = "Strategy Performance") -> object:
            """Generate interactive equity curve plot."""
            if report is None:
                print("No backtest results to plot")
                return None
            try:
                return plotter.equity_curve([report], title=title)
            except Exception as e:
                print(f"Error plotting equity curve: {e}")
                return None
        
        # Add buy & hold comparison method (backward compatibility)
        def plot_vs_buy_hold(title: str = "Strategy vs Buy & Hold") -> object:
            """Generate interactive comparison with buy & hold."""
            return plot_vs_strategy(benchmark_strategy="buy_hold", title=title)
        
        # Add flexible strategy comparison method
        def plot_vs_strategy(benchmark_strategy=None, benchmark_report=None, title: str = "Strategy Comparison") -> object:
            """
            Generate interactive comparison with any benchmark strategy.
            
            Args:
                benchmark_strategy: Strategy name ('buy_hold') or Strategy instance to compare against
                benchmark_report: Pre-computed report to compare against
                title: Chart title
                
            Returns:
                Plotly figure with both strategies
            """
            if report is None or data is None:
                print("No data available for comparison")
                return None
                
            try:
                benchmark_report_data = None
                
                # Handle pre-computed benchmark report
                if benchmark_report is not None:
                    benchmark_report_data = benchmark_report
                
                # Handle benchmark strategy
                elif benchmark_strategy == "buy_hold":
                    # Create buy & hold benchmark aligned with strategy timeline
                    initial_cash = report['performance']['initial_cash']
                    
                    # Get strategy equity length for alignment
                    strategy_equity_len = len(report.get('equity_curve', []))
                    
                    if strategy_equity_len > 0:
                        # Align buy & hold with strategy timeline (might be shorter due to warmup)
                        data_start_idx = len(data) - strategy_equity_len
                        aligned_close_prices = data['close'].iloc[data_start_idx:].reset_index(drop=True)
                        buy_hold_equity = (initial_cash * aligned_close_prices / aligned_close_prices.iloc[0]).tolist()
                    else:
                        # Fallback to full data
                        buy_hold_equity = (initial_cash * data['close'] / data['close'].iloc[0]).tolist()
                    
                    # Use the same timestamps as the main strategy, ensuring length matches
                    equity_timestamps = report.get('equity_timestamps', [])
                    if len(equity_timestamps) != len(buy_hold_equity):
                        # Fallback: create timestamps matching buy & hold equity length
                        if strategy_equity_len > 0:
                            equity_timestamps = [str(ts) for ts in data['dt_close'].iloc[data_start_idx:]]
                        else:
                            equity_timestamps = [str(ts) for ts in data['dt_close']]
                    
                    benchmark_report_data = {
                        'strategy_name': 'Buy & Hold',
                        'equity_curve': buy_hold_equity,
                        'equity_timestamps': equity_timestamps
                    }
                
                elif benchmark_strategy is not None:
                    # Run benchmark strategy
                    print(f"Running benchmark strategy: {benchmark_strategy}")
                    
                    # Import required classes
                    from engine.backtest import run_backtest_json
                    
                    # Handle string strategy names
                    if isinstance(benchmark_strategy, str):
                        if benchmark_strategy == "buy_hold":
                            # Already handled above
                            pass
                        else:
                            print(f"Unknown benchmark strategy: {benchmark_strategy}")
                            print("Available: 'buy_hold' or Strategy instance")
                            return None
                    else:
                        # Assume it's a Strategy instance
                        benchmark_result = run_backtest_json(
                            data=data,
                            strategy=benchmark_strategy,
                            symbol=report['strategy_info']['symbol'],
                            initial_cash=report['performance']['initial_cash'],
                            fee_bps=10,  # Use same fees as main strategy
                            slippage_bps=1,
                            pretty_results=0
                        )
                        
                        benchmark_report_data = benchmark_result
                
                else:
                    print("No benchmark specified. Use benchmark_strategy or benchmark_report parameter.")
                    return None
                
                if benchmark_report_data is None:
                    print("Failed to create benchmark data")
                    return None
                
                # Compare strategies using equity curves
                comparison_results = [report, benchmark_report_data]
                return plotter.equity_curve(comparison_results, title=title, show_drawdown=True)
                
            except Exception as e:
                print(f"Error plotting strategy comparison: {e}")
                import traceback
                print(f"Detailed error: {traceback.format_exc()}")
                return None
        
        # Add returns distribution method
        def plot_returns_distribution(title: str = "Returns Distribution") -> object:
            """Generate interactive returns distribution plot."""
            if report is None or 'returns_series' not in report:
                print("No returns data available")
                return None
            try:
                import pandas as pd
                returns_series = pd.Series(report['returns_series']) if report['returns_series'] else pd.Series([])
                if len(returns_series) == 0:
                    print("No returns data to plot")
                    return None
                return plotter.returns_distribution(
                    {"Strategy": returns_series},
                    add_var_lines=True,
                    confidence_levels=[0.95, 0.99]
                )
            except Exception as e:
                print(f"Error plotting returns distribution: {e}")
                return None
        
        # Add show_all method for comprehensive visualization
        def show_all() -> None:
            """Display all available interactive plots."""
            print("📊 Generating comprehensive interactive analysis...")
            
            # Equity curve
            fig1 = plot_equity()
            if fig1:
                fig1.show()
                print("✅ Interactive equity curve displayed")
            
            # Comparison with buy & hold
            fig2 = plot_vs_buy_hold()
            if fig2:
                fig2.show()
                print("✅ Strategy vs Buy & Hold comparison displayed")
            
            # Returns distribution
            fig3 = plot_returns_distribution()
            if fig3:
                fig3.show()
                print("✅ Returns distribution displayed")
            
            print("🎯 All interactive plots generated!")
        
        # Attach methods to report
        report['plot_equity'] = plot_equity
        report['plot_vs_buy_hold'] = plot_vs_buy_hold
        report['plot_vs_strategy'] = plot_vs_strategy
        report['plot_returns'] = plot_returns_distribution
        report['show_all_plots'] = show_all
        
    except ImportError:
        # If plotting is not available, add dummy methods
        def _no_plotting():
            print("Interactive plotting not available. Install plotly to enable plotting.")
            return None
        
        report['plot_equity'] = lambda title="": _no_plotting()
        report['plot_vs_buy_hold'] = lambda title="": _no_plotting()
        report['plot_vs_strategy'] = lambda benchmark_strategy=None, benchmark_report=None, title="": _no_plotting()
        report['plot_returns'] = lambda title="": _no_plotting()
        report['show_all_plots'] = lambda: _no_plotting()


class StrategyOptimizer:
    """
    Standardized parameter optimization utilities for strategies.
    
    This moves all the optimization logic from notebooks into the engine.
    """
    
    def __init__(self, data: pd.DataFrame, symbol: str = "SYMBOL", 
                 initial_cash: float = 100000.0, fee_bps: float = 10, slippage_bps: float = 2):
        self.data = data
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self._cache = {}  # Cache for storing backtest results
        
    def optimize_parameters(self, strategy_class, param_ranges: Dict[str, List], 
                           optimization_metric: str = "sharpe_ratio",
                           n_best: int = 5, verbose: bool = True, pretty_results: int = 0) -> pd.DataFrame:
        """
        Run parameter optimization for a strategy.
        
        Args:
            strategy_class: Strategy class to optimize
            param_ranges: Dict of parameter name -> list of values to test
            optimization_metric: Metric to optimize for ('sharpe_ratio', 'total_return_pct', 'profit_factor')
            n_best: Number of best results to return
            verbose: Whether to print progress
            
        Returns:
            DataFrame with optimization results
        """
        import itertools
        
        # Generate all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        combinations = list(itertools.product(*param_values))
        
        results = []
        cache_hits = 0
        new_runs = 0
        
        if verbose:
            print(f"🔄 Testing {len(combinations)} parameter combinations...")
            print(f"📊 Optimizing for: {optimization_metric}")
            if self._cache:
                print(f"💾 Using cache with {len(self._cache)} existing results")
        
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            # Create cache key from strategy class name and parameters
            cache_key = f"{strategy_class.__name__}_{hash(frozenset(params.items()))}"
            
            try:
                # Check cache first
                if cache_key in self._cache:
                    cache_hits += 1
                    report = self._cache[cache_key]
                else:
                    new_runs += 1
                    
                    strategy = strategy_class(params)
                    report = run_backtest_json(
                        data=self.data,
                        strategy=strategy,
                        symbol=self.symbol,
                        initial_cash=self.initial_cash,
                        fee_bps=self.fee_bps,
                        slippage_bps=self.slippage_bps,
                        pretty_results=0  # Always suppress output during optimization
                    )
                    
                    # Store in cache
                    self._cache[cache_key] = report
                
                # Progress update
                if verbose and (i + 1) % 10 == 0:
                    print(f"   Progress: {i+1}/{len(combinations)} ({cache_hits} cached, {new_runs} new)")
                
                # Extract key metrics
                result = {
                    **params,
                    'total_return_pct': report['performance']['total_return_pct'],
                    'sharpe_ratio': report['risk_metrics']['sharpe_ratio'] or 0,
                    'max_drawdown_pct': report['risk_metrics']['max_drawdown_pct'] or 0,
                    'profit_factor': report['trading_stats']['profit_factor'] if report['trading_stats']['profit_factor'] != "∞" else 999,
                    'win_rate_pct': report['trading_stats']['win_rate_pct'],
                    'total_trades': report['trading_stats']['total_trades'],
                    'total_costs': report['costs']['total_costs'],
                    'beat_buy_hold': report['summary']['beat_buy_hold']
                }
                
                results.append(result)
                
                if verbose and (i + 1) % max(1, len(combinations) // 10) == 0:
                    progress = (i + 1) / len(combinations) * 100
                    print(f"   Progress: {progress:.0f}% ({i + 1}/{len(combinations)})")
                    
            except Exception as e:
                if verbose:
                    print(f"   ❌ Error with params {params}: {e}")
                continue
        
        if verbose:
            print(f"✅ Optimization completed: {cache_hits} cached, {new_runs} new runs")
            
        if not results:
            return pd.DataFrame()
            
        df = pd.DataFrame(results)
        
        # Sort by optimization metric
        ascending = optimization_metric in ['max_drawdown_pct']  # Lower is better
        df_sorted = df.sort_values(optimization_metric, ascending=ascending)
        
        if verbose:
            print(f"\n🎯 Optimization completed! Found {len(results)} valid combinations.")
            print(f"\n🏆 Top {n_best} parameter combinations (by {optimization_metric}):")
            print("=" * 80)
            
            for i, (_, row) in enumerate(df_sorted.head(n_best).iterrows()):
                param_str = ", ".join([f"{k}={row[k]}" for k in param_names])
                print(f"{i+1}. {param_str} | {optimization_metric}={row[optimization_metric]:.3f} | Return={row['total_return_pct']:.1f}% | DD={row['max_drawdown_pct']:.1f}%")
        
        return df_sorted
    
    def compare_strategies(self, strategies_with_params: List[tuple], 
                          strategy_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple strategies with their parameters.
        
        Args:
            strategies_with_params: List of (strategy_class, params_dict) tuples
            strategy_names: Optional list of names for strategies
            
        Returns:
            DataFrame with comparison results
        """
        if strategy_names is None:
            strategy_names = [f"Strategy_{i+1}" for i in range(len(strategies_with_params))]
            
        results = []
        
        print(f"🔄 Comparing {len(strategies_with_params)} strategies...")
        
        for i, (strategy_class, params) in enumerate(strategies_with_params):
            name = strategy_names[i]
            
            try:
                strategy = strategy_class(params)
                report = run_backtest_json(
                    data=self.data,
                    strategy=strategy,
                    symbol=self.symbol,
                    initial_cash=self.initial_cash,
                    fee_bps=self.fee_bps,
                    slippage_bps=self.slippage_bps
                )
                
                result = {
                    'Strategy': name,
                    'Total Return %': report['performance']['total_return_pct'],
                    'vs Buy&Hold %': report['performance']['excess_return_pct'],
                    'Sharpe Ratio': report['risk_metrics']['sharpe_ratio'] or 0,
                    'Max DD %': report['risk_metrics']['max_drawdown_pct'] or 0,
                    'Win Rate %': report['trading_stats']['win_rate_pct'],
                    'Total Trades': report['trading_stats']['total_trades'],
                    'Profit Factor': report['trading_stats']['profit_factor'] if report['trading_stats']['profit_factor'] != "∞" else 999,
                    'Total Costs $': report['costs']['total_costs'],
                    'Beat B&H': '✅' if report['summary']['beat_buy_hold'] else '❌',
                    'Status': report['summary']['status']
                }
                
                results.append(result)
                print(f"✅ {name}: {result['Total Return %']:.1f}% return, Sharpe {result['Sharpe Ratio']:.2f}")
                
            except Exception as e:
                print(f"❌ Error testing {name}: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
            
        df = pd.DataFrame(results)
        return df.sort_values('Sharpe Ratio', ascending=False)
    
    def plot_optimization_heatmap(self, optimization_results: pd.DataFrame, 
                                 param1: str, param2: str, metric: str = "sharpe_ratio") -> object:
        """Create interactive parameter optimization heatmap."""
        try:
            from .plots import TradingPlots
            plotter = TradingPlots(theme="plotly_white", width=800, height=600)
            return plotter.optimization_heatmap(optimization_results, param1, param2, metric)
        except ImportError:
            print("Plotly not available for optimization heatmap")
            return None
    
    def plot_optimization_strategies(self, optimization_results: pd.DataFrame, 
                                   strategy_class, n_strategies: int = 5, 
                                   title: str = "Top Strategies Comparison") -> object:
        """Create equity curve comparison plot for top strategies."""
        try:
            from .plots import TradingPlots
            plotter = TradingPlots(theme="plotly_white", width=1000, height=600)
            
            # Get top strategies
            top_strategies = optimization_results.head(n_strategies)
            
            # Run backtests for each top strategy to get equity curves
            strategy_reports = []
            
            print(f"🔄 Generating equity curves for top {n_strategies} strategies...")
            
            for i, row in top_strategies.iterrows():
                # Extract parameters (exclude metric columns)
                param_cols = [col for col in row.index if col not in 
                             ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'profit_factor', 
                              'win_rate_pct', 'total_trades', 'total_costs', 'beat_buy_hold']]
                params = {col: row[col] for col in param_cols}
                
                try:
                    # Re-run backtest to get full equity curve data
                    strategy = strategy_class(params)
                    report = run_backtest_json(
                        data=self.data,
                        strategy=strategy,
                        symbol=self.symbol,
                        initial_cash=self.initial_cash,
                        fee_bps=self.fee_bps,
                        slippage_bps=self.slippage_bps,
                        pretty_results=0  # Suppress output
                    )
                    
                    # Create strategy name from parameters
                    param_str = ", ".join([f"{k}={v}" for k, v in params.items() if k != 'notional'])
                    strategy_name = f"Strategy #{i+1}: {param_str}"
                    report['strategy_name'] = strategy_name
                    
                    strategy_reports.append(report)
                    print(f"✅ {strategy_name}: {report['performance']['total_return_pct']:.1f}% return")
                    
                except Exception as e:
                    print(f"❌ Error running strategy {i+1}: {e}")
                    continue
            
            if strategy_reports:
                # Create equity curve comparison
                fig = plotter.equity_curve(strategy_reports, title=title, show_drawdown=True)
                print(f"✅ Equity curves plot created for {len(strategy_reports)} strategies!")
                return fig
            else:
                print("❌ No valid strategy reports generated")
                return None
                
        except Exception as e:
            print(f"Error creating optimization strategies plot: {e}")
            return None
    
    def show_optimization_results(self, optimization_results: pd.DataFrame, pretty_results: int = 1):
        """Display optimization results with optional formatting."""
        if pretty_results == 1:
            print("\n" + "="*70)
            print("🎯 PARAMETER OPTIMIZATION RESULTS")
            print("="*70)
            
            if not optimization_results.empty:
                best = optimization_results.iloc[0]
                print(f"\n🏆 BEST PARAMETERS:")
                
                # Extract parameter columns
                param_cols = [col for col in best.index if col not in 
                             ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'profit_factor', 
                              'win_rate_pct', 'total_trades', 'total_costs', 'beat_buy_hold']]
                              
                for param in param_cols:
                    print(f"   {param}: {best[param]}")
                    
                print(f"\n📊 PERFORMANCE METRICS:")
                print(f"   Sharpe Ratio: {best['sharpe_ratio']:.3f}")
                print(f"   Total Return: {best['total_return_pct']:.1f}%")
                print(f"   Max Drawdown: {best['max_drawdown_pct']:.1f}%")
                print(f"   Win Rate: {best['win_rate_pct']:.1f}%")
                print(f"   Total Trades: {int(best['total_trades'])}")
                print(f"   Beat Buy & Hold: {'✅ Yes' if best['beat_buy_hold'] else '❌ No'}")
                
                print(f"\n📈 TOP 5 COMBINATIONS:")
                display_cols = param_cols + ['sharpe_ratio', 'total_return_pct', 'max_drawdown_pct']
                print(optimization_results[display_cols].head().to_string(index=False))
                
            else:
                print("❌ No optimization results available")
                
            print("="*70)
        
        return optimization_results
    
    def clear_cache(self):
        """Clear the optimization cache."""
        self._cache.clear()
        print("✅ Optimization cache cleared")
    
    def cache_info(self):
        """Display information about the current cache."""
        print(f"📊 Cache contains {len(self._cache)} stored results")
        if self._cache:
            print("   Cached parameter combinations:")
            for i, key in enumerate(list(self._cache.keys())[:5]):  # Show first 5
                print(f"   {i+1}. {key}")
            if len(self._cache) > 5:
                print(f"   ... and {len(self._cache) - 5} more")


class StrategyComparison:
    """
    Utilities for comparing strategy performance with standardized visualization.
    """
    
    @staticmethod
    def create_comparison_table(results: Dict[str, dict]) -> pd.DataFrame:
        """Create a clean comparison table from multiple backtest results."""
        comparison_data = []
        
        for name, report in results.items():
            if report:
                comparison_data.append({
                    'Strategy': name,
                    'Return %': report['performance']['total_return_pct'],
                    'vs B&H %': report['performance'].get('excess_return_pct', 0),
                    'Sharpe': report['risk_metrics']['sharpe_ratio'] or 0,
                    'Max DD %': report['risk_metrics']['max_drawdown_pct'] or 0,
                    'Win Rate %': report['trading_stats']['win_rate_pct'],
                    'Trades': report['trading_stats']['total_trades'],
                    'Profit Factor': report['trading_stats']['profit_factor'] if report['trading_stats']['profit_factor'] != "∞" else 999,
                    'Costs $': report.get('costs', {}).get('total_costs', 0),
                    'Beat B&H': '✅' if report.get('summary', {}).get('beat_buy_hold', False) else '❌'
                })
        
        return pd.DataFrame(comparison_data).sort_values('Sharpe', ascending=False)
    
    @staticmethod
    def print_comparison_summary(results: Dict[str, dict]):
        """Print a standardized comparison summary."""
        print("\n" + "="*80)
        print("📊 STRATEGY COMPARISON SUMMARY")  
        print("="*80)

        if not results:
            print("No results to compare")
            return
            
        print(f"{'Strategy':<30} {'Return%':<8} {'Sharpe':<7} {'DD%':<6} {'Trades':<7} {'Win%':<6} {'Status':<12}")
        print("-" * 80)
        
        # Sort by Sharpe ratio
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1]['risk_metrics']['sharpe_ratio'] or 0 if x[1] else 0, 
                              reverse=True)
        
        for name, report in sorted_results:
            if report:
                print(f"{name:<30} "
                      f"{report['performance']['total_return_pct']:>7.1f}% "
                      f"{report['risk_metrics']['sharpe_ratio'] if report['risk_metrics']['sharpe_ratio'] else 0:>6.2f} "
                      f"{report['risk_metrics']['max_drawdown_pct'] if report['risk_metrics']['max_drawdown_pct'] else 0:>5.1f}% "
                      f"{report['trading_stats']['total_trades']:>6d} "
                      f"{report['trading_stats']['win_rate_pct']:>5.1f}% "
                      f"{report['summary']['status']:<12}")


def create_strategy_template(strategy_name: str, parameters: Dict[str, Dict]) -> str:
    """
    Generate a standardized strategy template with proper documentation.
    
    Args:
        strategy_name: Name for the strategy
        parameters: Dict of param_name -> {"type": "int", "min": 1, "max": 100, "default": 10, "description": "..."}
        
    Returns:
        Python code string for the strategy template
    """
    
    # Generate parameter schema
    param_schema_lines = []
    param_init_lines = []
    param_docs = []
    
    for param, config in parameters.items():
        # Schema line
        schema_parts = [f'"type": "{config["type"]}"']
        if "min" in config:
            schema_parts.append(f'"min": {config["min"]}')
        if "max" in config:
            schema_parts.append(f'"max": {config["max"]}')
        schema_parts.append(f'"default": {config["default"]}')
        
        param_schema_lines.append(f'        "{param}": {{{", ".join(schema_parts)}}}')
        
        # Init line
        if config["type"] == "int":
            param_init_lines.append(f'        self.{param} = int(self.params.get("{param}", {config["default"]}))')
        elif config["type"] == "float":
            param_init_lines.append(f'        self.{param} = float(self.params.get("{param}", {config["default"]}))')
        else:
            param_init_lines.append(f'        self.{param} = self.params.get("{param}", {config["default"]})')
        
        # Documentation
        desc = config.get("description", f"{param} parameter")
        if "min" in config and "max" in config:
            param_docs.append(f'    • {param} ({config["min"]}-{config["max"]}): {desc}')
        else:
            param_docs.append(f'    • {param}: {desc}')
    
    template = f'''@register_strategy("{strategy_name.lower()}")
class {strategy_name}Strategy(Strategy):
    """
    {strategy_name} Trading Strategy
    
    🔧 HYPERPARAMETERS TO CUSTOMIZE:
    
{chr(10).join(param_docs)}
      - Try different values within the specified ranges
      - Start with defaults and adjust based on performance
    
    📊 STRATEGY LOGIC:
    - TODO: Describe your entry conditions
    - TODO: Describe your exit conditions
    - TODO: Describe risk management rules
    """
    
    name = "{strategy_name.lower()}"
    
    @classmethod
    def param_schema(cls):
        return {{
{",".join([chr(10) + line for line in param_schema_lines])}
        }}
    
    def on_init(self, context):
        """Initialize strategy parameters"""
{chr(10).join(param_init_lines)}
        
        # TODO: Add parameter validation if needed
        context.log("info", f"{strategy_name} initialized with parameters: {{self.params}}")
    
    def on_start(self, context):
        """Called at backtest start"""
        context.log("info", f"{strategy_name} strategy started")
    
    def on_bar(self, context, symbol: str, bar: Bar):
        """Main trading logic called each bar"""
        # TODO: Add your strategy logic here
        
        # Example pattern:
        # 1. Check if we have enough data
        # if context.bar_index < self.some_period:
        #     return
        
        # 2. Get historical data
        # closes = context.data.history(symbol, "close", self.some_period)
        
        # 3. Calculate indicators
        # indicator_value = calculate_indicator(closes)
        
        # 4. Get current position
        # current_pos = context.position.qty
        
        # 5. Entry logic
        # if entry_condition and current_pos == 0:
        #     qty = context.size.from_notional(self.notional, bar.close)
        #     context.buy(qty, reason="Entry_Signal", size_mode="qty")
        
        # 6. Exit logic
        # if exit_condition and current_pos != 0:
        #     context.close(reason="Exit_Signal")
        
        # 7. Record metrics
        # context.record("indicator", indicator_value)
        
        pass
    
    def on_trade_open(self, context, trade):
        """Called when a new trade opens"""
        context.log("info", f"Trade opened: {{trade.trade_id}}")
    
    def on_trade_close(self, context, trade):
        """Called when a trade closes"""
        if hasattr(trade, 'pnl_abs') and trade.pnl_abs is not None:
            context.log("info", f"Trade closed: PnL=${{trade.pnl_abs:.2f}}")
            context.record("trade_pnl", trade.pnl_abs)

print("✅ {strategy_name}Strategy template created!")
print("📝 TODO: Implement the strategy logic in on_bar() method")
print("🔧 Parameters configured: {list(parameters.keys())}")
'''
    
    return template


def run_strategy_suite(data: pd.DataFrame, strategies_config: List[tuple], 
                      symbol: str = "SYMBOL", initial_cash: float = 100000,
                      fee_bps: float = 10, slippage_bps: float = 2, 
                      verbose: bool = True) -> Dict[str, dict]:
    """
    Run a suite of strategies and return standardized comparison.
    
    This is the main API for running multiple strategies with clean output.
    
    Args:
        data: OHLCV DataFrame
        strategies_config: List of (name, strategy_class, params) tuples
        symbol: Symbol name
        initial_cash: Starting cash
        fee_bps: Fee rate in basis points
        slippage_bps: Slippage rate in basis points
        verbose: Print progress and results
        
    Returns:
        Dictionary of {strategy_name: backtest_report}
    """
    results = {}
    
    if verbose:
        print("\n" + "="*80)
        print("🚀 RUNNING STRATEGY SUITE")
        print("="*80)
    
    for name, strategy_class, params in strategies_config:
        if verbose:
            print(f"\n📈 Testing: {name}")
            print("-" * 60)
        
        try:
            strategy = strategy_class(params)
            report = run_backtest_json(
                data=data,
                strategy=strategy,
                symbol=symbol,
                initial_cash=initial_cash,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps
            )
            
            results[name] = report
            
            if verbose:
                # Display clean results using the JSON report
                print(f"💰 Total Return: {report['performance']['total_return_pct']}%")
                print(f"📈 vs Buy & Hold: {report['performance']['excess_return_pct']}%")
                print(f"⚡ Sharpe Ratio: {report['risk_metrics']['sharpe_ratio']}")
                print(f"📉 Max Drawdown: {report['risk_metrics']['max_drawdown_pct']}%")
                print(f"🎯 Win Rate: {report['trading_stats']['win_rate_pct']}%")
                print(f"🔄 Total Trades: {report['trading_stats']['total_trades']}")
                print(f"💸 Total Costs: ${report['costs']['total_costs']}")
                print(f"✅ Status: {report['summary']['status']}")
            
        except Exception as e:
            if verbose:
                print(f"❌ Error testing {name}: {e}")
            results[name] = None
    
    if verbose:
        StrategyComparison.print_comparison_summary(results)
        print(f"\n💡 JSON reports available for detailed analysis")
        print("   Access via: results['strategy_name'] for full report")
    
    return results


# ================================
# INTERACTIVE PLOTTING INTEGRATION
# ================================

def add_plotting_to_results(results: Union[Dict, List[Dict]]) -> Union[Dict, List[Dict]]:
    """
    Add plotting methods to strategy results.
    
    Args:
        results: Single result dict or list of results
        
    Returns:
        Results with added plotting methods
    """
    from .plots import TradingPlots
    
    plotter = TradingPlots()
    
    def add_plot_methods(result_dict):
        """Add plotting methods to a single result dictionary."""
        if result_dict is None:
            return result_dict
            
        # Add plotting methods
        result_dict['plot_equity'] = lambda **kwargs: plotter.equity_curve([result_dict], **kwargs)
        result_dict['plot_returns'] = lambda **kwargs: plotter.returns_distribution(
            pd.Series(result_dict.get('returns_series', [])), **kwargs
        )
        return result_dict
    
    if isinstance(results, list):
        return [add_plot_methods(r) for r in results]
    else:
        return add_plot_methods(results)


class PlottingMixin:
    """Mixin class to add plotting capabilities to backtest results."""
    
    @staticmethod
    def plot_strategy_suite(results: Dict[str, Dict], **kwargs):
        """
        Create comprehensive plots for strategy suite results.
        
        Args:
            results: Results from run_strategy_suite()
            **kwargs: Additional plotting arguments
            
        Returns:
            Dictionary of plotly figures
        """
        from .plots import TradingPlots
        
        plotter = TradingPlots()
        plots = {}
        
        # Filter out None results
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            print("❌ No valid results to plot")
            return plots
        
        # Convert to list format for plotting
        results_list = []
        for name, result in valid_results.items():
            result_copy = result.copy()
            result_copy['strategy_name'] = name
            results_list.append(result_copy)
        
        # Strategy comparison plots
        plots['comparison_bar'] = plotter.strategy_comparison(
            results_list, metric="sharpe_ratio", chart_type="bar"
        )
        plots['comparison_scatter'] = plotter.strategy_comparison(
            results_list, chart_type="scatter"
        )
        plots['comparison_radar'] = plotter.strategy_comparison(
            results_list, chart_type="radar"
        )
        
        # Individual equity curves
        plots['equity_curves'] = plotter.equity_curve(results_list)
        
        # Returns distribution comparison
        returns_dict = {}
        for name, result in valid_results.items():
            if 'returns_series' in result:
                returns_dict[name] = pd.Series(result['returns_series'])
        
        if returns_dict:
            plots['returns_distribution'] = plotter.returns_distribution(returns_dict)
        
        return plots
    
    @staticmethod
    def plot_optimization_results(optimization_df: pd.DataFrame, 
                                param1: str, param2: str,
                                metric: str = "sharpe_ratio", **kwargs):
        """
        Plot parameter optimization results as heatmap.
        
        Args:
            optimization_df: DataFrame from StrategyOptimizer.optimize_parameters()
            param1: First parameter name (x-axis)
            param2: Second parameter name (y-axis)
            metric: Metric to visualize
            **kwargs: Additional plotting arguments
            
        Returns:
            Plotly heatmap figure
        """
        from .plots import TradingPlots
        
        plotter = TradingPlots()
        return plotter.optimization_heatmap(optimization_df, param1, param2, metric, **kwargs)
    
    @staticmethod  
    def plot_walk_forward_results(wf_results: pd.DataFrame, **kwargs):
        """
        Plot walk-forward analysis results.
        
        Args:
            wf_results: DataFrame with walk-forward results
            **kwargs: Additional plotting arguments
            
        Returns:
            Plotly figure with walk-forward analysis
        """
        from .plots import TradingPlots
        
        plotter = TradingPlots()
        return plotter.walk_forward_analysis(wf_results, **kwargs)
    
    @staticmethod
    def plot_monte_carlo_results(mc_results: pd.DataFrame, **kwargs):
        """
        Plot Monte Carlo simulation results.
        
        Args:
            mc_results: DataFrame with Monte Carlo results
            **kwargs: Additional plotting arguments
            
        Returns:
            Plotly figure with Monte Carlo analysis
        """
        from .plots import TradingPlots
        
        plotter = TradingPlots()
        return plotter.monte_carlo_analysis(mc_results, **kwargs)


# Add plotting utilities to the main classes
StrategyComparison.plot_strategy_suite = PlottingMixin.plot_strategy_suite
StrategyComparison.plot_optimization_results = PlottingMixin.plot_optimization_results
StrategyComparison.plot_walk_forward_results = PlottingMixin.plot_walk_forward_results
StrategyComparison.plot_monte_carlo_results = PlottingMixin.plot_monte_carlo_results


# Convenience plotting functions for quick access
def quick_plot_results(results: Union[Dict, List[Dict]], plot_type: str = "equity", **kwargs):
    """
    Quick plotting function for immediate visualization.
    
    Args:
        results: Strategy results from run_backtest_json or run_strategy_suite
        plot_type: Type of plot ('equity', 'comparison', 'returns')
        **kwargs: Additional plotting arguments
        
    Returns:
        Plotly figure
    """
    from .plots import TradingPlots, quick_equity_curve, quick_strategy_comparison, quick_returns_distribution
    
    if plot_type == "equity":
        return quick_equity_curve(results, **kwargs)
    elif plot_type == "comparison":
        if not isinstance(results, list):
            results = [results]
        return quick_strategy_comparison(results, **kwargs)  
    elif plot_type == "returns":
        if isinstance(results, dict) and 'returns_series' in results:
            returns_data = pd.Series(results['returns_series'])
        else:
            returns_data = results
        return quick_returns_distribution(returns_data, **kwargs)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Use 'equity', 'comparison', or 'returns'")


# Update run_strategy_suite to automatically add plotting capabilities
def run_strategy_suite_with_plots(data: pd.DataFrame, strategies_config: List[tuple],
                                symbol: str = "SYMBOL", initial_cash: float = 100000.0,
                                fee_bps: float = 10, slippage_bps: float = 2,
                                verbose: bool = True, add_plots: bool = True) -> dict:
    """
    Enhanced version of run_strategy_suite with automatic plotting integration.
    
    Args:
        Same as run_strategy_suite
        add_plots: Whether to add plotting methods to results
        
    Returns:
        Results dictionary with added plotting capabilities
    """
    results = run_strategy_suite(data, strategies_config, symbol, initial_cash, 
                               fee_bps, slippage_bps, verbose)
    
    if add_plots:
        # Add plotting methods to each result
        for name, result in results.items():
            if result is not None:
                result['plot_equity'] = lambda r=result, **kwargs: quick_plot_results(r, "equity", **kwargs)
                result['plot_returns'] = lambda r=result, **kwargs: quick_plot_results(r, "returns", **kwargs)
        
        # Add suite-level plotting methods
        results['plot_comparison'] = lambda **kwargs: StrategyComparison.plot_strategy_suite(results, **kwargs)
        results['plot_suite'] = results['plot_comparison']  # Alias
    
    return results