"""
Main backtesting engine and execution loop.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from .strategy import Strategy, Context, Position
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
    final_position: Position
    
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
        self.context = Context(self)
        
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
    def position(self) -> Position:
        """Current position."""
        return Position(
            qty=self.position_manager.position_qty,
            avg_price=self.position_manager.position_avg_price,
            direction="long" if self.position_manager.position_qty > 0 else 
                     "short" if self.position_manager.position_qty < 0 else "flat"
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
        
        # 4. Call strategy
        self.strategy.on_bar(self.context, bar)
        
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
                 random_seed: Optional[int] = 42) -> BacktestResult:
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
    
    return result