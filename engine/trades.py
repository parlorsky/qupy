"""
Trade representation and tracking for backtesting engine.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class Trade:
    """
    Represents a completed trade with entry and exit.
    """
    trade_id: str
    direction: str  # "Long" or "Short"
    symbol: str
    
    # Entry details
    entry_time: datetime
    entry_price: float
    entry_signal: str
    
    # Exit details
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_signal: Optional[str] = None
    
    # Trade sizing
    qty: float = 0.0  # Base quantity (always positive)
    notional_quote_at_entry: float = 0.0  # Quote currency notional
    
    # P&L metrics
    pnl_abs: float = 0.0  # Absolute P&L in quote currency
    pnl_pct: float = 0.0  # Percentage return
    
    # Trade duration
    bars_in_trade: int = 0
    
    # Maximum Favorable/Adverse Excursion
    mfe_abs: float = 0.0  # Max favorable excursion (absolute)
    mae_abs: float = 0.0  # Max adverse excursion (absolute)
    mfe_pct: float = 0.0  # Max favorable excursion (percent)
    mae_pct: float = 0.0  # Max adverse excursion (percent)
    
    # Additional tracking
    cumulative_pnl_after_exit: float = 0.0  # Total portfolio P&L after this trade
    fees_paid: float = 0.0  # Total fees for this trade
    slippage_cost: float = 0.0  # Total slippage costs
    
    # Internal tracking for MFE/MAE calculation
    _entry_idx: int = field(default=-1, repr=False)
    _exit_idx: int = field(default=-1, repr=False)
    _is_open: bool = field(default=True, repr=False)
    
    @property
    def is_open(self) -> bool:
        """Whether this trade is still open."""
        return self._is_open
    
    @property
    def is_closed(self) -> bool:
        """Whether this trade is closed."""
        return not self._is_open
    
    @property
    def is_long(self) -> bool:
        """Whether this is a long trade."""
        return self.direction == "Long"
    
    @property
    def is_short(self) -> bool:
        """Whether this is a short trade."""
        return self.direction == "Short"
    
    def close_trade(self, exit_time: datetime, exit_price: float, exit_signal: str, exit_idx: int) -> None:
        """
        Close the trade and calculate basic P&L.
        MFE/MAE should be calculated separately using finalize_trade().
        """
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_signal = exit_signal
        self._exit_idx = exit_idx
        self._is_open = False
        
        # Calculate bars in trade
        self.bars_in_trade = max(1, exit_idx - self._entry_idx + 1)
        
        # Calculate P&L
        if self.is_long:
            self.pnl_abs = self.qty * (exit_price - self.entry_price)
        else:
            self.pnl_abs = self.qty * (self.entry_price - exit_price)
        
        # Subtract fees and slippage
        self.pnl_abs -= (self.fees_paid + self.slippage_cost)
        
        # Calculate percentage return
        if self.notional_quote_at_entry > 0:
            self.pnl_pct = self.pnl_abs / self.notional_quote_at_entry
        else:
            self.pnl_pct = 0.0


def finalize_trade(trade: Trade, data_slice: pd.DataFrame) -> Trade:
    """
    Calculate MFE/MAE for a closed trade using OHLC data.
    
    Args:
        trade: Closed trade object
        data_slice: DataFrame slice from entry to exit (inclusive)
        
    Returns:
        Updated trade with MFE/MAE calculated
    """
    if trade.is_open:
        raise ValueError("Cannot finalize an open trade")
    
    if len(data_slice) == 0:
        return trade
    
    entry_price = trade.entry_price
    highs = data_slice['high'].values
    lows = data_slice['low'].values
    
    if trade.is_long:
        # For long trades: MFE is max gain (high - entry), MAE is max loss (entry - low)
        favorable_moves = highs - entry_price
        adverse_moves = entry_price - lows
        
        trade.mfe_abs = np.max(favorable_moves)
        trade.mae_abs = np.max(adverse_moves)
    else:
        # For short trades: MFE is max gain (entry - low), MAE is max loss (high - entry)
        favorable_moves = entry_price - lows
        adverse_moves = highs - entry_price
        
        trade.mfe_abs = np.max(favorable_moves)
        trade.mae_abs = np.max(adverse_moves)
    
    # Calculate percentages
    if entry_price > 0:
        trade.mfe_pct = trade.mfe_abs / entry_price
        trade.mae_pct = trade.mae_abs / entry_price
    
    return trade


def create_trade_id(trade_num: int, prefix: str = "y") -> str:
    """Generate trade ID in format: y_000001"""
    return f"{prefix}_{trade_num:06d}"


class TradeTracker:
    """
    Tracks active and completed trades during backtesting.
    """
    
    def __init__(self, symbol: str = ""):
        self.symbol = symbol
        self.trades: List[Trade] = []
        self.open_trades: Dict[str, Trade] = {}
        self._trade_counter = 0
    
    def open_trade(self, direction: str, entry_time: datetime, entry_price: float, 
                   qty: float, entry_signal: str, entry_idx: int, 
                   fees: float = 0.0, slippage: float = 0.0) -> str:
        """
        Open a new trade.
        
        Returns:
            Trade ID
        """
        self._trade_counter += 1
        trade_id = create_trade_id(self._trade_counter)
        
        trade = Trade(
            trade_id=trade_id,
            direction=direction,
            symbol=self.symbol,
            entry_time=entry_time,
            entry_price=entry_price,
            entry_signal=entry_signal,
            qty=qty,
            notional_quote_at_entry=qty * entry_price,
            fees_paid=fees,
            slippage_cost=slippage,
            _entry_idx=entry_idx
        )
        
        self.open_trades[trade_id] = trade
        self.trades.append(trade)
        
        return trade_id
    
    def close_trade(self, trade_id: str, exit_time: datetime, exit_price: float, 
                    exit_signal: str, exit_idx: int, fees: float = 0.0, 
                    slippage: float = 0.0) -> Optional[Trade]:
        """
        Close an existing trade.
        
        Returns:
            Closed trade or None if trade_id not found
        """
        if trade_id not in self.open_trades:
            return None
        
        trade = self.open_trades.pop(trade_id)
        trade.fees_paid += fees
        trade.slippage_cost += slippage
        trade.close_trade(exit_time, exit_price, exit_signal, exit_idx)
        
        return trade
    
    def get_open_trades(self) -> List[Trade]:
        """Get list of currently open trades."""
        return list(self.open_trades.values())
    
    def get_closed_trades(self) -> List[Trade]:
        """Get list of closed trades."""
        return [t for t in self.trades if t.is_closed]
    
    def get_all_trades(self) -> List[Trade]:
        """Get all trades (open and closed)."""
        return self.trades.copy()
    
    def force_close_all(self, exit_time: datetime, exit_price: float, 
                       exit_signal: str = "Force close", exit_idx: int = -1) -> List[Trade]:
        """
        Force close all open trades (e.g., at end of backtest).
        
        Returns:
            List of newly closed trades
        """
        closed_trades = []
        for trade_id in list(self.open_trades.keys()):
            trade = self.close_trade(trade_id, exit_time, exit_price, exit_signal, exit_idx)
            if trade:
                closed_trades.append(trade)
        
        return closed_trades


def build_trade_table(trades: List[Trade], cumulative_pnl_series: pd.Series, 
                      currency_label: str = "usp") -> List[Dict[str, Any]]:
    """
    Build a detailed trade table with Entry/Exit rows and Summary headers.
    
    Format matches the specification:
    - Summary header with trade_id, direction, and percentages
    - Entry row with entry details
    - Exit row with exit details and P&L metrics
    
    Args:
        trades: List of closed trades
        cumulative_pnl_series: Series of cumulative P&L over time
        currency_label: Currency label for display (e.g., "usp")
        
    Returns:
        List of dictionaries representing table rows
    """
    table_rows = []
    
    for trade in trades:
        if trade.is_open:
            continue
            
        # Summary header row
        summary_row = {
            'Type': 'Summary',
            'Trade ID': trade.trade_id,
            'Direction': trade.direction,
            'Date/Time': '',
            'Signal': f"Net P&L {trade.pnl_pct:.2%}, Run-up {trade.mfe_pct:.2%}, Drawdown {trade.mae_pct:.2%}",
            'Price': '',
            'Position Size': '',
            'Net P&L': f"{trade.pnl_abs:.2f} {currency_label}",
            'Run-up': f"{trade.mfe_abs:.2f}",
            'Drawdown': f"{trade.mae_abs:.2f}",
            'Cumulative P&L': f"{trade.cumulative_pnl_after_exit:.2f}"
        }
        table_rows.append(summary_row)
        
        # Entry row
        entry_row = {
            'Type': 'Entry',
            'Trade ID': trade.trade_id,
            'Direction': trade.direction,
            'Date/Time': trade.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Signal': trade.entry_signal,
            'Price': f"{trade.entry_price:.6f}",
            'Position Size': f"{trade.notional_quote_at_entry/1000:.2f}K {currency_label}",
            'Net P&L': '',
            'Run-up': '',
            'Drawdown': '',
            'Cumulative P&L': ''
        }
        table_rows.append(entry_row)
        
        # Exit row
        exit_row = {
            'Type': 'Exit',
            'Trade ID': trade.trade_id,
            'Direction': trade.direction,
            'Date/Time': trade.exit_time.strftime('%Y-%m-%d %H:%M:%S') if trade.exit_time else '',
            'Signal': trade.exit_signal or 'Close',
            'Price': f"{trade.exit_price:.6f}" if trade.exit_price else '',
            'Position Size': '',
            'Net P&L': f"{trade.pnl_abs:.2f} {currency_label}",
            'Run-up': f"{trade.mfe_abs:.2f}",
            'Drawdown': f"{trade.mae_abs:.2f}",
            'Cumulative P&L': f"{trade.cumulative_pnl_after_exit:.2f}"
        }
        table_rows.append(exit_row)
    
    return table_rows