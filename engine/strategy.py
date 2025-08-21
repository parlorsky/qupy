"""
Strategy base class and execution context for backtesting engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any, Dict
import pandas as pd
from datetime import datetime


@dataclass
class Position:
    """Current position state."""
    qty: float = 0.0  # Positive for long, negative for short
    avg_price: float = 0.0
    direction: str = "flat"  # "long", "short", "flat"
    
    @property
    def notional(self) -> float:
        """Notional value of the position."""
        return abs(self.qty * self.avg_price)
    
    @property
    def is_long(self) -> bool:
        return self.qty > 0
    
    @property
    def is_short(self) -> bool:
        return self.qty < 0
    
    @property
    def is_flat(self) -> bool:
        return self.qty == 0


@dataclass
class Order:
    """Order to be executed."""
    side: str  # "buy", "sell"
    size: float  # Positive number (quantity or notional depending on size_mode)
    size_mode: str  # "qty" or "notional"
    reason: str  # Signal description
    order_type: str = "market"  # Only market orders for now
    
    
class Context:
    """
    Context object passed to strategy methods.
    Provides access to current state and order placement methods.
    """
    
    def __init__(self, backtest_engine):
        self._engine = backtest_engine
        self._orders = []  # Orders placed this bar
        
    @property
    def position(self) -> Position:
        """Current position."""
        return self._engine.position
    
    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._engine.cash
    
    @property
    def equity(self) -> float:
        """Current equity (cash + position value)."""
        return self._engine.equity
    
    @property
    def now(self) -> datetime:
        """Current bar timestamp (close time by default)."""
        return self._engine.current_bar['dt_close']
    
    @property
    def current_bar(self) -> pd.Series:
        """Current bar data."""
        return self._engine.current_bar
    
    @property
    def data(self) -> pd.DataFrame:
        """Full dataset (read-only)."""
        return self._engine.data.copy()
    
    @property
    def bar_index(self) -> int:
        """Current bar index."""
        return self._engine.current_idx
    
    def buy(self, size: float, reason: str = "Buy signal", size_mode: str = "notional") -> None:
        """
        Place a buy order.
        
        Args:
            size: Order size (positive number)
            reason: Signal description
            size_mode: "qty" for base quantity or "notional" for quote currency amount
        """
        if size <= 0:
            raise ValueError("Order size must be positive")
            
        order = Order(
            side="buy",
            size=size,
            size_mode=size_mode,
            reason=reason
        )
        self._orders.append(order)
    
    def sell(self, size: float, reason: str = "Sell signal", size_mode: str = "notional") -> None:
        """
        Place a sell order.
        
        Args:
            size: Order size (positive number)
            reason: Signal description
            size_mode: "qty" for base quantity or "notional" for quote currency amount
        """
        if size <= 0:
            raise ValueError("Order size must be positive")
            
        order = Order(
            side="sell",
            size=size,
            size_mode=size_mode,
            reason=reason
        )
        self._orders.append(order)
    
    def close(self, reason: str = "Close position") -> None:
        """Close the current position."""
        if self.position.is_flat:
            return
            
        if self.position.is_long:
            self.sell(abs(self.position.qty), reason, size_mode="qty")
        else:
            self.buy(abs(self.position.qty), reason, size_mode="qty")
    
    def get_orders(self) -> list[Order]:
        """Get orders placed this bar and clear the list."""
        orders = self._orders.copy()
        self._orders.clear()
        return orders


class Strategy(ABC):
    """
    Base class for trading strategies.
    
    Implement on_start, on_bar, and on_stop methods to define strategy logic.
    """
    
    def __init__(self, **params):
        """Initialize strategy with parameters."""
        self.params = params
        self._setup_params()
    
    def _setup_params(self):
        """Override to set up strategy parameters."""
        pass
    
    @abstractmethod
    def on_start(self, context: Context) -> None:
        """
        Called once at the beginning of the backtest.
        
        Args:
            context: Strategy execution context
        """
        pass
    
    @abstractmethod
    def on_bar(self, context: Context, bar: pd.Series) -> None:
        """
        Called for each bar of data.
        
        Args:
            context: Strategy execution context
            bar: Current bar data with columns [dt_open, dt_close, open, high, low, close, volume, quote_volume]
        """
        pass
    
    def on_stop(self, context: Context) -> None:
        """
        Called once at the end of the backtest.
        Override if cleanup is needed.
        
        Args:
            context: Strategy execution context
        """
        pass


class FixedSizeStrategy(Strategy):
    """
    Base class for strategies with fixed position sizing.
    """
    
    def _setup_params(self):
        self.size_mode = self.params.get('size_mode', 'notional')
        self.size_value = self.params.get('size_value', 1000.0)
        
        if self.size_mode not in ['notional', 'qty']:
            raise ValueError("size_mode must be 'notional' or 'qty'")
        
        if self.size_value <= 0:
            raise ValueError("size_value must be positive")


class BuyAndHoldStrategy(Strategy):
    """
    Simple buy and hold strategy for testing.
    """
    
    def _setup_params(self):
        self.size_mode = self.params.get('size_mode', 'notional')
        self.size_value = self.params.get('size_value', 1000.0)
        self.bought = False
    
    def on_start(self, context: Context) -> None:
        pass
    
    def on_bar(self, context: Context, bar: pd.Series) -> None:
        if not self.bought and context.position.is_flat:
            context.buy(self.size_value, "Buy and hold", self.size_mode)
            self.bought = True
    
    def on_stop(self, context: Context) -> None:
        if not context.position.is_flat:
            context.close("End of backtest")