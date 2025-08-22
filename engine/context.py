"""
Context object providing strategy interface to backtesting engine.

The Context object is the main interface between strategies and the backtesting
engine. It provides safe data access with lookahead protection, order placement,
portfolio access, and utilities for logging and reproducibility.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import random
import logging
from abc import ABC, abstractmethod

from .strategy_base import Bar


@dataclass
class Position:
    """Current position information."""
    symbol: str
    qty: float
    avg_price: float
    direction: str  # "long", "short", or "flat"
    unrealized_pnl: float
    market_value: float


@dataclass
class Portfolio:
    """Portfolio-level information."""
    cash: float
    equity: float
    leverage: float
    exposures: Dict[str, float]  # symbol -> market value
    total_margin: float


@dataclass
class Order:
    """Order representation."""
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    order_type: str  # "market", "limit", "stop", etc.
    size: float
    price: Optional[float]
    stop_price: Optional[float]
    tif: str  # "GTC", "IOC", "FOK", "DAY"
    reason: str
    status: str  # "pending", "filled", "cancelled"
    created_at: Any
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0


@dataclass
class Fill:
    """Order fill information."""
    order_id: str
    symbol: str
    side: str
    qty: float
    price: float
    commission: float
    timestamp: Any


@dataclass
class Trade:
    """Completed trade information."""
    trade_id: str
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    exit_price: Optional[float]
    qty: float
    entry_time: Any
    exit_time: Optional[Any]
    pnl: Optional[float]
    commission: float
    tags: List[str]
    reason_open: str
    reason_close: Optional[str]


class DataAccessor:
    """Safe data accessor with lookahead protection."""
    
    def __init__(self, backtest_engine, current_index: int):
        self._backtest_engine = backtest_engine
        self._current_index = current_index
    
    def history(self, symbol: str, field: str, n: int) -> List[float]:
        """
        Get historical values for a field without lookahead.
        
        Args:
            symbol: Symbol to query
            field: Field name ("open", "high", "low", "close", "volume")
            n: Number of bars to retrieve (including current)
            
        Returns:
            List of values, most recent last
            
        Raises:
            ValueError: If trying to access future data
        """
        if n <= 0:
            raise ValueError("n must be positive")
            
        # Get data from BacktestEngine
        data = self._backtest_engine.data
        
        # Calculate start and end indices
        end_idx = self._current_index + 1  # Include current bar
        start_idx = max(0, end_idx - n)
        
        if start_idx >= len(data):
            return []
        
        # Extract the field
        if field not in data.columns:
            raise ValueError(f"Field '{field}' not found in data")
        
        values = data[field].iloc[start_idx:end_idx].tolist()
        return values
    
    def window(self, symbol: str, n: int) -> List[Bar]:
        """
        Get historical bars without lookahead.
        
        Args:
            symbol: Symbol to query
            n: Number of bars to retrieve (including current)
            
        Returns:
            List of Bar objects, most recent last
        """
        if n <= 0:
            raise ValueError("n must be positive")
            
        # Get data from BacktestEngine
        data = self._backtest_engine.data
        
        # Calculate start and end indices
        end_idx = self._current_index + 1  # Include current bar
        start_idx = max(0, end_idx - n)
        
        if start_idx >= len(data):
            return []
        
        # Convert to Bar objects
        bars = []
        for idx in range(start_idx, end_idx):
            row = data.iloc[idx]
            bar = Bar(
                dt_open=row['dt_open'],
                dt_close=row['dt_close'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            bars.append(bar)
        return bars
    
    def panel_history(self, symbols: List[str], field: str, n: int) -> Dict[str, List[float]]:
        """
        Get aligned historical data for multiple symbols.
        
        Args:
            symbols: List of symbols
            field: Field name
            n: Number of bars
            
        Returns:
            Dict mapping symbol -> list of values
        """
        return {sym: self.history(sym, field, n) for sym in symbols}


class SizeCalculator:
    """Utilities for position sizing."""
    
    def __init__(self, context):
        self._context = context
    
    @staticmethod
    def from_notional(notional_quote: float, price: float) -> float:
        """
        Calculate quantity from notional value.
        
        Args:
            notional_quote: Notional value in quote currency
            price: Price per unit
            
        Returns:
            Quantity in base units
        """
        if price <= 0:
            raise ValueError("Price must be positive")
        return abs(notional_quote) / price
    
    @staticmethod
    def fixed(qty: float) -> float:
        """
        Return fixed quantity.
        
        Args:
            qty: Quantity
            
        Returns:
            The same quantity
        """
        return abs(qty)
    
    def percent_equity(self, percent: float) -> float:
        """
        Calculate quantity based on percentage of equity.
        
        Args:
            percent: Percentage of equity (0-100)
            
        Returns:
            Notional value
        """
        if not 0 <= percent <= 100:
            raise ValueError("Percent must be between 0 and 100")
        return self._context.portfolio.equity * (percent / 100)


class Context:
    """
    Main interface between strategies and the backtesting engine.
    
    Provides safe data access, order placement, portfolio information,
    and utilities for logging and reproducibility.
    """
    
    def __init__(self, engine, symbol: str, seed: Optional[int] = None):
        self._engine = engine
        self._current_symbol = symbol
        self._current_index = 0
        self._current_bar = None
        self._logger = logging.getLogger(f"strategy.{symbol}")
        self._records = {}
        self._trade_tags = []
        self._orders = []  # Orders placed this bar
        
        # Set up deterministic RNG
        self._rng = random.Random(seed)
        
        # Initialize data accessor and size calculator
        self.data = DataAccessor(engine, self._current_index)
        self.size = SizeCalculator(self)
    
    # Current state properties
    
    @property
    def now(self):
        """Current bar timestamp."""
        if self._current_bar:
            return self._current_bar.dt_close
        return None
    
    @property
    def bar_index(self) -> int:
        """Numeric index into the data series."""
        return self._current_index
    
    @property
    def bar(self) -> Optional[Bar]:
        """Current bar object."""
        return self._current_bar
    
    @property
    def position(self) -> Position:
        """Current position for the symbol."""
        # Get position from BacktestEngine's position_manager
        pm = self._engine.position_manager
        
        # Create position object compatible with strategy expectations
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
        
        return SimplePosition(pm.position_qty, pm.position_avg_price)
    
    @property
    def portfolio(self) -> Portfolio:
        """Portfolio-level information."""
        # Create simplified portfolio object
        class SimplePortfolio:
            def __init__(self, engine):
                self.cash = engine.cash
                self.equity = engine.equity
        
        return SimplePortfolio(self._engine)
    
    @property
    def pending_orders(self) -> List[Order]:
        """List of pending orders for current symbol."""
        return self._engine.get_pending_orders(self._current_symbol)
    
    @property
    def fees_slippage(self) -> Dict[str, float]:
        """Current fee and slippage settings."""
        return self._engine.get_fees_slippage()
    
    @property
    def rand(self) -> random.Random:
        """Deterministic random number generator."""
        return self._rng
    
    # Order placement methods
    
    def buy(self, size: float, reason: str = "", size_mode: str = "notional") -> None:
        """
        Place market buy order.
        
        Args:
            size: Quantity or notional to buy (positive)
            reason: Optional reason for the trade
            size_mode: "qty" for base quantity or "notional" for quote currency amount
        """
        from .strategy import Order  # Import simple Order class
        
        if size <= 0:
            raise ValueError("Order size must be positive")
            
        order = Order(
            side="buy",
            size=size,
            size_mode=size_mode,
            reason=reason
        )
        self._orders.append(order)
    
    def sell(self, size: float, reason: str = "", size_mode: str = "notional") -> None:
        """
        Place market sell order.
        
        Args:
            size: Quantity or notional to sell (positive)
            reason: Optional reason for the trade
            size_mode: "qty" for base quantity or "notional" for quote currency amount
        """
        from .strategy import Order  # Import simple Order class
        
        if size <= 0:
            raise ValueError("Order size must be positive")
            
        order = Order(
            side="sell",
            size=size,
            size_mode=size_mode,
            reason=reason
        )
        self._orders.append(order)
    
    def close(self, reason: str = "") -> None:
        """
        Close current position with market order.
        
        Args:
            reason: Optional reason for closing
        """
        from .strategy import Order  # Import simple Order class
        
        pos = self.position
        if pos.qty == 0:
            return
            
        side = "sell" if pos.qty > 0 else "buy"
        
        order = Order(
            side=side,
            size=abs(pos.qty),
            size_mode="qty",
            reason=reason
        )
        self._orders.append(order)
    
    def limit_buy(self, price: float, size: float, tif: str = "GTC", reason: str = "") -> str:
        """Place limit buy order."""
        return self._engine.place_order(
            symbol=self._current_symbol,
            side="buy",
            order_type="limit",
            size=abs(size),
            price=price,
            tif=tif,
            reason=reason,
            tags=self._trade_tags.copy()
        )
    
    def limit_sell(self, price: float, size: float, tif: str = "GTC", reason: str = "") -> str:
        """Place limit sell order."""
        return self._engine.place_order(
            symbol=self._current_symbol,
            side="sell",
            order_type="limit",
            size=abs(size),
            price=price,
            tif=tif,
            reason=reason,
            tags=self._trade_tags.copy()
        )
    
    def stop_buy(self, stop: float, size: float, reason: str = "") -> str:
        """Place stop buy order."""
        return self._engine.place_order(
            symbol=self._current_symbol,
            side="buy",
            order_type="stop",
            size=abs(size),
            stop_price=stop,
            reason=reason,
            tags=self._trade_tags.copy()
        )
    
    def stop_sell(self, stop: float, size: float, reason: str = "") -> str:
        """Place stop sell order."""
        return self._engine.place_order(
            symbol=self._current_symbol,
            side="sell",
            order_type="stop",
            size=abs(size),
            stop_price=stop,
            reason=reason,
            tags=self._trade_tags.copy()
        )
    
    def cancel(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancelled, False if not found/already filled
        """
        return self._engine.cancel_order(order_id)
    
    def replace(self, order_id: str, **kwargs) -> Optional[str]:
        """
        Replace an order with new parameters.
        
        Args:
            order_id: ID of order to replace
            **kwargs: New order parameters (price, size, etc.)
            
        Returns:
            New order ID if successful, None if failed
        """
        return self._engine.replace_order(order_id, **kwargs)
    
    # Utility methods
    
    def record(self, key: str, value: float) -> None:
        """
        Record time-series value for later visualization.
        
        Args:
            key: Metric name
            value: Metric value
        """
        if key not in self._records:
            self._records[key] = []
        self._records[key].append((self.now, value))
    
    def tag_trade(self, tag: str) -> None:
        """
        Attach tag/metadata to the next trade opened.
        
        Args:
            tag: Tag string
        """
        if tag not in self._trade_tags:
            self._trade_tags.append(tag)
    
    def log(self, level: str, msg: str) -> None:
        """
        Log message with standard formatting.
        
        Args:
            level: Log level ("debug", "info", "warning", "error")
            msg: Message to log
        """
        log_func = getattr(self._logger, level.lower(), self._logger.info)
        log_func(f"[{self._current_symbol}@{self.now}] {msg}")
    
    def emit_event(self, name: str, payload: Dict[str, Any]) -> None:
        """
        Emit user-defined event for timeline debugger.
        
        Args:
            name: Event name
            payload: Event data
        """
        self._engine.emit_event(name, {
            "symbol": self._current_symbol,
            "timestamp": self.now,
            "bar_index": self.bar_index,
            **payload
        })
    
    def get_records(self) -> Dict[str, List[tuple]]:
        """Get all recorded time-series data."""
        return self._records.copy()
    
    def get_orders(self):
        """Get orders placed this bar and clear the list."""
        orders = self._orders.copy()
        self._orders.clear()
        return orders
    
    # Internal methods for engine use
    
    def _update_state(self, bar_index: int, bar: Bar) -> None:
        """Update context state for new bar (internal use)."""
        self._current_index = bar_index
        self._current_bar = bar
        self.data._current_index = bar_index
        self._trade_tags.clear()