"""
Fill execution model for backtesting engine.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from .strategy import Order


@dataclass
class FillConfig:
    """Configuration for fill execution model."""
    
    # Slippage and fees
    slippage_bps: float = 0.0  # Slippage in basis points
    fee_rate_bps: float = 0.0  # Fee rate in basis points
    min_fee_abs: float = 0.0  # Minimum absolute fee
    
    # Execution timing
    fill_delay_bars: int = 1  # Bars to wait before fill (1 = next bar open)
    use_open_for_fills: bool = True  # Use open price vs close price for fills
    
    # Random seed for deterministic slippage
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)


@dataclass
class Fill:
    """Represents an executed fill."""
    
    order: Order
    fill_time: pd.Timestamp
    fill_price: float
    fill_qty: float  # Base quantity (positive)
    fill_notional: float  # Quote currency amount
    fees_paid: float
    slippage_cost: float
    fill_idx: int  # Bar index where fill occurred


class FillEngine:
    """
    Handles order execution with configurable slippage and fees.
    
    Default policy:
    - Orders placed on bar t are filled at bar t+1 open
    - Slippage applied as random normal distribution
    - Fees calculated as percentage of notional + minimum
    """
    
    def __init__(self, config: FillConfig = None):
        self.config = config or FillConfig()
        self._pending_orders: List[Tuple[Order, int]] = []  # (order, submit_bar_idx)
    
    def submit_orders(self, orders: List[Order], submit_bar_idx: int) -> None:
        """
        Submit orders for future execution.
        
        Args:
            orders: List of orders to submit
            submit_bar_idx: Bar index when orders were submitted
        """
        for order in orders:
            self._pending_orders.append((order, submit_bar_idx))
    
    def process_fills(self, current_bar_idx: int, current_bar: pd.Series) -> List[Fill]:
        """
        Process any orders that are ready to be filled at current bar.
        
        Args:
            current_bar_idx: Current bar index
            current_bar: Current bar data
            
        Returns:
            List of fills executed this bar
        """
        fills = []
        remaining_orders = []
        
        for order, submit_idx in self._pending_orders:
            # Check if order is ready to fill
            if current_bar_idx >= submit_idx + self.config.fill_delay_bars:
                fill = self._execute_order(order, current_bar, current_bar_idx)
                if fill:
                    fills.append(fill)
            else:
                remaining_orders.append((order, submit_idx))
        
        self._pending_orders = remaining_orders
        return fills
    
    def _execute_order(self, order: Order, bar: pd.Series, bar_idx: int) -> Optional[Fill]:
        """
        Execute a single order at the given bar.
        
        Args:
            order: Order to execute
            bar: Bar data for execution
            bar_idx: Bar index
            
        Returns:
            Fill object or None if order cannot be executed
        """
        # Determine base fill price
        if self.config.use_open_for_fills:
            base_price = bar['open']
        else:
            base_price = bar['close']
        
        if base_price <= 0:
            return None  # Invalid price
        
        # Apply slippage
        fill_price = self._apply_slippage(base_price, order.side)
        
        # Calculate fill quantity
        if order.size_mode == "notional":
            fill_notional = order.size
            fill_qty = fill_notional / fill_price
        else:  # qty mode
            fill_qty = order.size
            fill_notional = fill_qty * fill_price
        
        # Calculate fees
        fees_paid = self._calculate_fees(fill_notional)
        
        # Calculate slippage cost
        slippage_cost = abs(fill_price - base_price) * fill_qty
        
        return Fill(
            order=order,
            fill_time=bar['dt_open'] if self.config.use_open_for_fills else bar['dt_close'],
            fill_price=fill_price,
            fill_qty=fill_qty,
            fill_notional=fill_notional,
            fees_paid=fees_paid,
            slippage_cost=slippage_cost,
            fill_idx=bar_idx
        )
    
    def _apply_slippage(self, base_price: float, side: str) -> float:
        """
        Apply slippage to the base price.
        
        Args:
            base_price: Base execution price
            side: Order side ("buy" or "sell")
            
        Returns:
            Price after slippage
        """
        if self.config.slippage_bps == 0:
            return base_price
        
        # Convert basis points to decimal
        slippage_rate = self.config.slippage_bps / 10000.0
        
        # Generate random slippage (normal distribution centered at 0)
        # Positive slippage hurts the trader (higher buy price, lower sell price)
        random_factor = np.random.normal(0, 0.5)  # 0.5 std dev for randomness
        actual_slippage_rate = slippage_rate * (0.5 + random_factor)
        
        if side == "buy":
            # Buy slippage increases the price
            return base_price * (1 + actual_slippage_rate)
        else:
            # Sell slippage decreases the price
            return base_price * (1 - actual_slippage_rate)
    
    def _calculate_fees(self, notional: float) -> float:
        """
        Calculate fees for a trade.
        
        Args:
            notional: Notional trade value in quote currency
            
        Returns:
            Total fees
        """
        if self.config.fee_rate_bps == 0 and self.config.min_fee_abs == 0:
            return 0.0
        
        # Percentage fee
        percentage_fee = notional * (self.config.fee_rate_bps / 10000.0)
        
        # Apply minimum fee
        return max(percentage_fee, self.config.min_fee_abs)
    
    def get_pending_orders(self) -> List[Order]:
        """Get list of pending orders."""
        return [order for order, _ in self._pending_orders]
    
    def cancel_all_orders(self) -> List[Order]:
        """Cancel all pending orders and return them."""
        cancelled = [order for order, _ in self._pending_orders]
        self._pending_orders.clear()
        return cancelled


def create_fill_config(fee_rate_bps: float = 0.0, slippage_bps: float = 0.0, 
                       min_fee_abs: float = 0.0, random_seed: Optional[int] = 42) -> FillConfig:
    """
    Convenience function to create a FillConfig with common settings.
    
    Args:
        fee_rate_bps: Fee rate in basis points (e.g., 10 for 0.1%)
        slippage_bps: Slippage in basis points (e.g., 5 for 0.05%)
        min_fee_abs: Minimum absolute fee
        random_seed: Random seed for deterministic behavior
        
    Returns:
        Configured FillConfig
    """
    return FillConfig(
        fee_rate_bps=fee_rate_bps,
        slippage_bps=slippage_bps,
        min_fee_abs=min_fee_abs,
        random_seed=random_seed
    )


class PositionManager:
    """
    Manages position state and applies fills.
    """
    
    def __init__(self, initial_cash: float = 100000.0):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        
        # Position tracking
        self.position_qty = 0.0  # Positive for long, negative for short
        self.position_avg_price = 0.0
        
        # P&L tracking
        self.realized_pnl = 0.0
        self.total_fees_paid = 0.0
        self.total_slippage_cost = 0.0
    
    def apply_fill(self, fill: Fill) -> Tuple[bool, str]:
        """
        Apply a fill to the current position.
        
        Args:
            fill: Fill to apply
            
        Returns:
            Tuple of (success, message)
        """
        side = fill.order.side
        fill_qty_signed = fill.fill_qty if side == "buy" else -fill.fill_qty
        
        # Check if we have enough cash for buys
        if side == "buy":
            required_cash = fill.fill_notional + fill.fees_paid
            if required_cash > self.cash:
                return False, f"Insufficient cash: need {required_cash:.2f}, have {self.cash:.2f}"
        
        # Update position
        old_qty = self.position_qty
        new_qty = old_qty + fill_qty_signed
        
        if old_qty == 0:
            # Opening new position
            self.position_qty = new_qty
            self.position_avg_price = fill.fill_price
        elif (old_qty > 0 and fill_qty_signed > 0) or (old_qty < 0 and fill_qty_signed < 0):
            # Adding to existing position
            total_cost = abs(old_qty) * self.position_avg_price + fill.fill_notional
            self.position_qty = new_qty
            self.position_avg_price = total_cost / abs(new_qty)
        else:
            # Reducing or closing position
            if abs(new_qty) < abs(old_qty):
                # Partial close - realize P&L on closed portion
                closed_qty = abs(fill_qty_signed)
                if old_qty > 0:  # Closing long
                    pnl = closed_qty * (fill.fill_price - self.position_avg_price)
                else:  # Closing short
                    pnl = closed_qty * (self.position_avg_price - fill.fill_price)
                
                self.realized_pnl += pnl
                self.position_qty = new_qty
                # Avg price stays the same for remaining position
            else:
                # Full close or flip
                if old_qty != 0:
                    # Realize P&L on closed portion
                    closed_qty = abs(old_qty)
                    if old_qty > 0:  # Closing long
                        pnl = closed_qty * (fill.fill_price - self.position_avg_price)
                    else:  # Closing short
                        pnl = closed_qty * (self.position_avg_price - fill.fill_price)
                    
                    self.realized_pnl += pnl
                
                # Handle remaining quantity (flip)
                remaining_qty = abs(new_qty) - abs(old_qty)
                if remaining_qty > 0:
                    self.position_qty = remaining_qty if fill_qty_signed > 0 else -remaining_qty
                    self.position_avg_price = fill.fill_price
                else:
                    self.position_qty = 0.0
                    self.position_avg_price = 0.0
        
        # Update cash
        if side == "buy":
            self.cash -= (fill.fill_notional + fill.fees_paid)
        else:
            self.cash += (fill.fill_notional - fill.fees_paid)
        
        # Track costs
        self.total_fees_paid += fill.fees_paid
        self.total_slippage_cost += fill.slippage_cost
        
        return True, "Fill applied successfully"
    
    def get_equity(self, mark_price: float) -> float:
        """
        Calculate current equity (cash + position value).
        
        Args:
            mark_price: Current market price for marking position
            
        Returns:
            Total equity
        """
        if self.position_qty == 0:
            return self.cash
        
        position_value = self.position_qty * mark_price
        unrealized_pnl = position_value - (abs(self.position_qty) * self.position_avg_price)
        
        return self.cash + abs(position_value) + unrealized_pnl
    
    def get_unrealized_pnl(self, mark_price: float) -> float:
        """Get unrealized P&L at current market price."""
        if self.position_qty == 0:
            return 0.0
        
        if self.position_qty > 0:  # Long
            return self.position_qty * (mark_price - self.position_avg_price)
        else:  # Short
            return abs(self.position_qty) * (self.position_avg_price - mark_price)
    
    def get_total_pnl(self, mark_price: float) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.realized_pnl + self.get_unrealized_pnl(mark_price)