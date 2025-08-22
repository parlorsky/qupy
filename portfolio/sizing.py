"""
Position sizing algorithms.

Implementations of various position sizing methods for risk management.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple


ArrayLike = Union[np.ndarray, pd.Series]


# ============================================================================
# Fixed Sizing Methods
# ============================================================================

def fixed_notional_size(notional: float, price: float) -> float:
    """
    Fixed notional position sizing.
    
    Parameters:
    -----------
    notional : float
        Notional value in quote currency
    price : float
        Current price per unit
    
    Returns:
    --------
    float
        Position size in units
    """
    if price <= 0:
        raise ValueError("Price must be positive")
    return abs(notional) / price


def fixed_units_size(units: float) -> float:
    """
    Fixed units position sizing.
    
    Parameters:
    -----------
    units : float
        Fixed number of units
    
    Returns:
    --------
    float
        Position size in units
    """
    return abs(units)


def percent_equity_size(equity: float, percent: float, price: float) -> float:
    """
    Position size as percentage of equity.
    
    Parameters:
    -----------
    equity : float
        Current account equity
    percent : float
        Percentage of equity (0-100)
    price : float
        Current price per unit
    
    Returns:
    --------
    float
        Position size in units
    """
    if not 0 <= percent <= 100:
        raise ValueError("Percent must be between 0 and 100")
    if price <= 0:
        raise ValueError("Price must be positive")
    
    notional = equity * (percent / 100)
    return notional / price


# ============================================================================
# Risk-Based Sizing
# ============================================================================

def fixed_risk_size(entry_price: float, stop_price: float, risk_dollars: float) -> float:
    """
    Position size based on fixed dollar risk.
    
    Parameters:
    -----------
    entry_price : float
        Entry price
    stop_price : float
        Stop loss price
    risk_dollars : float
        Maximum dollar risk
    
    Returns:
    --------
    float
        Position size in units
    """
    if entry_price <= 0 or stop_price <= 0:
        raise ValueError("Prices must be positive")
    if risk_dollars <= 0:
        raise ValueError("Risk amount must be positive")
    
    risk_per_unit = abs(entry_price - stop_price)
    
    if risk_per_unit == 0:
        raise ValueError("Entry and stop prices cannot be equal")
    
    return abs(risk_dollars) / risk_per_unit


def percent_risk_size(equity: float, risk_percent: float, entry_price: float, 
                     stop_price: float) -> float:
    """
    Position size based on percentage risk of equity.
    
    Parameters:
    -----------
    equity : float
        Current account equity
    risk_percent : float
        Risk as percentage of equity (0-100)
    entry_price : float
        Entry price
    stop_price : float
        Stop loss price
    
    Returns:
    --------
    float
        Position size in units
    """
    if not 0 <= risk_percent <= 100:
        raise ValueError("Risk percent must be between 0 and 100")
    
    risk_dollars = equity * (risk_percent / 100)
    return fixed_risk_size(entry_price, stop_price, risk_dollars)


def atr_position_size(atr: float, risk_dollars: float, atr_multiplier: float = 2.0) -> float:
    """
    Position size based on ATR risk.
    
    Parameters:
    -----------
    atr : float
        Average True Range
    risk_dollars : float
        Maximum dollar risk
    atr_multiplier : float
        ATR multiplier for stop distance
    
    Returns:
    --------
    float
        Position size in units
    """
    if atr <= 0:
        raise ValueError("ATR must be positive")
    if risk_dollars <= 0:
        raise ValueError("Risk amount must be positive")
    
    risk_per_unit = atr * atr_multiplier
    return abs(risk_dollars) / risk_per_unit


def atr_percent_risk_size(equity: float, risk_percent: float, atr: float,
                         atr_multiplier: float = 2.0) -> float:
    """
    Position size based on ATR with percentage risk.
    
    Parameters:
    -----------
    equity : float
        Current account equity
    risk_percent : float
        Risk as percentage of equity (0-100)
    atr : float
        Average True Range
    atr_multiplier : float
        ATR multiplier for stop distance
    
    Returns:
    --------
    float
        Position size in units
    """
    if not 0 <= risk_percent <= 100:
        raise ValueError("Risk percent must be between 0 and 100")
    
    risk_dollars = equity * (risk_percent / 100)
    return atr_position_size(atr, risk_dollars, atr_multiplier)


# ============================================================================
# Volatility-Based Sizing
# ============================================================================

def inverse_vol_position_size(vol_estimate: float, target_risk: float) -> float:
    """
    Position size inversely proportional to volatility.
    
    Parameters:
    -----------
    vol_estimate : float
        Volatility estimate (e.g., standard deviation of returns)
    target_risk : float
        Target risk/volatility contribution
    
    Returns:
    --------
    float
        Position size scaling factor
    """
    if vol_estimate <= 0:
        raise ValueError("Volatility must be positive")
    if target_risk <= 0:
        raise ValueError("Target risk must be positive")
    
    return target_risk / vol_estimate


def volatility_scaled_size(equity: float, vol_estimate: float, target_vol: float,
                          price: float) -> float:
    """
    Position size scaled to achieve target portfolio volatility.
    
    Parameters:
    -----------
    equity : float
        Current account equity
    vol_estimate : float
        Asset volatility estimate
    target_vol : float
        Target portfolio volatility
    price : float
        Current price per unit
    
    Returns:
    --------
    float
        Position size in units
    """
    if vol_estimate <= 0 or target_vol <= 0:
        raise ValueError("Volatilities must be positive")
    if price <= 0:
        raise ValueError("Price must be positive")
    
    # Scale factor to achieve target vol
    scale = target_vol / vol_estimate
    
    # Notional allocation
    notional = equity * scale
    
    return notional / price


def risk_parity_size(vol_estimates: ArrayLike, risk_budget: ArrayLike,
                    total_capital: float, prices: ArrayLike) -> np.ndarray:
    """
    Position sizes for risk parity allocation.
    
    Parameters:
    -----------
    vol_estimates : ArrayLike
        Volatility estimates for each asset
    risk_budget : ArrayLike
        Risk budget for each asset (sums to 1)
    total_capital : float
        Total capital to allocate
    prices : ArrayLike
        Current prices for each asset
    
    Returns:
    --------
    np.ndarray
        Position sizes in units for each asset
    """
    vols = np.asarray(vol_estimates)
    budgets = np.asarray(risk_budget)
    prices_arr = np.asarray(prices)
    
    if np.any(vols <= 0) or np.any(prices_arr <= 0):
        raise ValueError("Volatilities and prices must be positive")
    if abs(budgets.sum() - 1.0) > 1e-6:
        raise ValueError("Risk budgets must sum to 1")
    
    # Inverse volatility weights scaled by risk budget
    raw_weights = budgets / vols
    normalized_weights = raw_weights / raw_weights.sum()
    
    # Convert to position sizes
    notionals = normalized_weights * total_capital
    sizes = notionals / prices_arr
    
    return sizes


# ============================================================================
# Kelly Criterion and Optimal f
# ============================================================================

def kelly_size(win_rate: float, avg_win: float, avg_loss: float,
              equity: float, kelly_fraction: float = 0.25) -> float:
    """
    Kelly criterion position sizing.
    
    Parameters:
    -----------
    win_rate : float
        Probability of winning (0-1)
    avg_win : float
        Average win amount
    avg_loss : float
        Average loss amount (positive value)
    equity : float
        Current account equity
    kelly_fraction : float
        Fraction of Kelly to use (default 0.25 for safety)
    
    Returns:
    --------
    float
        Position size as fraction of equity
    """
    if not 0 <= win_rate <= 1:
        raise ValueError("Win rate must be between 0 and 1")
    if avg_win <= 0 or avg_loss <= 0:
        raise ValueError("Average win and loss must be positive")
    if not 0 < kelly_fraction <= 1:
        raise ValueError("Kelly fraction must be between 0 and 1")
    
    # Kelly formula: f = (p*b - q) / b
    # where p = win_rate, q = 1-p, b = avg_win/avg_loss
    loss_rate = 1 - win_rate
    b = avg_win / avg_loss
    
    kelly_f = (win_rate * b - loss_rate) / b
    
    # Apply Kelly fraction and ensure non-negative
    kelly_f = max(0, kelly_f * kelly_fraction)
    
    # Don't risk more than 100% of equity
    kelly_f = min(1.0, kelly_f)
    
    return equity * kelly_f


def optimal_f_size(returns: ArrayLike, equity: float, 
                  fraction: float = 0.25) -> float:
    """
    Ralph Vince's Optimal f position sizing.
    
    Parameters:
    -----------
    returns : ArrayLike
        Historical returns (as percentages)
    equity : float
        Current account equity
    fraction : float
        Fraction of optimal f to use
    
    Returns:
    --------
    float
        Position size as fraction of equity
    """
    rets = np.asarray(returns)
    
    # Remove NaN values
    rets = rets[~np.isnan(rets)]
    
    if len(rets) == 0:
        return 0
    
    # Find worst loss
    worst_loss = abs(min(rets.min(), 0))
    
    if worst_loss == 0:
        return 0
    
    # Search for optimal f
    best_f = 0
    best_twr = 0
    
    for f in np.linspace(0.01, 0.99, 99):
        # Calculate Terminal Wealth Relative
        twr = 1
        for ret in rets:
            holding_pct = ret / worst_loss
            twr *= (1 + f * holding_pct)
        
        if twr > best_twr:
            best_twr = twr
            best_f = f
    
    # Apply fraction for safety
    return equity * best_f * fraction


# ============================================================================
# Dynamic Position Sizing
# ============================================================================

def pyramiding_schedule(price: float, entry_price: float, 
                       add_levels: List[float]) -> float:
    """
    Determine pyramiding position size based on price levels.
    
    Parameters:
    -----------
    price : float
        Current price
    entry_price : float
        Initial entry price
    add_levels : List[float]
        Price levels for adding (as % from entry)
    
    Returns:
    --------
    float
        Scaling factor for position addition (0-1)
    """
    if price <= 0 or entry_price <= 0:
        raise ValueError("Prices must be positive")
    
    # Calculate price move from entry
    price_move = (price - entry_price) / entry_price
    
    # Determine which level we're at
    for i, level in enumerate(add_levels):
        if price_move >= level:
            # Scale down additions (50%, 25%, 12.5%, etc.)
            return 0.5 ** (i + 1)
    
    return 0


def martingale_size(base_size: float, consecutive_losses: int,
                   multiplier: float = 2.0, max_multiplier: float = 8.0) -> float:
    """
    Martingale position sizing (use with extreme caution).
    
    Parameters:
    -----------
    base_size : float
        Base position size
    consecutive_losses : int
        Number of consecutive losses
    multiplier : float
        Size multiplier per loss
    max_multiplier : float
        Maximum total multiplier
    
    Returns:
    --------
    float
        Adjusted position size
    """
    if consecutive_losses < 0:
        raise ValueError("Consecutive losses must be non-negative")
    if multiplier <= 1:
        raise ValueError("Multiplier must be greater than 1")
    
    total_multiplier = multiplier ** consecutive_losses
    total_multiplier = min(total_multiplier, max_multiplier)
    
    return base_size * total_multiplier


def anti_martingale_size(base_size: float, consecutive_wins: int,
                        multiplier: float = 1.5, max_multiplier: float = 4.0) -> float:
    """
    Anti-Martingale position sizing (increase size after wins).
    
    Parameters:
    -----------
    base_size : float
        Base position size
    consecutive_wins : int
        Number of consecutive wins
    multiplier : float
        Size multiplier per win
    max_multiplier : float
        Maximum total multiplier
    
    Returns:
    --------
    float
        Adjusted position size
    """
    if consecutive_wins < 0:
        raise ValueError("Consecutive wins must be non-negative")
    if multiplier <= 1:
        raise ValueError("Multiplier must be greater than 1")
    
    total_multiplier = multiplier ** consecutive_wins
    total_multiplier = min(total_multiplier, max_multiplier)
    
    return base_size * total_multiplier


def confidence_scaled_size(base_size: float, confidence: float,
                          min_scale: float = 0.25, max_scale: float = 2.0) -> float:
    """
    Scale position size based on signal confidence.
    
    Parameters:
    -----------
    base_size : float
        Base position size
    confidence : float
        Signal confidence (0-1)
    min_scale : float
        Minimum scaling factor
    max_scale : float
        Maximum scaling factor
    
    Returns:
    --------
    float
        Confidence-adjusted position size
    """
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1")
    if min_scale <= 0 or max_scale <= 0:
        raise ValueError("Scale factors must be positive")
    if min_scale > max_scale:
        raise ValueError("Min scale must be less than max scale")
    
    # Linear scaling between min and max based on confidence
    scale = min_scale + (max_scale - min_scale) * confidence
    
    return base_size * scale


# ============================================================================
# Stop Loss and Target Calculations
# ============================================================================

def stop_loss_price(entry_price: float, stop_pct: float, side: str = "long") -> float:
    """
    Calculate stop loss price.
    
    Parameters:
    -----------
    entry_price : float
        Entry price
    stop_pct : float
        Stop loss percentage (e.g., 0.05 for 5%)
    side : str
        "long" or "short"
    
    Returns:
    --------
    float
        Stop loss price
    """
    if entry_price <= 0:
        raise ValueError("Entry price must be positive")
    if stop_pct < 0:
        raise ValueError("Stop percentage must be non-negative")
    
    if side == "long":
        return entry_price * (1 - stop_pct)
    else:
        return entry_price * (1 + stop_pct)


def profit_target_price(entry_price: float, target_pct: float, side: str = "long") -> float:
    """
    Calculate profit target price.
    
    Parameters:
    -----------
    entry_price : float
        Entry price
    target_pct : float
        Target percentage (e.g., 0.10 for 10%)
    side : str
        "long" or "short"
    
    Returns:
    --------
    float
        Profit target price
    """
    if entry_price <= 0:
        raise ValueError("Entry price must be positive")
    if target_pct < 0:
        raise ValueError("Target percentage must be non-negative")
    
    if side == "long":
        return entry_price * (1 + target_pct)
    else:
        return entry_price * (1 - target_pct)


def risk_reward_ratio(entry_price: float, stop_price: float, 
                     target_price: float) -> float:
    """
    Calculate risk-reward ratio.
    
    Parameters:
    -----------
    entry_price : float
        Entry price
    stop_price : float
        Stop loss price
    target_price : float
        Profit target price
    
    Returns:
    --------
    float
        Risk-reward ratio (reward/risk)
    """
    if entry_price <= 0 or stop_price <= 0 or target_price <= 0:
        raise ValueError("All prices must be positive")
    
    risk = abs(entry_price - stop_price)
    reward = abs(target_price - entry_price)
    
    if risk == 0:
        return float('inf')
    
    return reward / risk