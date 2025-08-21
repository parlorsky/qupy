"""
Strategy base classes and core data structures for qupy backtesting engine.

This module provides:
- Bar dataclass for OHLCV data
- Strategy base class with lifecycle hooks
- StrategyMultiAsset for multi-symbol strategies
- Registration decorator for strategy discovery
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
import functools

# Registry for strategy discovery
_STRATEGY_REGISTRY: Dict[str, type] = {}


@dataclass(frozen=True)
class Bar:
    """OHLCV bar data structure."""
    dt_open: Any
    dt_close: Any
    open: float
    high: float
    low: float
    close: float
    volume: float


def register_strategy(name: str):
    """
    Decorator to register a strategy class for discovery.
    
    Usage:
        @register_strategy("barup")
        class BarUpStrategy(Strategy):
            ...
    """
    def decorator(cls):
        _STRATEGY_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


def get_registered_strategies() -> Dict[str, type]:
    """Get all registered strategy classes."""
    return _STRATEGY_REGISTRY.copy()


class Strategy:
    """
    Base class for all trading strategies.
    
    Provides lifecycle hooks, parameter management, and interface for
    interacting with the backtesting engine via context object.
    """
    
    name = "base"
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy with parameters.
        
        Args:
            params: Strategy-specific parameters dict
        """
        self.params = dict(params or {})
        self._validate_params()
    
    @classmethod
    def param_schema(cls) -> Optional[Dict[str, Any]]:
        """
        Return parameter schema for validation and documentation.
        
        Returns:
            Dict with parameter definitions or None if no schema defined.
            Format: {param_name: {type: str, min: float, max: float, default: Any}}
        """
        return None
    
    def _validate_params(self):
        """Validate parameters against schema if defined."""
        schema = self.param_schema()
        if not schema:
            return
            
        for param_name, spec in schema.items():
            if param_name in self.params:
                value = self.params[param_name]
                param_type = spec.get("type")
                
                # Type validation
                if param_type == "int" and not isinstance(value, int):
                    raise ValueError(f"Parameter {param_name} must be int, got {type(value)}")
                elif param_type == "float" and not isinstance(value, (int, float)):
                    raise ValueError(f"Parameter {param_name} must be float, got {type(value)}")
                elif param_type == "str" and not isinstance(value, str):
                    raise ValueError(f"Parameter {param_name} must be str, got {type(value)}")
                
                # Range validation
                if "min" in spec and value < spec["min"]:
                    raise ValueError(f"Parameter {param_name} must be >= {spec['min']}, got {value}")
                if "max" in spec and value > spec["max"]:
                    raise ValueError(f"Parameter {param_name} must be <= {spec['max']}, got {value}")
            
            elif "default" not in spec:
                raise ValueError(f"Required parameter {param_name} not provided")
            else:
                # Set default value
                self.params[param_name] = spec["default"]
    
    def universe(self) -> List[str]:
        """
        Return list of symbols this strategy trades.
        
        Default implementation returns empty list, meaning strategy
        will trade whatever symbols are provided by the backtest runner.
        """
        return []
    
    @property
    def timeframe_hint(self) -> str:
        """
        Hint for the strategy's primary timeframe.
        
        Used for scheduling and Sharpe ratio annualization.
        Examples: "1m", "5m", "1h", "1d"
        """
        return "1d"
    
    # Lifecycle hooks
    
    def on_init(self, context) -> None:
        """
        One-time initialization called during strategy construction.
        
        Use this to:
        - Validate parameters
        - Allocate caches and state variables
        - Set up any derived configuration
        
        Args:
            context: Context object providing access to data and portfolio
        """
        pass
    
    def on_start(self, context) -> None:
        """
        Called at start of backtest run after data and portfolio are ready.
        
        Use this to:
        - Initialize state that depends on data availability
        - Set up indicators or derived series
        - Log initial state
        
        Args:
            context: Context object
        """
        pass
    
    def on_bar(self, context, symbol: str, bar: Bar) -> None:
        """
        Main decision loop called for each bar of each symbol.
        
        This is where trading logic goes. Place orders using context methods:
        - context.buy(size, reason)
        - context.sell(size, reason)  
        - context.close(reason)
        - context.limit_buy/limit_sell/stop_buy/stop_sell
        
        Args:
            context: Context object for data access and order placement
            symbol: Symbol being processed
            bar: Current OHLCV bar
        """
        raise NotImplementedError("Subclasses must implement on_bar")
    
    def on_fill(self, context, fill) -> None:
        """
        Called when an order is filled (partial or complete).
        
        Args:
            context: Context object
            fill: Fill object with details about the execution
        """
        pass
    
    def on_trade_open(self, context, trade) -> None:
        """
        Called when a new position is established.
        
        Args:
            context: Context object
            trade: Trade object representing the new position
        """
        pass
    
    def on_trade_close(self, context, trade) -> None:
        """
        Called when a position is fully closed.
        
        Args:
            context: Context object  
            trade: Trade object representing the closed position
        """
        pass
    
    def on_stop(self, context) -> None:
        """
        Called at end of backtest run for cleanup and finalization.
        
        Use this to:
        - Log final statistics
        - Save artifacts
        - Clean up resources
        
        Args:
            context: Context object
        """
        pass
    
    # State management for checkpointing
    
    def to_state(self) -> Dict[str, Any]:
        """
        Serialize strategy state for checkpointing.
        
        Override to save custom state variables that need to persist
        across checkpoint/restore cycles.
        
        Returns:
            Dict containing serializable state
        """
        return {"params": self.params}
    
    def from_state(self, state: Dict[str, Any]) -> None:
        """
        Restore strategy state from checkpoint.
        
        Override to restore custom state variables.
        
        Args:
            state: State dict returned by to_state()
        """
        if "params" in state:
            self.params = state["params"]


class StrategyMultiAsset(Strategy):
    """
    Base class for multi-asset strategies with additional helpers.
    
    Provides utilities for managing multiple symbols and cross-asset signals.
    """
    
    def universe(self, context) -> List[str]:
        """
        Return list of symbols to trade based on context.
        
        Override to provide dynamic universe selection based on
        available data, market conditions, etc.
        
        Args:
            context: Context object with data access
            
        Returns:
            List of symbol strings
        """
        return []
    
    def on_bar_multi(self, context, bars: Dict[str, Bar]) -> None:
        """
        Alternative hook called with all symbols' bars at once.
        
        Useful for cross-asset signals and portfolio-level decisions.
        If implemented, this is called instead of individual on_bar calls.
        
        Args:
            context: Context object
            bars: Dict mapping symbol -> Bar for current timestamp
        """
        # Default implementation calls on_bar for each symbol
        for symbol, bar in bars.items():
            self.on_bar(context, symbol, bar)