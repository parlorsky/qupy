"""
BarUp strategy example using the new Strategy base class.

This demonstrates the formal Strategy interface with proper lifecycle hooks,
parameter validation, and deterministic behavior.
"""

from engine.strategy_base import Strategy, Bar, register_strategy


@register_strategy("barup")
class BarUpStrategy(Strategy):
    """
    Simple momentum strategy that enters long when current close > previous close.
    
    Entry Rules:
    - Enter long when close[t] > close[t-1] AND no position
    
    Exit Rules:
    - Exit when close[t] < close[t-1] (opposite signal)
    - OR after N bars (time stop)
    
    Parameters:
    - lookback: Number of bars to look back for comparison (default: 1)
    - notional: Notional value per trade in quote currency (default: 10000.0)
    - time_stop: Maximum bars to hold position (default: 10)
    """
    
    name = "barup"
    
    @classmethod
    def param_schema(cls):
        return {
            "lookback": {"type": "int", "min": 1, "default": 1},
            "notional": {"type": "float", "min": 0, "default": 10_000.0},
            "time_stop": {"type": "int", "min": 1, "default": 10}
        }
    
    def on_init(self, context):
        """Initialize strategy parameters and state."""
        self.lookback = int(self.params.get("lookback", 1))
        self.notional = float(self.params.get("notional", 10_000.0))
        self.time_stop = int(self.params.get("time_stop", 10))
        
        # Strategy state
        self.bars_held = 0
        
        context.log("info", f"BarUp strategy initialized with lookback={self.lookback}, "
                           f"notional={self.notional}, time_stop={self.time_stop}")
    
    def on_start(self, context):
        """Called at start of backtest run."""
        context.log("info", "BarUp strategy starting")
        context.record("lookback", self.lookback)
        context.record("notional", self.notional)
    
    def on_bar(self, context, symbol: str, bar: Bar):
        """Main trading logic called for each bar."""
        
        # Close logic first - check if we should exit existing position
        if context.position.qty != 0:
            self.bars_held += 1
            
            # Get previous close for exit signal
            if self.lookback + 1 <= context.bar_index:
                prev_closes = context.data.history(symbol, "close", 2)
                if len(prev_closes) >= 2:
                    prev_close = prev_closes[0]  # Previous bar's close
                    
                    # Exit conditions
                    exit_reason = None
                    if bar.close < prev_close:
                        exit_reason = "Exit:BarDown"
                    elif self.bars_held >= self.time_stop:
                        exit_reason = "Exit:TimeStop"
                    
                    if exit_reason:
                        context.close(reason=exit_reason)
                        context.log("info", f"Position closed: {exit_reason}, "
                                          f"price={bar.close:.6f}, bars_held={self.bars_held}")
                        self.bars_held = 0
                        return
        
        # Entry logic - only enter if flat
        if context.position.qty == 0:
            # Need enough history for comparison
            if context.bar_index >= self.lookback:
                closes = context.data.history(symbol, "close", self.lookback + 1)
                
                if len(closes) >= self.lookback + 1:
                    current_close = closes[-1]  # Current bar close
                    prev_close = closes[-2]     # Previous bar close
                    
                    # Enter long if current close > previous close
                    if current_close > prev_close:
                        qty = context.size.from_notional(self.notional, price=bar.close)
                        context.buy(qty, reason="Entry:BarUp")
                        context.tag_trade("BarUp")
                        
                        context.log("info", f"Long entry: price={bar.close:.6f}, "
                                          f"qty={qty:.4f}, signal={current_close:.6f}>{prev_close:.6f}")
                        
                        # Record entry metrics
                        context.record("entry_price", bar.close)
                        context.record("signal_strength", (current_close - prev_close) / prev_close)
    
    def on_trade_open(self, context, trade):
        """Called when a new position is established."""
        self.bars_held = 0
        context.log("info", f"Trade opened: {trade.side} {trade.qty} @ {trade.entry_price}")
        context.record("trade_entry", trade.entry_price)
    
    def on_trade_close(self, context, trade):
        """Called when a position is fully closed."""
        if trade.pnl is not None:
            context.log("info", f"Trade closed: PnL={trade.pnl:.2f}, "
                              f"entry={trade.entry_price:.6f}, exit={trade.exit_price:.6f}")
            context.record("trade_pnl", trade.pnl)
            context.record("trade_exit", trade.exit_price or 0)
    
    def on_stop(self, context):
        """Called at end of backtest."""
        # Close any remaining position
        if context.position.qty != 0:
            context.close(reason="EndOfBacktest")
        
        context.log("info", "BarUp strategy completed")


@register_strategy("barup_multi")
class BarUpMultiAsset(Strategy):
    """
    Multi-asset version of BarUp that can trade multiple symbols.
    
    Demonstrates cross-asset signal generation and portfolio-level logic.
    """
    
    name = "barup_multi"
    
    @classmethod
    def param_schema(cls):
        return {
            "lookback": {"type": "int", "min": 1, "default": 1},
            "notional_per_symbol": {"type": "float", "min": 0, "default": 5_000.0},
            "max_positions": {"type": "int", "min": 1, "default": 3},
            "time_stop": {"type": "int", "min": 1, "default": 10}
        }
    
    def on_init(self, context):
        """Initialize multi-asset strategy."""
        self.lookback = int(self.params.get("lookback", 1))
        self.notional_per_symbol = float(self.params.get("notional_per_symbol", 5_000.0))
        self.max_positions = int(self.params.get("max_positions", 3))
        self.time_stop = int(self.params.get("time_stop", 10))
        
        # Track bars held per symbol
        self.bars_held = {}
    
    def universe(self) -> list:
        """Return symbols to trade - empty means use whatever is provided."""
        return []
    
    def on_bar(self, context, symbol: str, bar: Bar):
        """Per-symbol trading logic."""
        
        # Initialize symbol tracking
        if symbol not in self.bars_held:
            self.bars_held[symbol] = 0
        
        # Get current position for this symbol
        position = context.position
        
        # Close logic
        if position.qty != 0:
            self.bars_held[symbol] += 1
            
            if context.bar_index >= 1:
                prev_closes = context.data.history(symbol, "close", 2)
                if len(prev_closes) >= 2 and bar.close < prev_closes[0]:
                    context.close(reason="Exit:BarDown")
                    self.bars_held[symbol] = 0
                    return
                elif self.bars_held[symbol] >= self.time_stop:
                    context.close(reason="Exit:TimeStop")
                    self.bars_held[symbol] = 0
                    return
        
        # Entry logic - check portfolio constraints
        else:
            # Count current positions across portfolio
            current_positions = len([exp for exp in context.portfolio.exposures.values() if abs(exp) > 0])
            
            if current_positions < self.max_positions and context.bar_index >= self.lookback:
                closes = context.data.history(symbol, "close", self.lookback + 1)
                
                if len(closes) >= 2 and closes[-1] > closes[-2]:
                    qty = context.size.from_notional(self.notional_per_symbol, price=bar.close)
                    context.buy(qty, reason="Entry:BarUp")
                    context.tag_trade(f"BarUp-{symbol}")
                    
                    context.log("info", f"Multi-asset entry: {symbol} @ {bar.close:.6f}")
    
    def on_trade_open(self, context, trade):
        """Reset bars held counter when trade opens."""
        symbol = trade.symbol
        self.bars_held[symbol] = 0


# Factory functions for easy strategy creation
def create_basic_barup(lookback: int = 1, notional: float = 10_000.0, 
                       time_stop: int = 10) -> BarUpStrategy:
    """Create a basic BarUp strategy with specified parameters."""
    return BarUpStrategy(params={
        "lookback": lookback,
        "notional": notional,
        "time_stop": time_stop
    })


def create_barup_multi(lookback: int = 1, notional_per_symbol: float = 5_000.0,
                       max_positions: int = 3, time_stop: int = 10) -> BarUpMultiAsset:
    """Create a multi-asset BarUp strategy."""
    return BarUpMultiAsset(params={
        "lookback": lookback,
        "notional_per_symbol": notional_per_symbol,
        "max_positions": max_positions,
        "time_stop": time_stop
    })