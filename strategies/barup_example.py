"""
BarUp strategy: Enter long when close > previous close, exit on opposite signal or time stop.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.strategy import Strategy, Context, FixedSizeStrategy
import pandas as pd


class BarUpStrategy(FixedSizeStrategy):
    """
    Simple momentum strategy that enters long when current close > previous close.
    
    Entry Rules:
    - Enter long when close[t] > close[t-1] AND no position
    
    Exit Rules:
    - Exit when close[t] < close[t-1] (opposite signal)
    - OR after N bars (time stop)
    
    Parameters:
    - size_mode: "notional" or "qty" 
    - size_value: Trade size (dollar amount or quantity)
    - max_bars: Maximum bars to hold position (0 = no limit)
    - min_bars: Minimum bars to hold before considering exit
    """
    
    def _setup_params(self):
        super()._setup_params()
        
        self.max_bars = self.params.get('max_bars', 0)  # 0 = no time stop
        self.min_bars = self.params.get('min_bars', 1)  # Minimum holding period
        
        # Internal state
        self.prev_close = None
        self.entry_bar = None
        self.bars_in_position = 0
        
        if self.max_bars < 0:
            raise ValueError("max_bars must be >= 0")
        if self.min_bars < 0:
            raise ValueError("min_bars must be >= 0")
    
    def on_start(self, context: Context) -> None:
        """Initialize strategy state."""
        pass
    
    def on_bar(self, context: Context, bar: pd.Series) -> None:
        """Process each bar for entry/exit signals."""
        current_close = bar['close']
        current_position = context.position
        
        # Update bars in position counter
        if not current_position.is_flat:
            self.bars_in_position += 1
        else:
            self.bars_in_position = 0
        
        # Entry logic
        if current_position.is_flat and self.prev_close is not None:
            if current_close > self.prev_close:
                # Enter long position
                context.buy(
                    size=self.size_value,
                    reason=f"BarUp entry: {current_close:.6f} > {self.prev_close:.6f}",
                    size_mode=self.size_mode
                )
                self.entry_bar = context.bar_index
                self.bars_in_position = 1
        
        # Exit logic
        elif not current_position.is_flat:
            should_exit = False
            exit_reason = ""
            
            # Check time stop
            if self.max_bars > 0 and self.bars_in_position >= self.max_bars:
                should_exit = True
                exit_reason = f"Time stop: {self.bars_in_position} bars"
            
            # Check opposite signal (but only after minimum holding period)
            elif (self.bars_in_position >= self.min_bars and 
                  self.prev_close is not None and 
                  current_close < self.prev_close):
                should_exit = True
                exit_reason = f"BarDown exit: {current_close:.6f} < {self.prev_close:.6f}"
            
            if should_exit:
                context.close(reason=exit_reason)
                self.entry_bar = None
                self.bars_in_position = 0
        
        # Update state for next bar
        self.prev_close = current_close
    
    def on_stop(self, context: Context) -> None:
        """Clean up at end of backtest."""
        if not context.position.is_flat:
            context.close("End of backtest")


class BarUpWithStopLoss(BarUpStrategy):
    """
    Enhanced BarUp strategy with stop-loss functionality.
    
    Additional Parameters:
    - stop_loss_pct: Stop loss percentage (e.g., 0.05 for 5%)
    - take_profit_pct: Take profit percentage (optional)
    """
    
    def _setup_params(self):
        super()._setup_params()
        
        self.stop_loss_pct = self.params.get('stop_loss_pct', 0.0)
        self.take_profit_pct = self.params.get('take_profit_pct', 0.0)
        
        if self.stop_loss_pct < 0:
            raise ValueError("stop_loss_pct must be >= 0")
        if self.take_profit_pct < 0:
            raise ValueError("take_profit_pct must be >= 0")
    
    def on_bar(self, context: Context, bar: pd.Series) -> None:
        """Enhanced bar processing with stop-loss logic."""
        current_close = bar['close']
        current_low = bar['low']
        current_high = bar['high']
        current_position = context.position
        
        # Update bars in position counter
        if not current_position.is_flat:
            self.bars_in_position += 1
        else:
            self.bars_in_position = 0
        
        # Entry logic (same as parent)
        if current_position.is_flat and self.prev_close is not None:
            if current_close > self.prev_close:
                context.buy(
                    size=self.size_value,
                    reason=f"BarUp entry: {current_close:.6f} > {self.prev_close:.6f}",
                    size_mode=self.size_mode
                )
                self.entry_bar = context.bar_index
                self.bars_in_position = 1
        
        # Exit logic with stop-loss
        elif not current_position.is_flat:
            should_exit = False
            exit_reason = ""
            
            # Check stop-loss (using intrabar low for long positions)
            if (self.stop_loss_pct > 0 and current_position.is_long):
                stop_price = current_position.avg_price * (1 - self.stop_loss_pct)
                if current_low <= stop_price:
                    should_exit = True
                    exit_reason = f"Stop loss: {current_low:.6f} <= {stop_price:.6f}"
            
            # Check take profit (using intrabar high for long positions)
            elif (self.take_profit_pct > 0 and current_position.is_long):
                take_profit_price = current_position.avg_price * (1 + self.take_profit_pct)
                if current_high >= take_profit_price:
                    should_exit = True
                    exit_reason = f"Take profit: {current_high:.6f} >= {take_profit_price:.6f}"
            
            # Check time stop
            elif self.max_bars > 0 and self.bars_in_position >= self.max_bars:
                should_exit = True
                exit_reason = f"Time stop: {self.bars_in_position} bars"
            
            # Check opposite signal
            elif (self.bars_in_position >= self.min_bars and 
                  self.prev_close is not None and 
                  current_close < self.prev_close):
                should_exit = True
                exit_reason = f"BarDown exit: {current_close:.6f} < {self.prev_close:.6f}"
            
            if should_exit:
                context.close(reason=exit_reason)
                self.entry_bar = None
                self.bars_in_position = 0
        
        # Update state for next bar
        self.prev_close = current_close


class BarUpMeanReversion(Strategy):
    """
    Mean reversion version: Enter long when price drops below moving average,
    exit when it rises above.
    
    Parameters:
    - lookback: Moving average lookback period
    - size_mode: "notional" or "qty"
    - size_value: Trade size
    - threshold_pct: Percentage below MA to trigger entry (e.g., 0.02 for 2%)
    """
    
    def _setup_params(self):
        self.lookback = self.params.get('lookback', 20)
        self.size_mode = self.params.get('size_mode', 'notional')
        self.size_value = self.params.get('size_value', 1000.0)
        self.threshold_pct = self.params.get('threshold_pct', 0.01)
        
        # State
        self.price_history = []
        
        if self.lookback <= 0:
            raise ValueError("lookback must be > 0")
        if self.threshold_pct < 0:
            raise ValueError("threshold_pct must be >= 0")
    
    def on_start(self, context: Context) -> None:
        pass
    
    def on_bar(self, context: Context, bar: pd.Series) -> None:
        current_close = bar['close']
        current_position = context.position
        
        # Update price history
        self.price_history.append(current_close)
        if len(self.price_history) > self.lookback:
            self.price_history.pop(0)
        
        # Need enough history for moving average
        if len(self.price_history) < self.lookback:
            return
        
        # Calculate moving average
        ma = sum(self.price_history) / len(self.price_history)
        
        # Entry logic: price significantly below MA
        if current_position.is_flat:
            threshold_price = ma * (1 - self.threshold_pct)
            if current_close < threshold_price:
                context.buy(
                    size=self.size_value,
                    reason=f"Mean reversion entry: {current_close:.6f} < {threshold_price:.6f} (MA: {ma:.6f})",
                    size_mode=self.size_mode
                )
        
        # Exit logic: price back above MA
        elif current_position.is_long and current_close > ma:
            context.close(f"Mean reversion exit: {current_close:.6f} > MA {ma:.6f}")
    
    def on_stop(self, context: Context) -> None:
        if not context.position.is_flat:
            context.close("End of backtest")


# Factory functions for easy strategy creation
def create_basic_barup(size_value: float = 1000, size_mode: str = "notional", 
                       max_bars: int = 0, min_bars: int = 1) -> BarUpStrategy:
    """Create a basic BarUp strategy with common parameters."""
    return BarUpStrategy(
        size_value=size_value,
        size_mode=size_mode,
        max_bars=max_bars,
        min_bars=min_bars
    )


def create_barup_with_stops(size_value: float = 1000, size_mode: str = "notional",
                           max_bars: int = 0, stop_loss_pct: float = 0.05,
                           take_profit_pct: float = 0.0) -> BarUpWithStopLoss:
    """Create a BarUp strategy with stop-loss and take-profit."""
    return BarUpWithStopLoss(
        size_value=size_value,
        size_mode=size_mode,
        max_bars=max_bars,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct
    )


def create_mean_reversion(size_value: float = 1000, lookback: int = 20,
                         threshold_pct: float = 0.02) -> BarUpMeanReversion:
    """Create a mean reversion strategy."""
    return BarUpMeanReversion(
        size_value=size_value,
        lookback=lookback,
        threshold_pct=threshold_pct
    )