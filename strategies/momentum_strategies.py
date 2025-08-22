"""
Momentum trading strategies implementations using the Strategy base class.

This module contains various momentum-based trading strategies from cookbook 02:
- Classic price momentum
- RSI momentum
- Breakout strategies (Donchian channels)
- ADX trend strength momentum
- Dual momentum
- Cross-sectional momentum
"""

import numpy as np
from typing import List, Dict, Optional
from engine.strategy_base import Strategy, Bar, register_strategy


@register_strategy("momentum_classic")
class ClassicMomentumStrategy(Strategy):
    """
    Classic momentum strategy based on price rate of change.
    
    Entry: Buy when momentum > threshold
    Exit: Sell when momentum < -threshold OR time stop
    """
    
    name = "momentum_classic"
    
    @classmethod
    def param_schema(cls):
        return {
            "lookback": {"type": "int", "min": 5, "max": 200, "default": 20},
            "threshold": {"type": "float", "min": 0.01, "max": 0.5, "default": 0.05},
            "notional": {"type": "float", "min": 100, "default": 10_000.0},
            "time_stop": {"type": "int", "min": 1, "default": 30}
        }
    
    def on_init(self, context):
        self.lookback = int(self.params.get("lookback", 20))
        self.threshold = float(self.params.get("threshold", 0.05))
        self.notional = float(self.params.get("notional", 10_000.0))
        self.time_stop = int(self.params.get("time_stop", 30))
        
        # State tracking
        self.bars_held = 0
        
        context.log("info", f"Classic Momentum initialized: lookback={self.lookback}, "
                           f"threshold={self.threshold:.3f}, notional=${self.notional:,.0f}")
    
    def on_bar(self, context, symbol: str, bar: Bar):
        # Need enough history
        if context.bar_index < self.lookback:
            return
        
        # Get price history
        closes = context.data.history(symbol, "close", self.lookback + 1)
        if len(closes) < self.lookback + 1:
            return
        
        # Calculate momentum (rate of change)
        momentum = (closes[-1] / closes[0] - 1) if closes[0] != 0 else 0
        
        current_pos = context.position.qty
        
        # Exit logic first
        if current_pos != 0:
            self.bars_held += 1
            
            # Exit conditions
            exit_signal = False
            exit_reason = None
            
            if momentum < -self.threshold:
                exit_signal = True
                exit_reason = "Exit:Momentum_Negative"
            elif self.bars_held >= self.time_stop:
                exit_signal = True
                exit_reason = "Exit:TimeStop"
            
            if exit_signal:
                context.close(reason=exit_reason)
                context.log("info", f"Position closed: {exit_reason}, momentum={momentum:.4f}")
                self.bars_held = 0
                return
        
        # Entry logic
        if current_pos == 0 and momentum > self.threshold:
            qty = context.size.from_notional(self.notional, bar.close)
            context.buy(qty, reason="Entry:Momentum_Positive")
            context.tag_trade("Classic_Momentum")
            
            context.log("info", f"Long entry: momentum={momentum:.4f}, price=${bar.close:.6f}")
            context.record("momentum", momentum)
    
    def on_trade_open(self, context, trade):
        self.bars_held = 0
        context.record("entry_price", trade.entry_price)
    
    def on_trade_close(self, context, trade):
        if trade.pnl is not None:
            context.log("info", f"Trade PnL: ${trade.pnl:.2f}")
            context.record("trade_pnl", trade.pnl)


@register_strategy("rsi_momentum")
class RSIMomentumStrategy(Strategy):
    """
    RSI-based momentum strategy using momentum mode (not mean reversion).
    
    Entry: RSI > 50 (momentum continuation)
    Exit: RSI < 50 OR time stop
    """
    
    name = "rsi_momentum"
    
    @classmethod
    def param_schema(cls):
        return {
            "rsi_period": {"type": "int", "min": 5, "max": 50, "default": 14},
            "rsi_threshold": {"type": "float", "min": 45, "max": 70, "default": 50},
            "notional": {"type": "float", "min": 100, "default": 10_000.0},
            "time_stop": {"type": "int", "min": 1, "default": 20}
        }
    
    def on_init(self, context):
        self.rsi_period = int(self.params.get("rsi_period", 14))
        self.rsi_threshold = float(self.params.get("rsi_threshold", 50))
        self.notional = float(self.params.get("notional", 10_000.0))
        self.time_stop = int(self.params.get("time_stop", 20))
        
        self.bars_held = 0
        
        context.log("info", f"RSI Momentum initialized: RSI period={self.rsi_period}, "
                           f"threshold={self.rsi_threshold}")
    
    def calculate_rsi(self, closes: List[float], period: int) -> float:
        """Calculate RSI for the given price series."""
        if len(closes) < period + 1:
            return 50.0  # Neutral RSI
        
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        if avg_gain == 0:
            return 0.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def on_bar(self, context, symbol: str, bar: Bar):
        # Need enough history for RSI
        if context.bar_index < self.rsi_period + 1:
            return
        
        closes = context.data.history(symbol, "close", self.rsi_period + 1)
        if len(closes) < self.rsi_period + 1:
            return
        
        rsi = self.calculate_rsi(closes, self.rsi_period)
        current_pos = context.position.qty
        
        # Exit logic
        if current_pos != 0:
            self.bars_held += 1
            
            exit_signal = False
            exit_reason = None
            
            if rsi < self.rsi_threshold:
                exit_signal = True
                exit_reason = "Exit:RSI_Below_Threshold"
            elif self.bars_held >= self.time_stop:
                exit_signal = True
                exit_reason = "Exit:TimeStop"
            
            if exit_signal:
                context.close(reason=exit_reason)
                context.log("info", f"Position closed: {exit_reason}, RSI={rsi:.2f}")
                self.bars_held = 0
                return
        
        # Entry logic
        if current_pos == 0 and rsi > self.rsi_threshold:
            qty = context.size.from_notional(self.notional, bar.close)
            context.buy(qty, reason="Entry:RSI_Momentum")
            context.tag_trade("RSI_Momentum")
            
            context.log("info", f"Long entry: RSI={rsi:.2f}, price=${bar.close:.6f}")
            context.record("rsi", rsi)
    
    def on_trade_open(self, context, trade):
        self.bars_held = 0


@register_strategy("breakout_donchian")
class DonchianBreakoutStrategy(Strategy):
    """
    Breakout momentum strategy using Donchian Channels.
    
    Entry: Price breaks above upper channel
    Exit: Price breaks below lower channel OR time stop
    """
    
    name = "breakout_donchian"
    
    @classmethod
    def param_schema(cls):
        return {
            "lookback": {"type": "int", "min": 10, "max": 100, "default": 20},
            "notional": {"type": "float", "min": 100, "default": 10_000.0},
            "time_stop": {"type": "int", "min": 1, "default": 25}
        }
    
    def on_init(self, context):
        self.lookback = int(self.params.get("lookback", 20))
        self.notional = float(self.params.get("notional", 10_000.0))
        self.time_stop = int(self.params.get("time_stop", 25))
        
        self.bars_held = 0
        
        context.log("info", f"Donchian Breakout initialized: lookback={self.lookback}")
    
    def calculate_donchian_channels(self, highs: List[float], lows: List[float]) -> Dict[str, float]:
        """Calculate Donchian channel values."""
        if len(highs) < self.lookback or len(lows) < self.lookback:
            return {"upper": 0, "lower": 0, "middle": 0}
        
        upper = max(highs[-self.lookback:])
        lower = min(lows[-self.lookback:])
        middle = (upper + lower) / 2
        
        return {"upper": upper, "lower": lower, "middle": middle}
    
    def on_bar(self, context, symbol: str, bar: Bar):
        # Need enough history
        if context.bar_index < self.lookback:
            return
        
        highs = context.data.history(symbol, "high", self.lookback)
        lows = context.data.history(symbol, "low", self.lookback)
        
        if len(highs) < self.lookback or len(lows) < self.lookback:
            return
        
        channels = self.calculate_donchian_channels(highs, lows)
        current_pos = context.position.qty
        
        # Exit logic
        if current_pos != 0:
            self.bars_held += 1
            
            exit_signal = False
            exit_reason = None
            
            if bar.close < channels["lower"]:
                exit_signal = True
                exit_reason = "Exit:Below_Lower_Channel"
            elif self.bars_held >= self.time_stop:
                exit_signal = True
                exit_reason = "Exit:TimeStop"
            
            if exit_signal:
                context.close(reason=exit_reason)
                context.log("info", f"Position closed: {exit_reason}")
                self.bars_held = 0
                return
        
        # Entry logic - breakout above upper channel
        if current_pos == 0 and bar.close > channels["upper"]:
            qty = context.size.from_notional(self.notional, bar.close)
            context.buy(qty, reason="Entry:Upper_Breakout")
            context.tag_trade("Donchian_Breakout")
            
            context.log("info", f"Breakout entry: price=${bar.close:.6f}, "
                               f"upper=${channels['upper']:.6f}")
            
            context.record("upper_channel", channels["upper"])
            context.record("lower_channel", channels["lower"])
    
    def on_trade_open(self, context, trade):
        self.bars_held = 0


@register_strategy("adx_momentum")
class ADXMomentumStrategy(Strategy):
    """
    ADX-filtered momentum strategy.
    
    Only trades when ADX > threshold (strong trend)
    Direction determined by DI+/DI- and momentum
    """
    
    name = "adx_momentum"
    
    @classmethod
    def param_schema(cls):
        return {
            "adx_period": {"type": "int", "min": 10, "max": 30, "default": 14},
            "adx_threshold": {"type": "float", "min": 20, "max": 40, "default": 25},
            "momentum_period": {"type": "int", "min": 5, "max": 30, "default": 10},
            "notional": {"type": "float", "min": 100, "default": 10_000.0},
            "time_stop": {"type": "int", "min": 1, "default": 20}
        }
    
    def on_init(self, context):
        self.adx_period = int(self.params.get("adx_period", 14))
        self.adx_threshold = float(self.params.get("adx_threshold", 25))
        self.momentum_period = int(self.params.get("momentum_period", 10))
        self.notional = float(self.params.get("notional", 10_000.0))
        self.time_stop = int(self.params.get("time_stop", 20))
        
        self.bars_held = 0
        
        context.log("info", f"ADX Momentum initialized: ADX threshold={self.adx_threshold}")
    
    def calculate_adx_simple(self, highs: List[float], lows: List[float], 
                           closes: List[float]) -> Dict[str, float]:
        """Simplified ADX calculation."""
        if len(highs) < self.adx_period + 1:
            return {"adx": 0, "di_plus": 50, "di_minus": 50}
        
        # True Range
        tr_list = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
        
        if len(tr_list) < self.adx_period:
            return {"adx": 0, "di_plus": 50, "di_minus": 50}
        
        # Simplified directional movement
        dm_plus_list = []
        dm_minus_list = []
        
        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            dm_plus = up_move if up_move > down_move and up_move > 0 else 0
            dm_minus = down_move if down_move > up_move and down_move > 0 else 0
            
            dm_plus_list.append(dm_plus)
            dm_minus_list.append(dm_minus)
        
        # Average values
        avg_tr = np.mean(tr_list[-self.adx_period:])
        avg_dm_plus = np.mean(dm_plus_list[-self.adx_period:])
        avg_dm_minus = np.mean(dm_minus_list[-self.adx_period:])
        
        # DI calculations
        di_plus = (avg_dm_plus / avg_tr * 100) if avg_tr > 0 else 0
        di_minus = (avg_dm_minus / avg_tr * 100) if avg_tr > 0 else 0
        
        # ADX calculation (simplified)
        dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100 if (di_plus + di_minus) > 0 else 0
        adx = dx  # Simplified - should be smoothed
        
        return {"adx": adx, "di_plus": di_plus, "di_minus": di_minus}
    
    def on_bar(self, context, symbol: str, bar: Bar):
        # Need enough history
        if context.bar_index < max(self.adx_period, self.momentum_period) + 1:
            return
        
        highs = context.data.history(symbol, "high", self.adx_period + 1)
        lows = context.data.history(symbol, "low", self.adx_period + 1)
        closes = context.data.history(symbol, "close", max(self.adx_period, self.momentum_period) + 1)
        
        if len(closes) < max(self.adx_period, self.momentum_period) + 1:
            return
        
        # Calculate indicators
        adx_data = self.calculate_adx_simple(highs, lows, closes)
        momentum = (closes[-1] / closes[-self.momentum_period] - 1) if closes[-self.momentum_period] != 0 else 0
        
        current_pos = context.position.qty
        
        # Exit logic
        if current_pos != 0:
            self.bars_held += 1
            
            exit_signal = False
            exit_reason = None
            
            # Exit if trend weakens
            if adx_data["adx"] < self.adx_threshold:
                exit_signal = True
                exit_reason = "Exit:Weak_Trend"
            elif momentum < 0 and adx_data["di_minus"] > adx_data["di_plus"]:
                exit_signal = True
                exit_reason = "Exit:Trend_Reversal"
            elif self.bars_held >= self.time_stop:
                exit_signal = True
                exit_reason = "Exit:TimeStop"
            
            if exit_signal:
                context.close(reason=exit_reason)
                context.log("info", f"Position closed: {exit_reason}")
                self.bars_held = 0
                return
        
        # Entry logic - strong uptrend with positive momentum
        if (current_pos == 0 and 
            adx_data["adx"] > self.adx_threshold and
            adx_data["di_plus"] > adx_data["di_minus"] and
            momentum > 0):
            
            qty = context.size.from_notional(self.notional, bar.close)
            context.buy(qty, reason="Entry:Strong_Uptrend")
            context.tag_trade("ADX_Momentum")
            
            context.log("info", f"Strong trend entry: ADX={adx_data['adx']:.2f}, "
                               f"momentum={momentum:.4f}")
            
            context.record("adx", adx_data["adx"])
            context.record("momentum", momentum)
    
    def on_trade_open(self, context, trade):
        self.bars_held = 0


@register_strategy("dual_momentum")
class DualMomentumStrategy(Strategy):
    """
    Dual momentum strategy combining absolute and relative momentum.
    
    Absolute momentum: Asset trending up vs itself
    Relative momentum: Asset outperforming benchmark (SMA)
    Both must be positive to enter
    """
    
    name = "dual_momentum"
    
    @classmethod
    def param_schema(cls):
        return {
            "lookback": {"type": "int", "min": 30, "max": 120, "default": 60},
            "benchmark_period": {"type": "int", "min": 30, "max": 120, "default": 60},
            "notional": {"type": "float", "min": 100, "default": 10_000.0}
        }
    
    def on_init(self, context):
        self.lookback = int(self.params.get("lookback", 60))
        self.benchmark_period = int(self.params.get("benchmark_period", 60))
        self.notional = float(self.params.get("notional", 10_000.0))
        
        context.log("info", f"Dual Momentum initialized: lookback={self.lookback}")
    
    def on_bar(self, context, symbol: str, bar: Bar):
        # Need enough history
        if context.bar_index < max(self.lookback, self.benchmark_period):
            return
        
        closes = context.data.history(symbol, "close", max(self.lookback, self.benchmark_period) + 1)
        if len(closes) < max(self.lookback, self.benchmark_period) + 1:
            return
        
        # Absolute momentum (trend)
        abs_momentum = closes[-1] / closes[-self.lookback] - 1 if closes[-self.lookback] != 0 else 0
        
        # Relative momentum (vs benchmark - moving average)
        benchmark = np.mean(closes[-self.benchmark_period:])
        rel_momentum = (closes[-1] - benchmark) / benchmark if benchmark != 0 else 0
        
        current_pos = context.position.qty
        
        # Strategy logic: only hold when both momentums are positive
        both_positive = abs_momentum > 0 and rel_momentum > 0
        
        if current_pos == 0 and both_positive:
            # Enter long position
            qty = context.size.from_notional(self.notional, bar.close)
            context.buy(qty, reason="Entry:Dual_Momentum_Positive")
            context.tag_trade("Dual_Momentum")
            
            context.log("info", f"Dual momentum entry: abs={abs_momentum:.4f}, "
                               f"rel={rel_momentum:.4f}")
            
        elif current_pos > 0 and not both_positive:
            # Exit position
            context.close(reason="Exit:Dual_Momentum_Negative")
            context.log("info", f"Dual momentum exit: abs={abs_momentum:.4f}, "
                               f"rel={rel_momentum:.4f}")
        
        # Record metrics
        context.record("abs_momentum", abs_momentum)
        context.record("rel_momentum", rel_momentum)
        context.record("dual_signal", 1 if both_positive else 0)


# Factory functions for easy strategy creation
def create_momentum_classic(lookback: int = 20, threshold: float = 0.05, 
                          notional: float = 10_000.0) -> ClassicMomentumStrategy:
    """Create a classic momentum strategy."""
    return ClassicMomentumStrategy(params={
        "lookback": lookback,
        "threshold": threshold,
        "notional": notional
    })


def create_rsi_momentum(rsi_period: int = 14, rsi_threshold: float = 50,
                       notional: float = 10_000.0) -> RSIMomentumStrategy:
    """Create an RSI momentum strategy."""
    return RSIMomentumStrategy(params={
        "rsi_period": rsi_period,
        "rsi_threshold": rsi_threshold,
        "notional": notional
    })


def create_breakout_donchian(lookback: int = 20, 
                           notional: float = 10_000.0) -> DonchianBreakoutStrategy:
    """Create a Donchian breakout strategy."""
    return DonchianBreakoutStrategy(params={
        "lookback": lookback,
        "notional": notional
    })


def create_adx_momentum(adx_threshold: float = 25, 
                       notional: float = 10_000.0) -> ADXMomentumStrategy:
    """Create an ADX-filtered momentum strategy."""
    return ADXMomentumStrategy(params={
        "adx_threshold": adx_threshold,
        "notional": notional
    })


def create_dual_momentum(lookback: int = 60, 
                        notional: float = 10_000.0) -> DualMomentumStrategy:
    """Create a dual momentum strategy."""
    return DualMomentumStrategy(params={
        "lookback": lookback,
        "notional": notional
    })