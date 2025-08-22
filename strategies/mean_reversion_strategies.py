"""
Mean reversion trading strategies implementations using the Strategy base class.

This module contains various mean reversion strategies from cookbook 03:
- Bollinger Bands mean reversion
- Z-score trading
- RSI oversold/overbought
- Combined mean reversion signals
"""

import numpy as np
from typing import List, Dict, Optional
from engine.strategy_base import Strategy, Bar, register_strategy


@register_strategy("bollinger_mean_reversion")
class BollingerMeanReversionStrategy(Strategy):
    """
    Bollinger Bands mean reversion strategy.
    
    Entry: Price touches bands (buy at lower, sell at upper)
    Exit: Price returns to middle band or time stop
    """
    
    name = "bollinger_mean_reversion"
    
    @classmethod
    def param_schema(cls):
        return {
            "bb_period": {"type": "int", "min": 10, "max": 50, "default": 20},
            "bb_std": {"type": "float", "min": 1.0, "max": 3.0, "default": 2.0},
            "entry_threshold": {"type": "float", "min": 0.01, "max": 0.2, "default": 0.05},
            "exit_threshold": {"type": "float", "min": 0.3, "max": 0.7, "default": 0.5},
            "notional": {"type": "float", "min": 100, "default": 10_000.0},
            "time_stop": {"type": "int", "min": 5, "default": 30}
        }
    
    def on_init(self, context):
        self.bb_period = int(self.params.get("bb_period", 20))
        self.bb_std = float(self.params.get("bb_std", 2.0))
        self.entry_threshold = float(self.params.get("entry_threshold", 0.05))  # %B threshold
        self.exit_threshold = float(self.params.get("exit_threshold", 0.5))
        self.notional = float(self.params.get("notional", 10_000.0))
        self.time_stop = int(self.params.get("time_stop", 30))
        
        self.bars_held = 0
        
        context.log("info", f"Bollinger Mean Reversion initialized: period={self.bb_period}, "
                           f"std={self.bb_std}")
    
    def calculate_bollinger_bands(self, closes: List[float]) -> Dict[str, float]:
        """Calculate Bollinger Bands and %B."""
        if len(closes) < self.bb_period:
            return {"upper": 0, "middle": 0, "lower": 0, "pctb": 0.5}
        
        recent_closes = closes[-self.bb_period:]
        sma = np.mean(recent_closes)
        std = np.std(recent_closes)
        
        upper = sma + (self.bb_std * std)
        lower = sma - (self.bb_std * std)
        
        # %B calculation
        current_price = closes[-1]
        if upper != lower:
            pctb = (current_price - lower) / (upper - lower)
        else:
            pctb = 0.5
        
        return {
            "upper": upper,
            "middle": sma,
            "lower": lower,
            "pctb": pctb
        }
    
    def on_bar(self, context, symbol: str, bar: Bar):
        # Need enough history
        if context.bar_index < self.bb_period:
            return
        
        closes = context.data.history(symbol, "close", self.bb_period)
        if len(closes) < self.bb_period:
            return
        
        bb = self.calculate_bollinger_bands(closes)
        current_pos = context.position.qty
        
        # Exit logic
        if current_pos != 0:
            self.bars_held += 1
            
            exit_signal = False
            exit_reason = None
            
            # Exit when price returns to middle area
            if self.exit_threshold * 0.8 <= bb["pctb"] <= self.exit_threshold * 1.2:
                exit_signal = True
                exit_reason = "Exit:Return_To_Middle"
            elif self.bars_held >= self.time_stop:
                exit_signal = True
                exit_reason = "Exit:TimeStop"
            
            if exit_signal:
                context.close(reason=exit_reason)
                context.log("info", f"Position closed: {exit_reason}, %B={bb['pctb']:.3f}")
                self.bars_held = 0
                return
        
        # Entry logic
        if current_pos == 0:
            # Buy when oversold (lower band touch)
            if bb["pctb"] < self.entry_threshold:
                qty = context.size.from_notional(self.notional, bar.close)
                context.buy(qty, reason="Entry:Oversold")
                context.tag_trade("BB_MeanReversion_Long")
                
                context.log("info", f"Long entry (oversold): %B={bb['pctb']:.3f}")
                
            # Short when overbought (upper band touch)
            elif bb["pctb"] > (1 - self.entry_threshold):
                qty = context.size.from_notional(self.notional, bar.close)
                context.sell(qty, reason="Entry:Overbought")
                context.tag_trade("BB_MeanReversion_Short")
                
                context.log("info", f"Short entry (overbought): %B={bb['pctb']:.3f}")
        
        # Record metrics
        context.record("bb_pctb", bb["pctb"])
        context.record("bb_upper", bb["upper"])
        context.record("bb_lower", bb["lower"])
    
    def on_trade_open(self, context, trade):
        self.bars_held = 0


@register_strategy("zscore_mean_reversion")
class ZScoreMeanReversionStrategy(Strategy):
    """
    Z-score mean reversion strategy.
    
    Normalizes price movements and trades extreme deviations.
    Entry: |Z-score| > entry threshold
    Exit: |Z-score| < exit threshold
    """
    
    name = "zscore_mean_reversion"
    
    @classmethod
    def param_schema(cls):
        return {
            "lookback": {"type": "int", "min": 20, "max": 200, "default": 50},
            "entry_zscore": {"type": "float", "min": 1.5, "max": 3.0, "default": 2.0},
            "exit_zscore": {"type": "float", "min": 0.2, "max": 1.0, "default": 0.5},
            "notional": {"type": "float", "min": 100, "default": 10_000.0},
            "time_stop": {"type": "int", "min": 5, "default": 40}
        }
    
    def on_init(self, context):
        self.lookback = int(self.params.get("lookback", 50))
        self.entry_zscore = float(self.params.get("entry_zscore", 2.0))
        self.exit_zscore = float(self.params.get("exit_zscore", 0.5))
        self.notional = float(self.params.get("notional", 10_000.0))
        self.time_stop = int(self.params.get("time_stop", 40))
        
        self.bars_held = 0
        
        context.log("info", f"Z-Score Mean Reversion initialized: lookback={self.lookback}, "
                           f"entry_z={self.entry_zscore}")
    
    def calculate_zscore(self, closes: List[float]) -> float:
        """Calculate Z-score for the most recent price."""
        if len(closes) < self.lookback:
            return 0.0
        
        recent_closes = closes[-self.lookback:]
        mean_price = np.mean(recent_closes)
        std_price = np.std(recent_closes)
        
        if std_price == 0:
            return 0.0
        
        current_price = closes[-1]
        zscore = (current_price - mean_price) / std_price
        return zscore
    
    def on_bar(self, context, symbol: str, bar: Bar):
        # Need enough history
        if context.bar_index < self.lookback:
            return
        
        closes = context.data.history(symbol, "close", self.lookback)
        if len(closes) < self.lookback:
            return
        
        zscore = self.calculate_zscore(closes)
        current_pos = context.position.qty
        
        # Exit logic
        if current_pos != 0:
            self.bars_held += 1
            
            exit_signal = False
            exit_reason = None
            
            # Exit when Z-score returns to normal range
            if abs(zscore) < self.exit_zscore:
                exit_signal = True
                exit_reason = "Exit:ZScore_Normal"
            elif self.bars_held >= self.time_stop:
                exit_signal = True
                exit_reason = "Exit:TimeStop"
            
            if exit_signal:
                context.close(reason=exit_reason)
                context.log("info", f"Position closed: {exit_reason}, Z-score={zscore:.3f}")
                self.bars_held = 0
                return
        
        # Entry logic
        if current_pos == 0:
            if zscore < -self.entry_zscore:
                # Extremely oversold - buy
                qty = context.size.from_notional(self.notional, bar.close)
                context.buy(qty, reason="Entry:Extremely_Oversold")
                context.tag_trade("ZScore_MeanReversion_Long")
                
                context.log("info", f"Long entry (oversold): Z-score={zscore:.3f}")
                
            elif zscore > self.entry_zscore:
                # Extremely overbought - sell
                qty = context.size.from_notional(self.notional, bar.close)
                context.sell(qty, reason="Entry:Extremely_Overbought")
                context.tag_trade("ZScore_MeanReversion_Short")
                
                context.log("info", f"Short entry (overbought): Z-score={zscore:.3f}")
        
        context.record("zscore", zscore)
    
    def on_trade_open(self, context, trade):
        self.bars_held = 0


@register_strategy("rsi_mean_reversion")
class RSIMeanReversionStrategy(Strategy):
    """
    RSI mean reversion strategy using overbought/oversold levels.
    
    Entry: RSI < oversold OR RSI > overbought
    Exit: RSI returns to neutral zone OR time stop
    """
    
    name = "rsi_mean_reversion"
    
    @classmethod
    def param_schema(cls):
        return {
            "rsi_period": {"type": "int", "min": 5, "max": 30, "default": 14},
            "oversold": {"type": "float", "min": 15, "max": 35, "default": 30},
            "overbought": {"type": "float", "min": 65, "max": 85, "default": 70},
            "exit_neutral": {"type": "float", "min": 45, "max": 55, "default": 50},
            "notional": {"type": "float", "min": 100, "default": 10_000.0},
            "time_stop": {"type": "int", "min": 5, "default": 25},
            "volume_filter": {"type": "bool", "default": True}
        }
    
    def on_init(self, context):
        self.rsi_period = int(self.params.get("rsi_period", 14))
        self.oversold = float(self.params.get("oversold", 30))
        self.overbought = float(self.params.get("overbought", 70))
        self.exit_neutral = float(self.params.get("exit_neutral", 50))
        self.notional = float(self.params.get("notional", 10_000.0))
        self.time_stop = int(self.params.get("time_stop", 25))
        self.volume_filter = bool(self.params.get("volume_filter", True))
        
        self.bars_held = 0
        
        context.log("info", f"RSI Mean Reversion initialized: RSI period={self.rsi_period}, "
                           f"oversold={self.oversold}, overbought={self.overbought}")
    
    def calculate_rsi(self, closes: List[float]) -> float:
        """Calculate RSI."""
        if len(closes) < self.rsi_period + 1:
            return 50.0
        
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100.0
        if avg_gain == 0:
            return 0.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def check_volume_confirmation(self, context, symbol: str) -> bool:
        """Check if volume confirms the signal."""
        if not self.volume_filter:
            return True
        
        if context.bar_index < 20:
            return True
        
        volumes = context.data.history(symbol, "volume", 20)
        if len(volumes) < 20:
            return True
        
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[:-1])
        
        # Require above-average volume
        return current_volume > avg_volume * 1.2
    
    def on_bar(self, context, symbol: str, bar: Bar):
        # Need enough history
        if context.bar_index < self.rsi_period + 1:
            return
        
        closes = context.data.history(symbol, "close", self.rsi_period + 1)
        if len(closes) < self.rsi_period + 1:
            return
        
        rsi = self.calculate_rsi(closes)
        volume_confirmed = self.check_volume_confirmation(context, symbol)
        current_pos = context.position.qty
        
        # Exit logic
        if current_pos != 0:
            self.bars_held += 1
            
            exit_signal = False
            exit_reason = None
            
            # Exit based on position direction and RSI
            if current_pos > 0 and rsi > self.exit_neutral:
                exit_signal = True
                exit_reason = "Exit:RSI_Neutral_Long"
            elif current_pos < 0 and rsi < self.exit_neutral:
                exit_signal = True
                exit_reason = "Exit:RSI_Neutral_Short"
            elif self.bars_held >= self.time_stop:
                exit_signal = True
                exit_reason = "Exit:TimeStop"
            
            if exit_signal:
                context.close(reason=exit_reason)
                context.log("info", f"Position closed: {exit_reason}, RSI={rsi:.2f}")
                self.bars_held = 0
                return
        
        # Entry logic
        if current_pos == 0 and volume_confirmed:
            if rsi < self.oversold:
                # Buy oversold
                qty = context.size.from_notional(self.notional, bar.close)
                context.buy(qty, reason="Entry:RSI_Oversold")
                context.tag_trade("RSI_MeanReversion_Long")
                
                context.log("info", f"Long entry (oversold): RSI={rsi:.2f}")
                
            elif rsi > self.overbought:
                # Sell overbought
                qty = context.size.from_notional(self.notional, bar.close)
                context.sell(qty, reason="Entry:RSI_Overbought")
                context.tag_trade("RSI_MeanReversion_Short")
                
                context.log("info", f"Short entry (overbought): RSI={rsi:.2f}")
        
        context.record("rsi", rsi)
    
    def on_trade_open(self, context, trade):
        self.bars_held = 0


@register_strategy("combined_mean_reversion")
class CombinedMeanReversionStrategy(Strategy):
    """
    Combined mean reversion strategy using multiple indicators.
    
    Uses ensemble voting from:
    - Bollinger Bands %B
    - Z-Score
    - RSI
    
    Entry: Majority vote (2+ signals agree)
    Exit: Majority vote changes or time stop
    """
    
    name = "combined_mean_reversion"
    
    @classmethod
    def param_schema(cls):
        return {
            "bb_period": {"type": "int", "min": 15, "max": 30, "default": 20},
            "bb_std": {"type": "float", "min": 1.5, "max": 2.5, "default": 2.0},
            "zscore_lookback": {"type": "int", "min": 30, "max": 80, "default": 50},
            "zscore_threshold": {"type": "float", "min": 1.5, "max": 2.5, "default": 2.0},
            "rsi_period": {"type": "int", "min": 10, "max": 20, "default": 14},
            "rsi_oversold": {"type": "float", "min": 20, "max": 35, "default": 30},
            "rsi_overbought": {"type": "float", "min": 65, "max": 80, "default": 70},
            "notional": {"type": "float", "min": 100, "default": 10_000.0},
            "time_stop": {"type": "int", "min": 10, "default": 35}
        }
    
    def on_init(self, context):
        self.bb_period = int(self.params.get("bb_period", 20))
        self.bb_std = float(self.params.get("bb_std", 2.0))
        self.zscore_lookback = int(self.params.get("zscore_lookback", 50))
        self.zscore_threshold = float(self.params.get("zscore_threshold", 2.0))
        self.rsi_period = int(self.params.get("rsi_period", 14))
        self.rsi_oversold = float(self.params.get("rsi_oversold", 30))
        self.rsi_overbought = float(self.params.get("rsi_overbought", 70))
        self.notional = float(self.params.get("notional", 10_000.0))
        self.time_stop = int(self.params.get("time_stop", 35))
        
        self.bars_held = 0
        
        context.log("info", "Combined Mean Reversion initialized")
    
    def get_bb_signal(self, closes: List[float]) -> int:
        """Get Bollinger Bands signal (-1, 0, 1)."""
        if len(closes) < self.bb_period:
            return 0
        
        recent_closes = closes[-self.bb_period:]
        sma = np.mean(recent_closes)
        std = np.std(recent_closes)
        
        upper = sma + (self.bb_std * std)
        lower = sma - (self.bb_std * std)
        
        current_price = closes[-1]
        if upper != lower:
            pctb = (current_price - lower) / (upper - lower)
        else:
            return 0
        
        if pctb < 0.05:  # Near lower band
            return 1  # Buy signal
        elif pctb > 0.95:  # Near upper band
            return -1  # Sell signal
        else:
            return 0  # Neutral
    
    def get_zscore_signal(self, closes: List[float]) -> int:
        """Get Z-score signal (-1, 0, 1)."""
        if len(closes) < self.zscore_lookback:
            return 0
        
        recent_closes = closes[-self.zscore_lookback:]
        mean_price = np.mean(recent_closes)
        std_price = np.std(recent_closes)
        
        if std_price == 0:
            return 0
        
        current_price = closes[-1]
        zscore = (current_price - mean_price) / std_price
        
        if zscore < -self.zscore_threshold:
            return 1  # Buy signal (oversold)
        elif zscore > self.zscore_threshold:
            return -1  # Sell signal (overbought)
        else:
            return 0
    
    def get_rsi_signal(self, closes: List[float]) -> int:
        """Get RSI signal (-1, 0, 1)."""
        if len(closes) < self.rsi_period + 1:
            return 0
        
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            rsi = 100.0
        elif avg_gain == 0:
            rsi = 0.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        if rsi < self.rsi_oversold:
            return 1  # Buy signal
        elif rsi > self.rsi_overbought:
            return -1  # Sell signal
        else:
            return 0
    
    def on_bar(self, context, symbol: str, bar: Bar):
        # Need enough history
        max_lookback = max(self.bb_period, self.zscore_lookback, self.rsi_period + 1)
        if context.bar_index < max_lookback:
            return
        
        closes = context.data.history(symbol, "close", max_lookback)
        if len(closes) < max_lookback:
            return
        
        # Get individual signals
        bb_signal = self.get_bb_signal(closes)
        zscore_signal = self.get_zscore_signal(closes)
        rsi_signal = self.get_rsi_signal(closes)
        
        # Ensemble voting
        vote_sum = bb_signal + zscore_signal + rsi_signal
        
        # Position based on majority vote
        if vote_sum >= 2:
            ensemble_signal = 1  # Long
        elif vote_sum <= -2:
            ensemble_signal = -1  # Short
        else:
            ensemble_signal = 0  # Neutral
        
        current_pos = context.position.qty
        
        # Exit logic
        if current_pos != 0:
            self.bars_held += 1
            
            exit_signal = False
            exit_reason = None
            
            # Exit if ensemble signal changes or time stop
            if (current_pos > 0 and ensemble_signal != 1) or \
               (current_pos < 0 and ensemble_signal != -1):
                exit_signal = True
                exit_reason = "Exit:Signal_Change"
            elif self.bars_held >= self.time_stop:
                exit_signal = True
                exit_reason = "Exit:TimeStop"
            
            if exit_signal:
                context.close(reason=exit_reason)
                context.log("info", f"Position closed: {exit_reason}, vote_sum={vote_sum}")
                self.bars_held = 0
                return
        
        # Entry logic
        if current_pos == 0:
            if ensemble_signal == 1:
                # Long signal
                qty = context.size.from_notional(self.notional, bar.close)
                context.buy(qty, reason="Entry:Ensemble_Long")
                context.tag_trade("Combined_MeanReversion_Long")
                
                context.log("info", f"Long entry: BB={bb_signal}, Z={zscore_signal}, "
                                   f"RSI={rsi_signal}, sum={vote_sum}")
                
            elif ensemble_signal == -1:
                # Short signal
                qty = context.size.from_notional(self.notional, bar.close)
                context.sell(qty, reason="Entry:Ensemble_Short")
                context.tag_trade("Combined_MeanReversion_Short")
                
                context.log("info", f"Short entry: BB={bb_signal}, Z={zscore_signal}, "
                                   f"RSI={rsi_signal}, sum={vote_sum}")
        
        # Record metrics
        context.record("bb_signal", bb_signal)
        context.record("zscore_signal", zscore_signal)
        context.record("rsi_signal", rsi_signal)
        context.record("vote_sum", vote_sum)
        context.record("ensemble_signal", ensemble_signal)
    
    def on_trade_open(self, context, trade):
        self.bars_held = 0


# Factory functions for easy strategy creation
def create_bollinger_mean_reversion(bb_period: int = 20, bb_std: float = 2.0,
                                   notional: float = 10_000.0) -> BollingerMeanReversionStrategy:
    """Create a Bollinger Bands mean reversion strategy."""
    return BollingerMeanReversionStrategy(params={
        "bb_period": bb_period,
        "bb_std": bb_std,
        "notional": notional
    })


def create_zscore_mean_reversion(lookback: int = 50, entry_zscore: float = 2.0,
                               notional: float = 10_000.0) -> ZScoreMeanReversionStrategy:
    """Create a Z-score mean reversion strategy."""
    return ZScoreMeanReversionStrategy(params={
        "lookback": lookback,
        "entry_zscore": entry_zscore,
        "notional": notional
    })


def create_rsi_mean_reversion(rsi_period: int = 14, oversold: float = 30,
                            overbought: float = 70, notional: float = 10_000.0) -> RSIMeanReversionStrategy:
    """Create an RSI mean reversion strategy."""
    return RSIMeanReversionStrategy(params={
        "rsi_period": rsi_period,
        "oversold": oversold,
        "overbought": overbought,
        "notional": notional
    })


def create_combined_mean_reversion(notional: float = 10_000.0) -> CombinedMeanReversionStrategy:
    """Create a combined mean reversion strategy."""
    return CombinedMeanReversionStrategy(params={
        "notional": notional
    })