from engine.strategy_base import Strategy, Bar, register_strategy


@register_strategy("buy_and_hold")
class BuyAndHoldStrategy(Strategy):
    """
    Simple Buy and Hold Strategy
    
    Buys at the first bar and holds until the end.
    Used as a benchmark for other strategies.
    """
    
    name = "buy_and_hold"
    
    @classmethod
    def param_schema(cls):
        return {
            "notional": {"type": "float", "min": 100, "default": 10_000.0}
        }
    
    def on_init(self, context):
        """Initialize strategy parameters"""
        self.notional = float(self.params.get("notional", 10_000.0))
        self.bought = False
        
        context.log("info", f"Buy & Hold Strategy initialized: Size=${self.notional}")
    
    def on_start(self, context):
        """Called at backtest start"""
        context.log("info", "Buy & Hold strategy started")
    
    def on_bar(self, context, symbol: str, bar: Bar):
        """Buy once and hold"""
        if not self.bought and context.bar_index == 0:
            qty = context.size.from_notional(self.notional, bar.close)
            context.buy(qty, reason="Initial_Buy", size_mode="qty")
            self.bought = True
            
            context.log("info", f"🟢 BUY & HOLD: Bought {qty:.4f} at ${bar.close:.4f}")
    
    def on_trade_close(self, context, trade):
        """Called when a trade closes"""
        if hasattr(trade, 'pnl_abs') and trade.pnl_abs is not None:
            context.log("info", f"💰 Buy & Hold trade closed: PnL=${trade.pnl_abs:.2f}")