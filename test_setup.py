"""
Test script to verify qupy setup is working correctly.
Run this before using the cookbook notebooks.
"""

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        # Core imports
        import pandas as pd
        import numpy as np
        print("✅ pandas, numpy")
        
        # Data loader
        from utils.data_loader import load_binance_data, generate_synthetic_data
        print("✅ data_loader")
        
        # Indicators - correct imports
        from indicators.trend import sma, ema, adx
        from indicators.momentum import rsi, macd, stochastic, williams_r, cci
        from indicators.volatility import bollinger_bands, atr, keltner_channels, donchian_channels
        print("✅ indicators (trend, momentum, volatility)")
        
        # Strategy engine
        from engine.strategy_base import Strategy, Bar
        print("✅ strategy engine")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    print("\nTesting data loading...")
    
    try:
        from utils.data_loader import load_binance_data, generate_synthetic_data
        
        # Try real data first
        try:
            data = load_binance_data('data/CAKEUSDT.csv')
            print(f"✅ Real Binance data loaded: {len(data)} rows")
            return data
        except:
            # Fallback to synthetic
            data = generate_synthetic_data(n_days=100)
            print(f"✅ Synthetic data generated: {len(data)} rows")
            return data
            
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return None

def test_indicators(data):
    """Test indicator calculations."""
    print("\nTesting indicators...")
    
    try:
        from indicators.trend import sma, ema
        from indicators.momentum import rsi, macd
        from indicators.volatility import bollinger_bands
        
        # Test basic indicators
        sma_20 = sma(data['close'], 20)
        print(f"✅ SMA(20) - last value: {sma_20.iloc[-1]:.4f}")
        
        rsi_14 = rsi(data['close'].values, 14)
        print(f"✅ RSI(14) - last value: {rsi_14[-1]:.2f}")
        
        macd_result = macd(data['close'].values)
        print(f"✅ MACD - histogram last: {macd_result['histogram'][-1]:.4f}")
        
        bb_result = bollinger_bands(data['close'].values)
        print(f"✅ Bollinger Bands - upper: {bb_result['upper'][-1]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Indicator error: {e}")
        return False

def test_frequency_detection(data):
    """Test data frequency detection."""
    print("\nTesting frequency detection...")
    
    try:
        import numpy as np
        
        # Calculate data frequency
        time_diff = (data.index[1] - data.index[0]).total_seconds() / 60
        periods_per_day = 24 * 60 / time_diff
        annualization_factor = np.sqrt(365 * periods_per_day)
        
        print(f"✅ Data frequency: {time_diff:.0f} minutes")
        print(f"✅ Periods per day: {periods_per_day:.1f}")
        print(f"✅ Annualization factor: {annualization_factor:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Frequency detection error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Qupy Setup Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Setup failed - import errors")
        return False
    
    # Test data loading
    data = test_data_loading()
    if data is None:
        print("\n❌ Setup failed - data loading errors")
        return False
    
    # Test indicators
    if not test_indicators(data):
        print("\n❌ Setup failed - indicator errors")
        return False
    
    # Test frequency detection
    if not test_frequency_detection(data):
        print("\n❌ Setup failed - frequency detection errors")
        return False
    
    print("\n🎉 All tests passed! Your setup is ready.")
    print("\nYou can now run the cookbook notebooks:")
    print("  - 01_getting_started.ipynb")
    print("  - 02_momentum_strategies.ipynb")
    print("  - 03_mean_reversion.ipynb")
    print("  - And more...")
    
    print("\n📝 Common imports for notebooks:")
    print("```python")
    print("from utils.data_loader import load_binance_data")
    print("from indicators.trend import sma, ema, adx")
    print("from indicators.momentum import rsi, macd, stochastic")
    print("from indicators.volatility import bollinger_bands, atr")
    print("```")
    
    return True

if __name__ == "__main__":
    main()