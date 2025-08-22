"""
Example: Loading and analyzing Binance data with qupy.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('..')

from utils.data_loader import load_binance_data, prepare_features
from indicators.trend import sma, ema
from indicators.momentum import rsi, macd
from indicators.volatility import bollinger_bands, atr


def main():
    # Load CAKE/USDT data
    print("Loading CAKE/USDT data...")
    df = load_binance_data('../data/CAKEUSDT.csv')
    
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Calculate indicators
    print("\nCalculating indicators...")
    
    # Trend indicators
    df['sma_20'] = sma(df['close'], 20)
    df['ema_20'] = ema(df['close'], 20)
    
    # MACD
    macd_result = macd(df['close'].values)
    df['macd'] = macd_result['macd']
    df['macd_signal'] = macd_result['signal']
    df['macd_hist'] = macd_result['histogram']
    
    # RSI
    df['rsi'] = rsi(df['close'].values, 14)
    
    # Bollinger Bands
    bb = bollinger_bands(df['close'].values)
    df['bb_upper'] = bb['upper']
    df['bb_middle'] = bb['middle']
    df['bb_lower'] = bb['lower']
    
    # ATR
    df['atr'] = atr(df['high'].values, df['low'].values, df['close'].values)
    
    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # Price and moving averages
    ax = axes[0]
    ax.plot(df.index, df['close'], label='Close', color='black', linewidth=1)
    ax.plot(df.index, df['sma_20'], label='SMA(20)', alpha=0.7)
    ax.plot(df.index, df['ema_20'], label='EMA(20)', alpha=0.7)
    ax.fill_between(df.index, df['bb_upper'], df['bb_lower'], alpha=0.1, color='gray')
    ax.set_title('CAKE/USDT Price with Indicators')
    ax.set_ylabel('Price (USDT)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Volume
    ax = axes[1]
    colors = ['green' if c >= o else 'red' for c, o in zip(df['close'], df['open'])]
    ax.bar(df.index, df['volume'], color=colors, alpha=0.5, width=0.8)
    ax.set_title('Volume')
    ax.set_ylabel('Volume')
    ax.grid(True, alpha=0.3)
    
    # RSI
    ax = axes[2]
    ax.plot(df.index, df['rsi'], label='RSI(14)', color='purple')
    ax.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=30, color='green', linestyle='--', alpha=0.5)
    ax.fill_between(df.index, 70, 100, alpha=0.1, color='red')
    ax.fill_between(df.index, 0, 30, alpha=0.1, color='green')
    ax.set_title('RSI')
    ax.set_ylabel('RSI')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # MACD
    ax = axes[3]
    ax.plot(df.index, df['macd'], label='MACD', color='blue')
    ax.plot(df.index, df['macd_signal'], label='Signal', color='red')
    ax.bar(df.index, df['macd_hist'], label='Histogram', alpha=0.3, color='gray')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title('MACD')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cake_analysis.png', dpi=100)
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("Summary Statistics")
    print("="*50)
    
    # Recent values
    last_row = df.iloc[-1]
    print(f"\nLatest values:")
    print(f"  Close: ${last_row['close']:.4f}")
    print(f"  RSI: {last_row['rsi']:.2f}")
    print(f"  MACD Histogram: {last_row['macd_hist']:.4f}")
    
    # Price position relative to Bollinger Bands
    bb_position = (last_row['close'] - last_row['bb_lower']) / (last_row['bb_upper'] - last_row['bb_lower'])
    print(f"  BB Position: {bb_position:.2%} (0=lower, 100%=upper)")
    
    # Volatility
    returns = df['close'].pct_change()
    current_vol = returns.tail(20).std() * np.sqrt(365)
    print(f"  20-day Volatility: {current_vol:.2%} annualized")
    
    # Trend
    sma_trend = "Bullish" if last_row['close'] > last_row['sma_20'] else "Bearish"
    print(f"  Trend (vs SMA20): {sma_trend}")
    
    # Performance metrics
    print(f"\nPerformance metrics:")
    total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    print(f"  Total Return: {total_return:.2f}%")
    
    # Sharpe ratio (simplified)
    daily_returns = returns.dropna()
    if len(daily_returns) > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365)
        print(f"  Sharpe Ratio: {sharpe:.2f}")
    
    # Max drawdown
    cummax = df['close'].cummax()
    drawdown = (df['close'] - cummax) / cummax
    max_dd = drawdown.min() * 100
    print(f"  Max Drawdown: {max_dd:.2f}%")
    
    # Trading signals
    print(f"\n" + "="*50)
    print("Current Trading Signals")
    print("="*50)
    
    # RSI signal
    if last_row['rsi'] > 70:
        print("  RSI: OVERBOUGHT - Consider selling")
    elif last_row['rsi'] < 30:
        print("  RSI: OVERSOLD - Consider buying")
    else:
        print("  RSI: NEUTRAL")
    
    # MACD signal
    if last_row['macd_hist'] > 0 and df['macd_hist'].iloc[-2] <= 0:
        print("  MACD: BULLISH CROSSOVER - Buy signal")
    elif last_row['macd_hist'] < 0 and df['macd_hist'].iloc[-2] >= 0:
        print("  MACD: BEARISH CROSSOVER - Sell signal")
    elif last_row['macd_hist'] > 0:
        print("  MACD: BULLISH")
    else:
        print("  MACD: BEARISH")
    
    # Bollinger Band signal
    if last_row['close'] > last_row['bb_upper']:
        print("  BB: Price ABOVE upper band - Overbought")
    elif last_row['close'] < last_row['bb_lower']:
        print("  BB: Price BELOW lower band - Oversold")
    else:
        print("  BB: Price within bands - Normal")
    
    # Save processed data
    output_file = 'cake_with_indicators.csv'
    df.to_csv(output_file)
    print(f"\n Processed data saved to {output_file}")


if __name__ == "__main__":
    main()