"""
Example of using cross-validation for strategy evaluation.

Demonstrates proper time-series CV with purging and embargo.
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cross_validation import (
    TimeSeriesSplit, ExpandingWindowSplit, RollingWindowSplit,
    WalkForwardSplit, MonteCarloSplit, PurgeEmbargo
)


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='1h')
    
    # Random walk for price
    returns = np.random.randn(n_samples) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    high = close * (1 + np.abs(np.random.randn(n_samples) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n_samples) * 0.005))
    open = np.roll(close, 1)
    open[0] = close[0]
    volume = np.random.exponential(1000, n_samples)
    
    return pd.DataFrame({
        'open': open,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


def calculate_returns(data: pd.DataFrame, horizon: int = 20) -> pd.Series:
    """Calculate forward returns as labels."""
    return data['close'].pct_change(horizon).shift(-horizon)


def simple_strategy(train_data: pd.DataFrame, test_data: pd.DataFrame,
                   train_labels: pd.Series, test_labels: pd.Series) -> dict:
    """
    Simple momentum strategy for demonstration.
    
    Returns performance metrics.
    """
    # Calculate features
    train_momentum = train_data['close'].pct_change(20)
    test_momentum = test_data['close'].pct_change(20)
    
    # Simple threshold strategy
    threshold = train_momentum.quantile(0.7)
    
    # Generate signals
    train_signals = (train_momentum > threshold).astype(int)
    test_signals = (test_momentum > threshold).astype(int)
    
    # Calculate returns
    train_returns = train_signals * train_labels
    test_returns = test_signals * test_labels
    
    # Metrics
    def calculate_metrics(returns):
        clean_returns = returns.dropna()
        if len(clean_returns) == 0:
            return {'sharpe': 0, 'mean_return': 0, 'hit_rate': 0}
        
        return {
            'sharpe': clean_returns.mean() / clean_returns.std() * np.sqrt(252 * 24) if clean_returns.std() > 0 else 0,
            'mean_return': clean_returns.mean(),
            'hit_rate': (clean_returns > 0).mean()
        }
    
    return {
        'train': calculate_metrics(train_returns),
        'test': calculate_metrics(test_returns)
    }


def run_cv_example():
    """Run cross-validation example."""
    
    # Generate data
    print("Generating sample data...")
    data = generate_sample_data(n_samples=2000)
    labels = calculate_returns(data, horizon=20)
    
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print()
    
    # 1. Simple Train/Test Split
    print("=" * 60)
    print("1. Simple Time Series Split")
    print("-" * 60)
    
    ts_split = TimeSeriesSplit(test_ratio=0.2, gap=20)
    folds = ts_split.split(len(data))
    
    for fold in folds:
        train_data = data.iloc[fold.train_start:fold.train_end]
        test_data = data.iloc[fold.test_start:fold.test_end]
        train_labels = labels.iloc[fold.train_start:fold.train_end]
        test_labels = labels.iloc[fold.test_start:fold.test_end]
        
        results = simple_strategy(train_data, test_data, train_labels, test_labels)
        
        print(f"Train: {fold.train_start:4d}-{fold.train_end:4d} ({len(train_data):4d} samples)")
        print(f"Test:  {fold.test_start:4d}-{fold.test_end:4d} ({len(test_data):4d} samples)")
        print(f"Train Sharpe: {results['train']['sharpe']:.3f}")
        print(f"Test Sharpe:  {results['test']['sharpe']:.3f}")
    
    # 2. Expanding Window
    print("\n" + "=" * 60)
    print("2. Expanding Window Split")
    print("-" * 60)
    
    exp_split = ExpandingWindowSplit(n_splits=3, gap=20)
    folds = exp_split.split(len(data))
    
    test_sharpes = []
    for i, fold in enumerate(folds):
        train_data = data.iloc[fold.train_start:fold.train_end]
        test_data = data.iloc[fold.test_start:fold.test_end]
        train_labels = labels.iloc[fold.train_start:fold.train_end]
        test_labels = labels.iloc[fold.test_start:fold.test_end]
        
        results = simple_strategy(train_data, test_data, train_labels, test_labels)
        test_sharpes.append(results['test']['sharpe'])
        
        print(f"Fold {i+1}:")
        print(f"  Train: {fold.train_start:4d}-{fold.train_end:4d} ({len(train_data):4d} samples)")
        print(f"  Test:  {fold.test_start:4d}-{fold.test_end:4d} ({len(test_data):4d} samples)")
        print(f"  Test Sharpe: {results['test']['sharpe']:.3f}")
    
    print(f"\nMean Test Sharpe: {np.mean(test_sharpes):.3f}")
    print(f"Std Test Sharpe:  {np.std(test_sharpes):.3f}")
    
    # 3. Rolling Window
    print("\n" + "=" * 60)
    print("3. Rolling Window Split")
    print("-" * 60)
    
    roll_split = RollingWindowSplit(n_splits=3, train_size=800, test_size=200, gap=20)
    folds = roll_split.split(len(data))
    
    test_sharpes = []
    for i, fold in enumerate(folds):
        train_data = data.iloc[fold.train_start:fold.train_end]
        test_data = data.iloc[fold.test_start:fold.test_end]
        train_labels = labels.iloc[fold.train_start:fold.train_end]
        test_labels = labels.iloc[fold.test_start:fold.test_end]
        
        results = simple_strategy(train_data, test_data, train_labels, test_labels)
        test_sharpes.append(results['test']['sharpe'])
        
        print(f"Fold {i+1}:")
        print(f"  Train: {fold.train_start:4d}-{fold.train_end:4d} ({len(train_data):4d} samples)")
        print(f"  Test:  {fold.test_start:4d}-{fold.test_end:4d} ({len(test_data):4d} samples)")
        print(f"  Test Sharpe: {results['test']['sharpe']:.3f}")
    
    print(f"\nMean Test Sharpe: {np.mean(test_sharpes):.3f}")
    print(f"Std Test Sharpe:  {np.std(test_sharpes):.3f}")
    
    # 4. Walk-Forward
    print("\n" + "=" * 60)
    print("4. Walk-Forward Split")
    print("-" * 60)
    
    wf_split = WalkForwardSplit(n_splits=4, train_periods=500, test_periods=200, 
                                anchored=False, gap=20)
    folds = wf_split.split(len(data))
    
    test_sharpes = []
    for i, fold in enumerate(folds):
        train_data = data.iloc[fold.train_start:fold.train_end]
        test_data = data.iloc[fold.test_start:fold.test_end]
        train_labels = labels.iloc[fold.train_start:fold.train_end]
        test_labels = labels.iloc[fold.test_start:fold.test_end]
        
        results = simple_strategy(train_data, test_data, train_labels, test_labels)
        test_sharpes.append(results['test']['sharpe'])
        
        print(f"Walk {i+1}:")
        print(f"  Train: {fold.train_start:4d}-{fold.train_end:4d}")
        print(f"  Test:  {fold.test_start:4d}-{fold.test_end:4d}")
        print(f"  Test Sharpe: {results['test']['sharpe']:.3f}")
    
    print(f"\nMean Test Sharpe: {np.mean(test_sharpes):.3f}")
    print(f"Std Test Sharpe:  {np.std(test_sharpes):.3f}")
    
    # 5. Monte Carlo
    print("\n" + "=" * 60)
    print("5. Monte Carlo Split (10 random samples)")
    print("-" * 60)
    
    mc_split = MonteCarloSplit(n_splits=10, test_ratio=0.2, min_train_ratio=0.4, 
                               gap=20, seed=42)
    folds = mc_split.split(len(data))
    
    test_sharpes = []
    for i, fold in enumerate(folds):
        train_data = data.iloc[fold.train_start:fold.train_end]
        test_data = data.iloc[fold.test_start:fold.test_end]
        train_labels = labels.iloc[fold.train_start:fold.train_end]
        test_labels = labels.iloc[fold.test_start:fold.test_end]
        
        results = simple_strategy(train_data, test_data, train_labels, test_labels)
        test_sharpes.append(results['test']['sharpe'])
        
        if i < 5:  # Show first 5 only
            print(f"Sample {i+1}:")
            print(f"  Train: {fold.train_start:4d}-{fold.train_end:4d} ({len(train_data):4d} samples)")
            print(f"  Test:  {fold.test_start:4d}-{fold.test_end:4d} ({len(test_data):4d} samples)")
            print(f"  Test Sharpe: {results['test']['sharpe']:.3f}")
    
    print(f"\nResults from {len(test_sharpes)} Monte Carlo samples:")
    print(f"Mean Test Sharpe:   {np.mean(test_sharpes):.3f}")
    print(f"Median Test Sharpe: {np.median(test_sharpes):.3f}")
    print(f"Std Test Sharpe:    {np.std(test_sharpes):.3f}")
    print(f"5th percentile:     {np.percentile(test_sharpes, 5):.3f}")
    print(f"95th percentile:    {np.percentile(test_sharpes, 95):.3f}")
    
    # 6. Demonstrate Purging
    print("\n" + "=" * 60)
    print("6. Purging and Embargo Example")
    print("-" * 60)
    
    # Create a simple split
    train_indices = np.arange(0, 1600)
    test_indices = np.arange(1600, 1800)
    
    # Apply purging with 20-bar label horizon
    purger = PurgeEmbargo(label_horizon=20)
    clean_train, purged, embargoed = purger.apply(
        train_indices, test_indices, len(data)
    )
    
    print(f"Original train size: {len(train_indices)}")
    print(f"Clean train size:    {len(clean_train)}")
    print(f"Purged samples:      {len(purged)}")
    print(f"Embargoed samples:   {len(embargoed)}")
    print(f"Total removed:       {len(purged) + len(embargoed)}")
    print(f"Removal percentage:  {100 * (len(purged) + len(embargoed)) / len(train_indices):.1f}%")


if __name__ == "__main__":
    run_cv_example()