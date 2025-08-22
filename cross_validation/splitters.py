"""
Simplified time-series cross-validation splitters.

All splitters maintain chronological order and support purging/embargo.
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class CVFold:
    """Container for a single CV fold."""
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    fold_id: int = 0


class TimeSeriesSplit:
    """
    Simple chronological train/test split.
    
    Most basic approach - single split with configurable ratio.
    """
    
    def __init__(self, test_ratio: float = 0.2, gap: int = 0):
        """
        Parameters:
        -----------
        test_ratio : float
            Portion of data for testing (0-1)
        gap : int
            Number of bars between train and test
        """
        self.test_ratio = test_ratio
        self.gap = gap
    
    def split(self, n_samples: int) -> List[CVFold]:
        """Generate single train/test split."""
        train_size = int(n_samples * (1 - self.test_ratio))
        train_end = train_size - self.gap
        test_start = train_size
        
        return [CVFold(
            train_start=0,
            train_end=train_end,
            test_start=test_start,
            test_end=n_samples,
            fold_id=0
        )]


class ExpandingWindowSplit:
    """
    Expanding window validation.
    
    Training window starts from beginning and expands with each fold.
    Good for parameter stability testing.
    """
    
    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None,
                 min_train_size: Optional[int] = None, gap: int = 0):
        """
        Parameters:
        -----------
        n_splits : int
            Number of CV folds
        test_size : int
            Fixed test window size (if None, auto-calculated)
        min_train_size : int
            Minimum training samples required
        gap : int
            Number of bars between train and test
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.gap = gap
    
    def split(self, n_samples: int) -> List[CVFold]:
        """Generate expanding window folds."""
        folds = []
        
        # Auto-calculate sizes
        if self.test_size is None:
            self.test_size = n_samples // (self.n_splits + 1)
        
        if self.min_train_size is None:
            self.min_train_size = self.test_size
        
        for i in range(self.n_splits):
            train_end = self.min_train_size + i * self.test_size - self.gap
            test_start = self.min_train_size + i * self.test_size
            test_end = test_start + self.test_size
            
            if test_end > n_samples:
                break
            
            folds.append(CVFold(
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                fold_id=i
            ))
        
        return folds


class RollingWindowSplit:
    """
    Rolling window validation.
    
    Both training and test windows slide forward with fixed sizes.
    Best for non-stationary markets.
    """
    
    def __init__(self, n_splits: int = 5, train_size: Optional[int] = None,
                 test_size: Optional[int] = None, gap: int = 0):
        """
        Parameters:
        -----------
        n_splits : int
            Number of CV folds
        train_size : int
            Fixed training window size
        test_size : int
            Fixed test window size
        gap : int
            Number of bars between train and test
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
    
    def split(self, n_samples: int) -> List[CVFold]:
        """Generate rolling window folds."""
        folds = []
        
        # Auto-calculate sizes
        if self.train_size is None:
            self.train_size = n_samples // 2
        if self.test_size is None:
            self.test_size = n_samples // (2 * self.n_splits)
        
        step_size = self.test_size
        
        for i in range(self.n_splits):
            train_start = i * step_size
            train_end = train_start + self.train_size - self.gap
            test_start = train_start + self.train_size
            test_end = test_start + self.test_size
            
            if test_end > n_samples:
                break
            
            folds.append(CVFold(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                fold_id=i
            ))
        
        return folds


class WalkForwardSplit:
    """
    Walk-Forward Optimization splitter.
    
    Standard approach for parameter optimization with immediate OOS testing.
    """
    
    def __init__(self, n_splits: int = 5, train_periods: int = 252,
                 test_periods: int = 63, anchored: bool = False, gap: int = 0):
        """
        Parameters:
        -----------
        n_splits : int
            Number of walk-forward steps
        train_periods : int
            Training window size (e.g., 252 days = 1 year)
        test_periods : int
            Test window size (e.g., 63 days = 1 quarter)
        anchored : bool
            If True, training expands; if False, it rolls
        gap : int
            Number of bars between train and test
        """
        self.n_splits = n_splits
        self.train_periods = train_periods
        self.test_periods = test_periods
        self.anchored = anchored
        self.gap = gap
    
    def split(self, n_samples: int) -> List[CVFold]:
        """Generate walk-forward folds."""
        folds = []
        
        for i in range(self.n_splits):
            if self.anchored:
                train_start = 0
                train_end = self.train_periods + i * self.test_periods - self.gap
            else:
                train_start = i * self.test_periods
                train_end = train_start + self.train_periods - self.gap
            
            test_start = train_end + self.gap
            test_end = test_start + self.test_periods
            
            if test_end > n_samples:
                break
            
            folds.append(CVFold(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                fold_id=i
            ))
        
        return folds


class MonteCarloSplit:
    """
    Monte Carlo time-series split.
    
    Random chronological train/test pairs for robustness testing.
    """
    
    def __init__(self, n_splits: int = 100, test_ratio: float = 0.2,
                 min_train_ratio: float = 0.5, gap: int = 0, seed: int = 42):
        """
        Parameters:
        -----------
        n_splits : int
            Number of random splits to generate
        test_ratio : float
            Maximum test size as ratio of total data
        min_train_ratio : float
            Minimum training size as ratio of total data
        gap : int
            Number of bars between train and test
        seed : int
            Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.test_ratio = test_ratio
        self.min_train_ratio = min_train_ratio
        self.gap = gap
        self.seed = seed
    
    def split(self, n_samples: int) -> List[CVFold]:
        """Generate random chronological splits."""
        np.random.seed(self.seed)
        folds = []
        
        min_train_size = int(n_samples * self.min_train_ratio)
        max_test_size = int(n_samples * self.test_ratio)
        
        for fold_id in range(self.n_splits):
            # Random test size
            test_size = np.random.randint(max_test_size // 2, max_test_size + 1)
            
            # Random test position (ensuring space for training)
            latest_test_start = n_samples - test_size
            earliest_test_start = min_train_size + self.gap
            
            if earliest_test_start >= latest_test_start:
                continue
            
            test_start = np.random.randint(earliest_test_start, latest_test_start + 1)
            test_end = test_start + test_size
            
            # Training: everything before test (with gap)
            train_start = 0
            train_end = test_start - self.gap
            
            # Random training start for additional variation
            if np.random.random() < 0.3:  # 30% chance of not using all history
                max_train_start = train_end - min_train_size
                if max_train_start > 0:
                    train_start = np.random.randint(0, max_train_start)
            
            folds.append(CVFold(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                fold_id=fold_id
            ))
        
        return folds