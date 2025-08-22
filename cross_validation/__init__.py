"""
Cross-validation module for time-series backtesting.

Simplified implementation with core splitters for robust strategy evaluation.
"""

from .splitters import *
from .purging import *

__all__ = [
    # Core splitters
    'TimeSeriesSplit',
    'ExpandingWindowSplit',
    'RollingWindowSplit',
    'WalkForwardSplit',
    'MonteCarloSplit',
    
    # Purging utilities
    'PurgeEmbargo',
    'apply_purge_embargo',
]