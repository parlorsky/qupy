"""
Purging and embargo utilities for preventing data leakage.

Handles overlapping labels and ensures proper temporal separation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class PurgeEmbargo:
    """
    Apply purging and embargo to prevent data leakage.
    
    Purging: Remove training samples whose labels overlap with test labels.
    Embargo: Add buffer period after test to prevent leakage.
    """
    
    def __init__(self, label_horizon: int = 1, purge_window: Optional[int] = None,
                 embargo_window: Optional[int] = None):
        """
        Parameters:
        -----------
        label_horizon : int
            Number of bars used to compute labels (e.g., 20 for 20-bar returns)
        purge_window : int
            Number of bars to purge before test (default: label_horizon)
        embargo_window : int
            Number of bars to embargo after test (default: label_horizon)
        """
        self.label_horizon = label_horizon
        self.purge_window = purge_window or label_horizon
        self.embargo_window = embargo_window or label_horizon
    
    def apply(self, train_indices: np.ndarray, test_indices: np.ndarray,
              n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply purging and embargo to training indices.
        
        Parameters:
        -----------
        train_indices : np.ndarray
            Original training indices
        test_indices : np.ndarray
            Test indices
        n_samples : int
            Total number of samples
        
        Returns:
        --------
        clean_train : np.ndarray
            Training indices after purging/embargo
        purged : np.ndarray
            Indices that were purged
        embargoed : np.ndarray
            Indices that were embargoed
        """
        if len(test_indices) == 0:
            return train_indices, np.array([]), np.array([])
        
        test_start = test_indices[0]
        test_end = test_indices[-1]
        
        # Purge: remove training samples before test that could leak
        purge_start = max(0, test_start - self.purge_window)
        purge_end = min(n_samples - 1, test_start - 1)
        
        # Embargo: remove training samples after test
        embargo_start = test_end + 1
        embargo_end = min(n_samples - 1, test_end + self.embargo_window)
        
        # Create masks
        purge_mask = (train_indices >= purge_start) & (train_indices <= purge_end)
        embargo_mask = (train_indices >= embargo_start) & (train_indices <= embargo_end)
        
        # Apply masks
        purged_indices = train_indices[purge_mask]
        embargoed_indices = train_indices[embargo_mask]
        
        # Clean training set
        clean_mask = ~(purge_mask | embargo_mask)
        clean_train = train_indices[clean_mask]
        
        return clean_train, purged_indices, embargoed_indices


def apply_purge_embargo(data: pd.DataFrame, train_start: int, train_end: int,
                       test_start: int, test_end: int,
                       label_horizon: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply purging and embargo to a DataFrame.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Full dataset
    train_start, train_end : int
        Training period boundaries
    test_start, test_end : int
        Test period boundaries
    label_horizon : int
        Number of bars for label computation
    
    Returns:
    --------
    train_data : pd.DataFrame
        Clean training data
    test_data : pd.DataFrame
        Test data
    """
    # Create purge/embargo handler
    purger = PurgeEmbargo(label_horizon=label_horizon)
    
    # Get indices
    train_indices = np.arange(train_start, train_end)
    test_indices = np.arange(test_start, test_end)
    
    # Apply purging
    clean_train_indices, _, _ = purger.apply(
        train_indices, test_indices, len(data)
    )
    
    # Extract data
    train_data = data.iloc[clean_train_indices]
    test_data = data.iloc[test_indices]
    
    return train_data, test_data


def compute_label_windows(timestamps: pd.DatetimeIndex, label_horizon: int,
                         label_type: str = 'fixed') -> pd.DataFrame:
    """
    Compute label windows for each sample.
    
    Parameters:
    -----------
    timestamps : pd.DatetimeIndex
        Sample timestamps
    label_horizon : int
        Number of bars for label
    label_type : str
        'fixed': fixed horizon
        'triple_barrier': variable endpoint
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: sample_idx, label_start, label_end
    """
    n = len(timestamps)
    windows = []
    
    for i in range(n):
        if label_type == 'fixed':
            label_start = i
            label_end = min(i + label_horizon, n - 1)
        else:
            # For triple barrier, endpoint varies
            # This is simplified - actual implementation would check barriers
            label_start = i
            label_end = min(i + label_horizon * 2, n - 1)  # Max horizon
        
        windows.append({
            'sample_idx': i,
            'label_start': label_start,
            'label_end': label_end,
            'timestamp': timestamps[i]
        })
    
    return pd.DataFrame(windows)


def check_overlap(train_windows: pd.DataFrame, test_windows: pd.DataFrame) -> pd.DataFrame:
    """
    Check for overlapping label windows between train and test.
    
    Parameters:
    -----------
    train_windows : pd.DataFrame
        Training label windows
    test_windows : pd.DataFrame
        Test label windows
    
    Returns:
    --------
    pd.DataFrame
        DataFrame of overlapping samples
    """
    overlaps = []
    
    for _, test_row in test_windows.iterrows():
        test_start = test_row['label_start']
        test_end = test_row['label_end']
        
        # Check each training window
        for _, train_row in train_windows.iterrows():
            train_start = train_row['label_start']
            train_end = train_row['label_end']
            
            # Check for overlap
            if not (train_end < test_start or train_start > test_end):
                overlaps.append({
                    'train_idx': train_row['sample_idx'],
                    'test_idx': test_row['sample_idx'],
                    'train_window': (train_start, train_end),
                    'test_window': (test_start, test_end)
                })
    
    return pd.DataFrame(overlaps)


def effective_sample_size(n_samples: int, label_horizon: int,
                         train_ratio: float = 0.8) -> dict:
    """
    Calculate effective sample sizes after accounting for overlaps.
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples
    label_horizon : int
        Label computation horizon
    train_ratio : float
        Training set ratio
    
    Returns:
    --------
    dict
        Statistics on effective samples
    """
    train_size = int(n_samples * train_ratio)
    test_size = n_samples - train_size
    
    # Approximate number of non-overlapping labels
    effective_train = train_size // label_horizon
    effective_test = test_size // label_horizon
    
    # Samples lost to purging (approximate)
    purged_samples = min(label_horizon, train_size)
    
    return {
        'total_samples': n_samples,
        'raw_train_size': train_size,
        'raw_test_size': test_size,
        'effective_train_samples': effective_train,
        'effective_test_samples': effective_test,
        'purged_samples': purged_samples,
        'label_horizon': label_horizon,
        'overlap_ratio': label_horizon / n_samples
    }