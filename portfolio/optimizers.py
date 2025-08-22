"""
Portfolio optimization algorithms.

Implementations of various portfolio construction methods.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List, Tuple
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform


ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]


# ============================================================================
# Simple Weighting Schemes
# ============================================================================

def equal_weight(n_assets: int) -> np.ndarray:
    """
    Equal weight portfolio.
    
    Parameters:
    -----------
    n_assets : int
        Number of assets
    
    Returns:
    --------
    np.ndarray
        Equal weights summing to 1
    """
    return np.ones(n_assets) / n_assets


def cap_weight(market_caps: ArrayLike) -> np.ndarray:
    """
    Market capitalization weighted portfolio.
    
    Parameters:
    -----------
    market_caps : ArrayLike
        Market capitalizations
    
    Returns:
    --------
    np.ndarray
        Cap-weighted portfolio
    """
    caps = np.asarray(market_caps)
    return caps / caps.sum()


def inverse_vol_weights(vol_estimates: ArrayLike, cap_to_max: Optional[float] = None) -> np.ndarray:
    """
    Inverse volatility weighting.
    
    Parameters:
    -----------
    vol_estimates : ArrayLike
        Volatility estimates for each asset
    cap_to_max : float, optional
        Maximum weight per asset
    
    Returns:
    --------
    np.ndarray
        Inverse vol weights
    """
    vols = np.asarray(vol_estimates)
    
    # Inverse volatility
    inv_vols = 1.0 / vols
    weights = inv_vols / inv_vols.sum()
    
    # Apply cap if specified
    if cap_to_max is not None:
        weights = np.minimum(weights, cap_to_max)
        weights = weights / weights.sum()
    
    return weights


# ============================================================================
# Mean-Variance Optimization
# ============================================================================

def min_variance_weights(cov: np.ndarray, bounds: Optional[List[Tuple]] = None) -> np.ndarray:
    """
    Minimum variance portfolio.
    
    Parameters:
    -----------
    cov : np.ndarray
        Covariance matrix
    bounds : List[Tuple], optional
        Bounds for each weight [(min, max), ...]
    
    Returns:
    --------
    np.ndarray
        Minimum variance weights
    """
    n = cov.shape[0]
    
    # Default bounds
    if bounds is None:
        bounds = [(0, 1) for _ in range(n)]
    
    # Objective: minimize portfolio variance
    def objective(w):
        return w @ cov @ w
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # Initial guess: equal weights
    w0 = np.ones(n) / n
    
    # Optimize
    result = minimize(objective, w0, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return result.x


def max_sharpe_weights(mu: np.ndarray, cov: np.ndarray, rf: float = 0, 
                      bounds: Optional[List[Tuple]] = None) -> np.ndarray:
    """
    Maximum Sharpe ratio portfolio.
    
    Parameters:
    -----------
    mu : np.ndarray
        Expected returns
    cov : np.ndarray
        Covariance matrix
    rf : float
        Risk-free rate
    bounds : List[Tuple], optional
        Bounds for each weight
    
    Returns:
    --------
    np.ndarray
        Maximum Sharpe weights
    """
    n = len(mu)
    
    # Default bounds
    if bounds is None:
        bounds = [(0, 1) for _ in range(n)]
    
    # Objective: maximize Sharpe ratio (minimize negative Sharpe)
    def objective(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -(ret - rf) / vol if vol > 0 else -np.inf
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # Initial guess: equal weights
    w0 = np.ones(n) / n
    
    # Optimize
    result = minimize(objective, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    return result.x


def mean_variance_frontier(mu: np.ndarray, cov: np.ndarray, 
                          n_portfolios: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate efficient frontier.
    
    Returns:
    --------
    returns : np.ndarray
        Expected returns of frontier portfolios
    risks : np.ndarray
        Volatilities of frontier portfolios
    weights : np.ndarray
        Weights of frontier portfolios (n_portfolios x n_assets)
    """
    n = len(mu)
    
    # Get min variance portfolio
    min_var_w = min_variance_weights(cov)
    min_var_ret = min_var_w @ mu
    
    # Get max return portfolio (100% in highest return asset)
    max_ret = np.max(mu)
    
    # Target returns along the frontier
    target_returns = np.linspace(min_var_ret, max_ret, n_portfolios)
    
    frontier_weights = []
    frontier_risks = []
    
    for target_ret in target_returns:
        # Minimize variance for given return
        def objective(w):
            return w @ cov @ w
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: w @ mu - target_ret}
        ]
        
        bounds = [(0, 1) for _ in range(n)]
        w0 = np.ones(n) / n
        
        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            frontier_weights.append(result.x)
            frontier_risks.append(np.sqrt(result.fun))
    
    return target_returns, np.array(frontier_risks), np.array(frontier_weights)


# ============================================================================
# Risk Parity
# ============================================================================

def risk_parity_weights(cov: np.ndarray, target_risk: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Risk parity portfolio (equal risk contribution).
    
    Parameters:
    -----------
    cov : np.ndarray
        Covariance matrix
    target_risk : np.ndarray, optional
        Target risk contributions (default: equal)
    
    Returns:
    --------
    np.ndarray
        Risk parity weights
    """
    n = cov.shape[0]
    
    # Default: equal risk contribution
    if target_risk is None:
        target_risk = np.ones(n) / n
    
    # Objective: minimize difference between actual and target risk contributions
    def objective(w):
        portfolio_vol = np.sqrt(w @ cov @ w)
        marginal_contrib = cov @ w / portfolio_vol
        contrib = w * marginal_contrib
        return np.sum((contrib - target_risk * portfolio_vol) ** 2)
    
    # Constraints: weights sum to 1, all positive
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0.001, 1) for _ in range(n)]  # Small lower bound to avoid division by zero
    
    # Initial guess: inverse vol weights
    sigma = np.sqrt(np.diag(cov))
    w0 = (1.0 / sigma) / np.sum(1.0 / sigma)
    
    # Optimize
    result = minimize(objective, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    return result.x


def equal_risk_contribution_weights(cov: np.ndarray, groups: Optional[List[int]] = None) -> np.ndarray:
    """
    Equal risk contribution with optional grouping.
    
    Parameters:
    -----------
    cov : np.ndarray
        Covariance matrix
    groups : List[int], optional
        Group assignment for each asset
    
    Returns:
    --------
    np.ndarray
        ERC weights
    """
    if groups is None:
        return risk_parity_weights(cov)
    
    # Group-wise risk parity
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    n_assets = cov.shape[0]
    
    # First, allocate equally among groups
    group_weights = np.ones(n_groups) / n_groups
    
    # Then, within each group, do risk parity
    final_weights = np.zeros(n_assets)
    
    for i, g in enumerate(unique_groups):
        group_mask = np.array(groups) == g
        group_indices = np.where(group_mask)[0]
        
        if len(group_indices) == 1:
            final_weights[group_indices[0]] = group_weights[i]
        else:
            # Extract group covariance
            group_cov = cov[np.ix_(group_indices, group_indices)]
            
            # Risk parity within group
            within_group_weights = risk_parity_weights(group_cov)
            
            # Scale by group allocation
            final_weights[group_indices] = within_group_weights * group_weights[i]
    
    return final_weights


# ============================================================================
# Hierarchical Risk Parity (HRP)
# ============================================================================

def _get_cluster_var(cov: np.ndarray, cluster_items: List[int]) -> float:
    """Calculate cluster variance (inverse variance weighting within cluster)."""
    cluster_cov = cov[np.ix_(cluster_items, cluster_items)]
    inv_diag = 1 / np.diag(cluster_cov)
    w = inv_diag / inv_diag.sum()
    return w @ cluster_cov @ w


def _get_quasi_diag(link: np.ndarray) -> List[int]:
    """Get quasi-diagonal ordering from linkage."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    
    return sort_ix.tolist()


def hrp_weights(returns: pd.DataFrame) -> np.ndarray:
    """
    Hierarchical Risk Parity portfolio.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Returns matrix (time x assets)
    
    Returns:
    --------
    np.ndarray
        HRP weights
    """
    # Calculate correlation and covariance
    cov = returns.cov().values
    corr = returns.corr().values
    
    # Calculate distance matrix
    dist = np.sqrt(0.5 * (1 - corr))
    
    # Hierarchical clustering
    link = linkage(squareform(dist), 'single')
    
    # Get quasi-diagonal ordering
    sort_ix = _get_quasi_diag(link)
    
    # Recursive bisection
    n = len(sort_ix)
    w = pd.Series(1, index=sort_ix)
    cluster_items = [sort_ix]
    
    while len(cluster_items) > 0:
        # Pop a cluster
        cluster = cluster_items.pop()
        
        if len(cluster) > 1:
            # Get cluster variance
            cluster_var = _get_cluster_var(cov, cluster)
            
            # Split cluster
            n_split = len(cluster) // 2
            left = cluster[:n_split]
            right = cluster[n_split:]
            
            # Calculate variances
            left_var = _get_cluster_var(cov, left)
            right_var = _get_cluster_var(cov, right)
            
            # Allocate between sub-clusters
            alpha = 1 - left_var / (left_var + right_var)
            
            w[left] *= alpha
            w[right] *= 1 - alpha
            
            # Add sub-clusters to list
            if len(left) > 1:
                cluster_items.append(left)
            if len(right) > 1:
                cluster_items.append(right)
    
    # Sort weights back to original order
    return w.sort_index().values


# ============================================================================
# Dynamic and Adaptive Strategies
# ============================================================================

def volatility_targeted_weights(returns_panel: pd.DataFrame, target_vol: float,
                               lookback: int = 60, floor_ceiling: Optional[Tuple] = None) -> pd.DataFrame:
    """
    Volatility targeting overlay on existing weights.
    
    Parameters:
    -----------
    returns_panel : pd.DataFrame
        Historical returns (time x assets)
    target_vol : float
        Target portfolio volatility
    lookback : int
        Lookback period for vol estimation
    floor_ceiling : Tuple, optional
        (min_leverage, max_leverage)
    
    Returns:
    --------
    pd.DataFrame
        Time series of volatility-targeted weights
    """
    # Calculate rolling portfolio volatility
    rolling_cov = returns_panel.rolling(lookback).cov()
    
    # Start with equal weights
    n_assets = returns_panel.shape[1]
    base_weights = np.ones(n_assets) / n_assets
    
    weights_list = []
    dates = returns_panel.index[lookback:]
    
    for date in dates:
        # Get covariance for this date
        try:
            cov = rolling_cov.loc[date].values
            
            # Portfolio vol with base weights
            port_vol = np.sqrt(base_weights @ cov @ base_weights)
            
            # Scale factor to hit target vol
            scale = target_vol / port_vol if port_vol > 0 else 1.0
            
            # Apply floor and ceiling
            if floor_ceiling:
                scale = np.clip(scale, floor_ceiling[0], floor_ceiling[1])
            
            # Scaled weights
            weights = base_weights * scale
            
        except:
            weights = base_weights
        
        weights_list.append(weights)
    
    return pd.DataFrame(weights_list, index=dates, columns=returns_panel.columns)


def momentum_tilt_weights(momentum_scores: ArrayLike, long_only: bool = True,
                         neutralize_mean: bool = True) -> np.ndarray:
    """
    Convert momentum scores to portfolio weights.
    
    Parameters:
    -----------
    momentum_scores : ArrayLike
        Momentum scores for each asset
    long_only : bool
        If True, convert to long-only weights
    neutralize_mean : bool
        If True, demean scores first
    
    Returns:
    --------
    np.ndarray
        Momentum-tilted weights
    """
    scores = np.asarray(momentum_scores)
    
    # Neutralize mean
    if neutralize_mean:
        scores = scores - scores.mean()
    
    if long_only:
        # Convert to positive weights
        scores = scores - scores.min()
        weights = scores / scores.sum() if scores.sum() > 0 else np.ones_like(scores) / len(scores)
    else:
        # Long-short weights
        weights = scores / np.abs(scores).sum() if np.abs(scores).sum() > 0 else np.zeros_like(scores)
    
    return weights


# ============================================================================
# Constraints and Adjustments
# ============================================================================

def normalize_weights(weights: ArrayLike, method: str = "sum1", target: float = 1.0) -> np.ndarray:
    """
    Normalize portfolio weights.
    
    Parameters:
    -----------
    weights : ArrayLike
        Raw weights
    method : str
        "sum1": weights sum to target
        "gross": gross exposure equals target
        "leverage": leverage equals target
    target : float
        Target value
    
    Returns:
    --------
    np.ndarray
        Normalized weights
    """
    w = np.asarray(weights)
    
    if method == "sum1":
        return w * target / w.sum() if w.sum() != 0 else w
    elif method == "gross":
        return w * target / np.abs(w).sum() if np.abs(w).sum() != 0 else w
    elif method == "leverage":
        current_leverage = np.abs(w).sum()
        return w * target / current_leverage if current_leverage != 0 else w
    else:
        raise ValueError(f"Unknown method: {method}")


def apply_bounds(weights: ArrayLike, lower: float = 0, upper: float = 1) -> np.ndarray:
    """Apply weight bounds."""
    return np.clip(weights, lower, upper)


def leverage_cap(weights: ArrayLike, max_leverage: float) -> np.ndarray:
    """Cap total leverage."""
    w = np.asarray(weights)
    current_leverage = np.abs(w).sum()
    
    if current_leverage > max_leverage:
        return w * max_leverage / current_leverage
    return w


def turnover_limit(new_weights: ArrayLike, old_weights: ArrayLike, 
                   max_turnover: float) -> np.ndarray:
    """
    Limit portfolio turnover.
    
    Parameters:
    -----------
    new_weights : ArrayLike
        Target weights
    old_weights : ArrayLike
        Current weights
    max_turnover : float
        Maximum allowed turnover
    
    Returns:
    --------
    np.ndarray
        Turnover-limited weights
    """
    new_w = np.asarray(new_weights)
    old_w = np.asarray(old_weights)
    
    # Calculate turnover
    turnover = np.abs(new_w - old_w).sum()
    
    if turnover > max_turnover:
        # Scale down the change
        scale = max_turnover / turnover
        return old_w + scale * (new_w - old_w)
    
    return new_w


def sector_neutralize(weights: ArrayLike, sector_map: Dict[int, int], 
                      net: float = 0.0) -> np.ndarray:
    """
    Make portfolio sector-neutral.
    
    Parameters:
    -----------
    weights : ArrayLike
        Raw weights
    sector_map : Dict[int, int]
        Maps asset index to sector
    net : float
        Target net exposure per sector
    
    Returns:
    --------
    np.ndarray
        Sector-neutralized weights
    """
    w = np.asarray(weights).copy()
    
    # Get unique sectors
    sectors = np.unique(list(sector_map.values()))
    
    for sector in sectors:
        # Get assets in this sector
        sector_assets = [i for i, s in sector_map.items() if s == sector]
        
        if len(sector_assets) > 0:
            # Current sector exposure
            sector_exposure = w[sector_assets].sum()
            
            # Adjust to target
            adjustment = (net - sector_exposure) / len(sector_assets)
            w[sector_assets] += adjustment
    
    return w