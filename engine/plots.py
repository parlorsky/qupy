"""
Interactive Plotting Module for Qupy Engine

This module provides standardized, interactive plotting functions for quantitative trading analysis.
All plots use Plotly for interactivity and professional appearance.

Usage:
    from engine.plots import TradingPlots
    plotter = TradingPlots()
    plotter.equity_curve(results)
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from scipy import stats


class TradingPlots:
    """Interactive plotting utilities for trading analysis."""
    
    def __init__(self, theme: str = "plotly_white", width: int = 1000, height: int = 600):
        """
        Initialize plotting configuration.
        
        Args:
            theme: Plotly theme ('plotly_white', 'plotly_dark', 'simple_white')
            width: Default plot width
            height: Default plot height
        """
        self.theme = theme
        self.default_width = width
        self.default_height = height
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
    
    def equity_curve(self, results: Union[Dict, List[Dict]], 
                    title: str = "Strategy Performance",
                    show_drawdown: bool = True,
                    benchmark_data: Optional[pd.Series] = None) -> go.Figure:
        """
        Create interactive equity curve plot with optional drawdown subplot.
        
        Args:
            results: Single result dict or list of results from run_backtest_json
            title: Plot title
            show_drawdown: Whether to show drawdown subplot
            benchmark_data: Optional benchmark price series for comparison
            
        Returns:
            Plotly figure object
        """
        if not isinstance(results, list):
            results = [results]
        
        # Create subplots
        if show_drawdown:
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                shared_xaxes=True,
                subplot_titles=('Equity Curve', 'Drawdown'),
                vertical_spacing=0.05
            )
        else:
            fig = go.Figure()
        
        for i, result in enumerate(results):
            # Extract equity curve data
            if 'equity_curve' in result:
                equity_data = pd.Series(result['equity_curve'])
                
                # Check for timestamps in the data
                if 'equity_timestamps' in result:
                    # Use proper timestamps if available
                    try:
                        timestamps = pd.to_datetime(result['equity_timestamps'])
                        x_data = timestamps
                    except:
                        x_data = list(range(len(equity_data)))
                elif hasattr(equity_data, 'index') and hasattr(equity_data.index, 'to_pydatetime'):
                    x_data = equity_data.index
                else:
                    x_data = list(range(len(equity_data)))
                
                strategy_name = result.get('strategy_name', f'Strategy {i+1}')
                
                # Use distinct colors for each strategy
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
                main_color = colors[i % len(colors)]
                
                # Main equity curve
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=equity_data,
                        mode='lines',
                        name=strategy_name,
                        line=dict(width=2, color=main_color),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Equity: $%{y:,.2f}<br>' +
                                    '<extra></extra>'
                    ),
                    row=1, col=1 if show_drawdown else None
                )
                
                # Drawdown subplot
                if show_drawdown:
                    # Calculate drawdown
                    running_max = equity_data.expanding().max()
                    drawdown = (equity_data - running_max) / running_max * 100
                    
                    # Debug info for troubleshooting (can be removed later)
                    # if strategy_name == 'Buy & Hold':
                    #     final_equity = equity_data.iloc[-1]
                    #     max_equity = running_max.iloc[-1] 
                    #     final_dd = drawdown.iloc[-1]
                    #     print(f"DEBUG Buy & Hold: Final={final_equity:.0f}, Max={max_equity:.0f}, Final DD={final_dd:.1f}%")
                    
                    # For single strategy, don't show DD in legend; for multiple strategies, do show it
                    show_dd_legend = len(results) > 1
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_data,
                            y=drawdown,
                            mode='lines',
                            name=f'{strategy_name} DD',
                            line=dict(width=1, color=main_color),
                            fill='tozeroy',
                            fillcolor='rgba(255,0,0,0.1)',
                            showlegend=show_dd_legend,
                            legendgroup=f"group{i}" if show_dd_legend else None,
                            hovertemplate='<b>Drawdown</b><br>' +
                                        'Date: %{x}<br>' +
                                        'DD: %{y:.2f}%<br>' +
                                        '<extra></extra>'
                        ),
                        row=2, col=1
                    )
        
        # Add benchmark if provided
        if benchmark_data is not None:
            benchmark_norm = (benchmark_data / benchmark_data.iloc[0]) * results[0].get('initial_cash', 100000)
            fig.add_trace(
                go.Scatter(
                    x=benchmark_data.index,
                    y=benchmark_norm,
                    mode='lines',
                    name='Benchmark',
                    line=dict(width=2, dash='dash', color='gray'),
                    opacity=0.7
                ),
                row=1, col=1 if show_drawdown else None
            )
        
        # Layout updates
        fig.update_layout(
            title=title,
            template=self.theme,
            width=self.default_width,
            height=self.default_height,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Time", row=2 if show_drawdown else 1, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        if show_drawdown:
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        return fig
    
    def strategy_comparison(self, results: List[Dict], 
                          metric: str = "sharpe_ratio",
                          chart_type: str = "bar") -> go.Figure:
        """
        Create strategy comparison chart.
        
        Args:
            results: List of strategy results
            metric: Metric to compare ('sharpe_ratio', 'total_return', 'max_drawdown', etc.)
            chart_type: 'bar', 'scatter', or 'radar'
            
        Returns:
            Plotly figure object
        """
        # Extract data
        strategies = []
        values = []
        
        for result in results:
            strategies.append(result.get('strategy_name', 'Unknown'))
            
            if metric in result:
                values.append(result[metric])
            elif 'risk_metrics' in result and metric in result['risk_metrics']:
                values.append(result['risk_metrics'][metric])
            elif 'performance_metrics' in result and metric in result['performance_metrics']:
                values.append(result['performance_metrics'][metric])
            else:
                values.append(0)
        
        if chart_type == "bar":
            fig = go.Figure(data=[
                go.Bar(
                    x=strategies,
                    y=values,
                    text=[f'{v:.3f}' for v in values],
                    textposition='auto',
                    marker_color=self.colors['primary']
                )
            ])
            fig.update_layout(
                title=f"Strategy Comparison: {metric.replace('_', ' ').title()}",
                xaxis_title="Strategy",
                yaxis_title=metric.replace('_', ' ').title()
            )
        
        elif chart_type == "scatter":
            # For scatter, use total return vs sharpe ratio
            returns = []
            sharpes = []
            
            for result in results:
                returns.append(result.get('total_return', 0))
                if 'risk_metrics' in result:
                    sharpes.append(result['risk_metrics'].get('sharpe_ratio', 0))
                else:
                    sharpes.append(0)
            
            fig = go.Figure(data=[
                go.Scatter(
                    x=returns,
                    y=sharpes,
                    mode='markers+text',
                    text=strategies,
                    textposition="middle right",
                    marker=dict(size=12, color=self.colors['primary']),
                    hovertemplate='<b>%{text}</b><br>' +
                                'Return: %{x:.2%}<br>' +
                                'Sharpe: %{y:.3f}<br>' +
                                '<extra></extra>'
                )
            ])
            fig.update_layout(
                title="Strategy Risk-Return Profile",
                xaxis_title="Total Return (%)",
                yaxis_title="Sharpe Ratio"
            )
        
        elif chart_type == "radar":
            # Multi-metric radar chart
            metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
            
            fig = go.Figure()
            
            for i, result in enumerate(results):
                metric_values = []
                for m in metrics:
                    if m in result:
                        val = result[m]
                    elif 'risk_metrics' in result and m in result['risk_metrics']:
                        val = result['risk_metrics'][m]
                    else:
                        val = 0
                    
                    # Normalize metrics (invert drawdown)
                    if m == 'max_drawdown':
                        val = -val
                    metric_values.append(val)
                
                fig.add_trace(go.Scatterpolar(
                    r=metric_values,
                    theta=metrics,
                    fill='toself',
                    name=result.get('strategy_name', f'Strategy {i+1}'),
                    opacity=0.6
                ))
        
        fig.update_layout(
            template=self.theme,
            width=self.default_width,
            height=self.default_height
        )
        
        return fig
    
    def returns_distribution(self, returns_data: Union[pd.Series, Dict[str, pd.Series]],
                           add_var_lines: bool = True,
                           confidence_levels: List[float] = [0.95, 0.99]) -> go.Figure:
        """
        Create interactive returns distribution plot with VaR lines.
        
        Args:
            returns_data: Single returns series or dict of multiple series
            add_var_lines: Whether to add VaR lines
            confidence_levels: VaR confidence levels to show
            
        Returns:
            Plotly figure object
        """
        if isinstance(returns_data, pd.Series):
            returns_data = {"Strategy": returns_data}
        
        fig = go.Figure()
        
        for name, returns in returns_data.items():
            # Convert to percentage
            returns_pct = returns * 100
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=returns_pct,
                name=name,
                nbinsx=50,
                opacity=0.7,
                histnorm='probability density'
            ))
            
            # Add VaR lines with better positioning
            if add_var_lines:
                for i, conf in enumerate(confidence_levels):
                    var_val = returns_pct.quantile(1 - conf)
                    color = "red" if conf == 0.99 else "orange"
                    
                    fig.add_vline(
                        x=var_val,
                        line_dash="dash",
                        line_color=color,
                        line_width=2
                    )
                    
                    # Add annotations with better positioning
                    fig.add_annotation(
                        x=var_val,
                        y=0.8 - (i * 0.15),  # Stagger annotations vertically
                        text=f"{conf:.0%} VaR<br>{var_val:.2f}%",
                        yref="paper",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=color,
                        ax=20,
                        ay=-30,
                        bordercolor=color,
                        borderwidth=1,
                        bgcolor="white",
                        opacity=0.8,
                        font=dict(size=10)
                    )
        
        fig.update_layout(
            title={
                'text': "Returns Distribution Analysis 📊",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title="Daily Returns (%)",
            yaxis_title="Probability Density",
            template=self.theme,
            width=self.default_width,
            height=self.default_height,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right", 
                x=1
            ),
            margin=dict(t=80, b=60, l=60, r=60),
            hovermode='x unified'
        )
        
        return fig
    
    def optimization_heatmap(self, optimization_results: pd.DataFrame,
                           param1: str, param2: str, 
                           metric: str = "sharpe_ratio") -> go.Figure:
        """
        Create parameter optimization heatmap.
        
        Args:
            optimization_results: DataFrame with optimization results
            param1: First parameter name (x-axis)
            param2: Second parameter name (y-axis) 
            metric: Optimization metric to visualize
            
        Returns:
            Plotly figure object
        """
        # Pivot the data for heatmap
        pivot_data = optimization_results.pivot(
            index=param2, columns=param1, values=metric
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlGn',
            hoverongaps=False,
            hovertemplate=f'<b>{param1}</b>: %{{x}}<br>' +
                         f'<b>{param2}</b>: %{{y}}<br>' +
                         f'<b>{metric}</b>: %{{z:.3f}}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Parameter Optimization: {metric.replace('_', ' ').title()}",
            xaxis_title=param1.replace('_', ' ').title(),
            yaxis_title=param2.replace('_', ' ').title(),
            template=self.theme,
            width=self.default_width,
            height=self.default_height
        )
        
        return fig
    
    def walk_forward_analysis(self, wf_results: pd.DataFrame) -> go.Figure:
        """
        Create walk-forward analysis visualization.
        
        Args:
            wf_results: DataFrame with walk-forward results
            
        Returns:
            Plotly figure object with multiple subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Parameter Stability',
                'IS vs OOS Performance', 
                'Performance Over Time',
                'Return Distribution'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Parameter stability (assuming RSI strategy)
        if 'best_rsi_period' in wf_results.columns:
            fig.add_trace(
                go.Scatter(
                    x=wf_results['fold_id'],
                    y=wf_results['best_rsi_period'],
                    mode='markers+lines',
                    name='RSI Period',
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
        
        # IS vs OOS Sharpe
        if 'in_sample_sharpe' in wf_results.columns and 'out_sample_sharpe' in wf_results.columns:
            fig.add_trace(
                go.Scatter(
                    x=wf_results['in_sample_sharpe'],
                    y=wf_results['out_sample_sharpe'],
                    mode='markers',
                    name='IS vs OOS',
                    marker=dict(size=10),
                    hovertemplate='<b>Fold %{customdata}</b><br>' +
                                'IS Sharpe: %{x:.3f}<br>' +
                                'OOS Sharpe: %{y:.3f}<br>' +
                                '<extra></extra>',
                    customdata=wf_results['fold_id']
                ),
                row=1, col=2
            )
            
            # Add perfect correlation line
            min_sharpe = min(wf_results['in_sample_sharpe'].min(), wf_results['out_sample_sharpe'].min())
            max_sharpe = max(wf_results['in_sample_sharpe'].max(), wf_results['out_sample_sharpe'].max())
            
            fig.add_trace(
                go.Scatter(
                    x=[min_sharpe, max_sharpe],
                    y=[min_sharpe, max_sharpe],
                    mode='lines',
                    name='Perfect Correlation',
                    line=dict(dash='dash', color='red'),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Performance over time
        if 'out_sample_sharpe' in wf_results.columns:
            fig.add_trace(
                go.Scatter(
                    x=wf_results['fold_id'],
                    y=wf_results['out_sample_sharpe'],
                    mode='markers+lines',
                    name='OOS Sharpe',
                    marker=dict(size=8, color='green')
                ),
                row=2, col=1
            )
        
        # Return distribution
        if 'out_sample_return_pct' in wf_results.columns:
            fig.add_trace(
                go.Histogram(
                    x=wf_results['out_sample_return_pct'],
                    name='OOS Returns',
                    nbinsx=10,
                    opacity=0.7
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Walk-Forward Cross-Validation Analysis",
            template=self.theme,
            width=self.default_width + 200,
            height=self.default_height + 100,
            showlegend=False
        )
        
        return fig
    
    def monte_carlo_analysis(self, mc_results: pd.DataFrame) -> go.Figure:
        """
        Create Monte Carlo simulation analysis plots.
        
        Args:
            mc_results: DataFrame with Monte Carlo results
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Sharpe Ratio Distribution',
                'Return Distribution',
                'Performance vs Test Size', 
                'Cumulative Distribution'
            )
        )
        
        # Sharpe ratio distribution
        if 'sharpe_ratio' in mc_results.columns:
            fig.add_trace(
                go.Histogram(
                    x=mc_results['sharpe_ratio'],
                    name='Sharpe Distribution',
                    nbinsx=20,
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Add mean and median lines
            mean_sharpe = mc_results['sharpe_ratio'].mean()
            median_sharpe = mc_results['sharpe_ratio'].median()
            
            fig.add_vline(x=mean_sharpe, line_dash="dash", line_color="green", 
                         annotation_text=f"Mean: {mean_sharpe:.3f}", row=1, col=1)
            fig.add_vline(x=median_sharpe, line_dash="dash", line_color="orange",
                         annotation_text=f"Median: {median_sharpe:.3f}", row=1, col=1)
        
        # Return distribution
        if 'total_return_pct' in mc_results.columns:
            fig.add_trace(
                go.Histogram(
                    x=mc_results['total_return_pct'],
                    name='Return Distribution',
                    nbinsx=20,
                    opacity=0.7,
                    marker_color='lightgreen'
                ),
                row=1, col=2
            )
        
        # Performance vs test size
        if 'test_size' in mc_results.columns and 'sharpe_ratio' in mc_results.columns:
            fig.add_trace(
                go.Scatter(
                    x=mc_results['test_size'],
                    y=mc_results['sharpe_ratio'],
                    mode='markers',
                    name='Size vs Performance',
                    marker=dict(size=6, opacity=0.6)
                ),
                row=2, col=1
            )
        
        # Cumulative distribution
        if 'sharpe_ratio' in mc_results.columns:
            sorted_sharpes = np.sort(mc_results['sharpe_ratio'])
            cumprob = np.arange(1, len(sorted_sharpes) + 1) / len(sorted_sharpes)
            
            fig.add_trace(
                go.Scatter(
                    x=sorted_sharpes,
                    y=cumprob,
                    mode='lines',
                    name='CDF',
                    line=dict(width=3, color='navy')
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Monte Carlo Cross-Validation Analysis",
            template=self.theme,
            width=self.default_width + 200,
            height=self.default_height + 100,
            showlegend=False
        )
        
        return fig
    
    def kelly_criterion_analysis(self, kelly_results: Dict[float, Dict]) -> go.Figure:
        """
        Create Kelly Criterion analysis visualization.
        
        Args:
            kelly_results: Dictionary of Kelly multiplier results
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Wealth Growth', 'Growth Rate', 'Bankruptcy Risk')
        )
        
        kelly_mults = list(kelly_results.keys())
        median_wealths = [kelly_results[k]['median_wealth'] for k in kelly_mults]
        growth_rates = [kelly_results[k]['growth_rate'] for k in kelly_mults]
        bankruptcy_rates = [kelly_results[k]['bankruptcy_rate'] * 100 for k in kelly_mults]
        
        # Wealth growth
        fig.add_trace(
            go.Scatter(
                x=kelly_mults,
                y=median_wealths,
                mode='markers+lines',
                name='Median Wealth',
                marker=dict(size=8),
                line=dict(width=2)
            ),
            row=1, col=1
        )
        
        # Growth rate
        fig.add_trace(
            go.Scatter(
                x=kelly_mults,
                y=growth_rates,
                mode='markers+lines',
                name='Growth Rate',
                marker=dict(size=8, color='green'),
                line=dict(width=2, color='green')
            ),
            row=1, col=2
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=2)
        
        # Bankruptcy rate
        fig.add_trace(
            go.Scatter(
                x=kelly_mults,
                y=bankruptcy_rates,
                mode='markers+lines',
                name='Bankruptcy Rate',
                marker=dict(size=8, color='red'),
                line=dict(width=2, color='red')
            ),
            row=1, col=3
        )
        
        # Add full Kelly line to all subplots
        for col in [1, 2, 3]:
            fig.add_vline(x=1.0, line_dash="dash", line_color="red", 
                         opacity=0.5, annotation_text="Full Kelly", row=1, col=col)
        
        fig.update_layout(
            title="Kelly Criterion Analysis",
            template=self.theme,
            width=self.default_width + 400,
            height=self.default_height - 100,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Kelly Multiplier")
        fig.update_yaxes(title_text="Median Wealth", row=1, col=1)
        fig.update_yaxes(title_text="Growth Rate", row=1, col=2) 
        fig.update_yaxes(title_text="Bankruptcy Rate (%)", row=1, col=3)
        
        return fig

    def price_and_signals(self, price_data: pd.Series, 
                         signals: Optional[pd.Series] = None,
                         trades: Optional[List] = None,
                         indicators: Optional[Dict[str, pd.Series]] = None) -> go.Figure:
        """
        Create price chart with trading signals and indicators.
        
        Args:
            price_data: Price time series
            signals: Trading signals (1=buy, -1=sell, 0=hold)
            trades: List of trade objects with entry/exit points
            indicators: Dictionary of technical indicators to overlay
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            subplot_titles=('Price & Signals', 'Indicators'),
            vertical_spacing=0.05
        )
        
        # Main price line
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data.values,
                mode='lines',
                name='Price',
                line=dict(width=1),
                hovertemplate='<b>Price</b><br>' +
                            'Date: %{x}<br>' +
                            'Price: $%{y:.4f}<br>' +
                            '<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add trading signals
        if signals is not None:
            buy_signals = signals[signals == 1]
            sell_signals = signals[signals == -1]
            
            if len(buy_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=price_data.loc[buy_signals.index],
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(symbol='triangle-up', size=12, color='green'),
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            if len(sell_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=price_data.loc[sell_signals.index],
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(symbol='triangle-down', size=12, color='red'),
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        # Add indicators to second subplot
        if indicators:
            for name, indicator_data in indicators.items():
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data.index,
                        y=indicator_data.values,
                        mode='lines',
                        name=name,
                        line=dict(width=1),
                        opacity=0.8
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            title="Price Action & Trading Signals",
            template=self.theme,
            width=self.default_width,
            height=self.default_height,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Indicator Value", row=2, col=1)
        
        return fig


# Convenience functions for quick plotting
def quick_equity_curve(results: Union[Dict, List[Dict]], **kwargs) -> go.Figure:
    """Quick equity curve plot."""
    plotter = TradingPlots()
    return plotter.equity_curve(results, **kwargs)

def quick_strategy_comparison(results: List[Dict], **kwargs) -> go.Figure:
    """Quick strategy comparison plot.""" 
    plotter = TradingPlots()
    return plotter.strategy_comparison(results, **kwargs)

def quick_returns_distribution(returns_data: Union[pd.Series, Dict[str, pd.Series]], **kwargs) -> go.Figure:
    """Quick returns distribution plot."""
    plotter = TradingPlots()
    return plotter.returns_distribution(returns_data, **kwargs)