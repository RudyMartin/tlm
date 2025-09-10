"""
Stock trading algorithms module for TLM.

Provides comprehensive algorithmic trading strategies including technical analysis,
momentum, mean reversion, statistical arbitrage, and machine learning approaches.
Essential for quantitative finance and automated trading systems.
"""

from .indicators import (
    # Moving averages
    simple_moving_average, exponential_moving_average, weighted_moving_average,
    hull_moving_average, kaufman_adaptive_moving_average,
    # Oscillators
    rsi, stochastic_oscillator, williams_r, commodity_channel_index, z_score,
    # Momentum indicators
    macd, momentum, rate_of_change, awesome_oscillator,
    # Volatility indicators
    bollinger_bands, average_true_range, keltner_channels,
    # Volume indicators
    on_balance_volume, volume_weighted_average_price, money_flow_index
)

from .strategies import (
    # Trend following
    moving_average_crossover, breakout_strategy, turtle_trading,
    parabolic_sar_strategy, ichimoku_cloud_strategy,
    # Mean reversion
    bollinger_bands_reversion, rsi_mean_reversion, pairs_trading,
    statistical_arbitrage, ornstein_uhlenbeck_reversion,
    # Momentum strategies
    momentum_strategy, relative_strength_strategy, gap_trading,
)

from .ml_strategies import (
    # Machine learning strategies
    woe_trading_strategy, bayesian_network_strategy, reinforcement_learning_trader,
    ensemble_strategy,
    # Pattern recognition
    head_shoulders_pattern, double_top_bottom, triangle_breakout,
)

from .risk_management import (
    # Position sizing
    kelly_criterion, fixed_fractional, volatility_position_sizing,
    # Risk metrics
    sharpe_ratio, sortino_ratio, maximum_drawdown, value_at_risk,
    # Portfolio management
    modern_portfolio_theory, black_litterman, risk_parity,
    # Performance metrics
    calculate_roc_metrics, analyze_missing_data
)

from .fairness_metrics import (
    # Fairness and diversity metrics
    calculate_gini_metrics, calculate_fairness_metrics, calculate_diversity_metrics,
    wealth_gini_coefficient
)

# Note: Backtesting module not implemented yet
# from .backtesting import (
#     # Backtesting engine
#     backtest_strategy, performance_metrics, trade_analysis,
#     # Optimization
#     parameter_optimization, walk_forward_analysis, monte_carlo_simulation
# )

__all__ = [
    # Technical indicators
    'simple_moving_average', 'exponential_moving_average', 'weighted_moving_average',
    'hull_moving_average', 'kaufman_adaptive_moving_average', 'z_score',
    'rsi', 'stochastic_oscillator', 'williams_r', 'commodity_channel_index',
    'macd', 'momentum', 'rate_of_change', 'awesome_oscillator',
    'bollinger_bands', 'average_true_range', 'keltner_channels',
    'on_balance_volume', 'volume_weighted_average_price', 'money_flow_index',
    
    # Trading strategies
    'moving_average_crossover', 'breakout_strategy', 'turtle_trading',
    'parabolic_sar_strategy', 'ichimoku_cloud_strategy',
    'bollinger_bands_reversion', 'rsi_mean_reversion', 'pairs_trading',
    'statistical_arbitrage', 'ornstein_uhlenbeck_reversion',
    'momentum_strategy', 'relative_strength_strategy', 'gap_trading',
    
    # ML trading strategies
    'woe_trading_strategy', 'bayesian_network_strategy', 'reinforcement_learning_trader',
    'ensemble_strategy', 'head_shoulders_pattern', 'double_top_bottom', 'triangle_breakout',
    
    # Risk management
    'kelly_criterion', 'fixed_fractional', 'volatility_position_sizing',
    'sharpe_ratio', 'sortino_ratio', 'maximum_drawdown', 'value_at_risk',
    'modern_portfolio_theory', 'black_litterman', 'risk_parity',
    
    # Performance and fairness metrics
    'calculate_roc_metrics', 'analyze_missing_data', 'calculate_gini_metrics',
    'calculate_fairness_metrics', 'calculate_diversity_metrics', 'wealth_gini_coefficient',
]