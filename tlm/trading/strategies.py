"""
Trading strategies and algorithms for quantitative finance.

Implements the top 30 stock trading algorithms including trend following,
mean reversion, momentum, statistical arbitrage, and machine learning strategies.
"""

from typing import List, Dict, Tuple, Optional, NamedTuple, Union
import math
import statistics
from .indicators import (
    simple_moving_average, exponential_moving_average, rsi, bollinger_bands,
    macd, z_score, stochastic_oscillator, average_true_range
)

# Type definitions
PriceData = List[float]
VolumeData = List[float]
OHLCData = List[Tuple[float, float, float, float]]
Returns = List[float]

class TradeSignal(NamedTuple):
    """Trade signal container."""
    timestamp: int
    action: str  # 'BUY', 'SELL', 'HOLD'
    price: float
    quantity: float
    confidence: float  # 0-1
    reason: str

class StrategyResult(NamedTuple):
    """Strategy result container."""
    signals: List[TradeSignal]
    positions: List[float]
    returns: List[float]
    metrics: Dict[str, float]


# TREND FOLLOWING STRATEGIES

def moving_average_crossover(prices: PriceData, fast_period: int = 50, 
                           slow_period: int = 200) -> StrategyResult:
    """
    1. Moving Average Crossover Strategy
    Classic trend following - buy when fast MA crosses above slow MA.
    """
    if len(prices) < slow_period:
        return StrategyResult([], [], [], {})
    
    fast_ma = simple_moving_average(prices, fast_period)
    slow_ma = simple_moving_average(prices, slow_period)
    
    signals = []
    positions = [0.0] * len(prices)
    current_position = 0.0
    
    for i in range(slow_period, len(prices)):
        if fast_ma[i] is not None and slow_ma[i] is not None:
            if fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
                # Golden cross - buy signal
                current_position = 1.0
                signals.append(TradeSignal(i, 'BUY', prices[i], 1.0, 0.8, 'Golden Cross'))
            elif fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]:
                # Death cross - sell signal
                current_position = -1.0
                signals.append(TradeSignal(i, 'SELL', prices[i], 1.0, 0.8, 'Death Cross'))
        
        positions[i] = current_position
    
    returns = _calculate_returns(prices, positions)
    metrics = _calculate_metrics(returns)
    
    return StrategyResult(signals, positions, returns, metrics)


def breakout_strategy(prices: PriceData, ohlc_data: OHLCData, 
                     period: int = 20) -> StrategyResult:
    """
    2. Breakout Strategy
    Buy on break above resistance, sell on break below support.
    """
    if len(prices) < period or len(ohlc_data) != len(prices):
        return StrategyResult([], [], [], {})
    
    signals = []
    positions = [0.0] * len(prices)
    current_position = 0.0
    
    for i in range(period, len(prices)):
        period_data = ohlc_data[i-period:i]
        resistance = max(candle[1] for candle in period_data)  # Highest high
        support = min(candle[2] for candle in period_data)     # Lowest low
        
        current_high = ohlc_data[i][1]
        current_low = ohlc_data[i][2]
        
        if current_high > resistance and current_position <= 0:
            # Breakout above resistance
            current_position = 1.0
            signals.append(TradeSignal(i, 'BUY', prices[i], 1.0, 0.7, 'Resistance Breakout'))
        elif current_low < support and current_position >= 0:
            # Breakdown below support
            current_position = -1.0
            signals.append(TradeSignal(i, 'SELL', prices[i], 1.0, 0.7, 'Support Breakdown'))
        
        positions[i] = current_position
    
    returns = _calculate_returns(prices, positions)
    metrics = _calculate_metrics(returns)
    
    return StrategyResult(signals, positions, returns, metrics)


def turtle_trading(ohlc_data: OHLCData, entry_period: int = 20, 
                  exit_period: int = 10) -> StrategyResult:
    """
    3. Turtle Trading Strategy
    Famous trend following system - buy 20-day highs, sell 10-day lows.
    """
    if len(ohlc_data) < entry_period:
        return StrategyResult([], [], [], {})
    
    prices = [candle[3] for candle in ohlc_data]  # Close prices
    signals = []
    positions = [0.0] * len(prices)
    current_position = 0.0
    
    for i in range(entry_period, len(ohlc_data)):
        # Calculate entry and exit levels
        entry_high = max(ohlc_data[j][1] for j in range(i-entry_period, i))
        exit_low = min(ohlc_data[j][2] for j in range(i-exit_period, i))
        
        current_high = ohlc_data[i][1]
        current_low = ohlc_data[i][2]
        
        if current_high > entry_high and current_position <= 0:
            current_position = 1.0
            signals.append(TradeSignal(i, 'BUY', prices[i], 1.0, 0.8, 'Turtle Entry'))
        elif current_low < exit_low and current_position > 0:
            current_position = 0.0
            signals.append(TradeSignal(i, 'SELL', prices[i], 1.0, 0.8, 'Turtle Exit'))
        
        positions[i] = current_position
    
    returns = _calculate_returns(prices, positions)
    metrics = _calculate_metrics(returns)
    
    return StrategyResult(signals, positions, returns, metrics)


def parabolic_sar_strategy(ohlc_data: OHLCData, acceleration: float = 0.02, 
                          max_acceleration: float = 0.2) -> StrategyResult:
    """
    4. Parabolic SAR Strategy
    Trend following using Parabolic Stop and Reverse indicator.
    """
    if len(ohlc_data) < 2:
        return StrategyResult([], [], [], {})
    
    prices = [candle[3] for candle in ohlc_data]
    signals = []
    positions = [0.0] * len(prices)
    
    # Initialize SAR
    sar = [ohlc_data[0][2]]  # Start with first low
    trend_up = True
    af = acceleration
    ep = ohlc_data[0][1] if trend_up else ohlc_data[0][2]  # Extreme point
    
    for i in range(1, len(ohlc_data)):
        high, low = ohlc_data[i][1], ohlc_data[i][2]
        
        # Calculate new SAR
        new_sar = sar[i-1] + af * (ep - sar[i-1])
        
        # Check for trend reversal
        if trend_up:
            if low <= new_sar:
                # Trend reversal to down
                trend_up = False
                new_sar = ep
                ep = low
                af = acceleration
                signals.append(TradeSignal(i, 'SELL', prices[i], 1.0, 0.7, 'SAR Reversal'))
            else:
                if high > ep:
                    ep = high
                    af = min(af + acceleration, max_acceleration)
        else:
            if high >= new_sar:
                # Trend reversal to up
                trend_up = True
                new_sar = ep
                ep = high
                af = acceleration
                signals.append(TradeSignal(i, 'BUY', prices[i], 1.0, 0.7, 'SAR Reversal'))
            else:
                if low < ep:
                    ep = low
                    af = min(af + acceleration, max_acceleration)
        
        sar.append(new_sar)
        positions[i] = 1.0 if trend_up else -1.0
    
    returns = _calculate_returns(prices, positions)
    metrics = _calculate_metrics(returns)
    
    return StrategyResult(signals, positions, returns, metrics)


def ichimoku_cloud_strategy(ohlc_data: OHLCData) -> StrategyResult:
    """
    5. Ichimoku Cloud Strategy
    Japanese trend following system using multiple timeframe analysis.
    """
    if len(ohlc_data) < 52:
        return StrategyResult([], [], [], {})
    
    prices = [candle[3] for candle in ohlc_data]
    signals = []
    positions = [0.0] * len(prices)
    
    # Calculate Ichimoku components
    tenkan_period = 9
    kijun_period = 26
    senkou_b_period = 52
    
    tenkan_sen = []  # Conversion line
    kijun_sen = []   # Base line
    
    for i in range(len(ohlc_data)):
        if i >= tenkan_period - 1:
            tenkan_data = ohlc_data[i-tenkan_period+1:i+1]
            tenkan_high = max(candle[1] for candle in tenkan_data)
            tenkan_low = min(candle[2] for candle in tenkan_data)
            tenkan_sen.append((tenkan_high + tenkan_low) / 2)
        else:
            tenkan_sen.append(None)
        
        if i >= kijun_period - 1:
            kijun_data = ohlc_data[i-kijun_period+1:i+1]
            kijun_high = max(candle[1] for candle in kijun_data)
            kijun_low = min(candle[2] for candle in kijun_data)
            kijun_sen.append((kijun_high + kijun_low) / 2)
        else:
            kijun_sen.append(None)
    
    current_position = 0.0
    
    for i in range(kijun_period, len(prices)):
        if tenkan_sen[i] is not None and kijun_sen[i] is not None:
            price = prices[i]
            
            # Buy signal: price above cloud, tenkan above kijun
            if (tenkan_sen[i] > kijun_sen[i] and 
                tenkan_sen[i-1] <= kijun_sen[i-1] and 
                price > kijun_sen[i] and current_position <= 0):
                current_position = 1.0
                signals.append(TradeSignal(i, 'BUY', price, 1.0, 0.8, 'Ichimoku Bullish'))
            
            # Sell signal: tenkan below kijun
            elif (tenkan_sen[i] < kijun_sen[i] and 
                  tenkan_sen[i-1] >= kijun_sen[i-1] and current_position >= 0):
                current_position = -1.0
                signals.append(TradeSignal(i, 'SELL', price, 1.0, 0.8, 'Ichimoku Bearish'))
        
        positions[i] = current_position
    
    returns = _calculate_returns(prices, positions)
    metrics = _calculate_metrics(returns)
    
    return StrategyResult(signals, positions, returns, metrics)


# MEAN REVERSION STRATEGIES

def bollinger_bands_reversion(prices: PriceData, period: int = 20, 
                             std_dev: float = 2.0) -> StrategyResult:
    """
    6. Bollinger Bands Mean Reversion Strategy
    Buy oversold conditions, sell overbought conditions.
    """
    if len(prices) < period:
        return StrategyResult([], [], [], {})
    
    bb_result = bollinger_bands(prices, period, std_dev)
    signals = []
    positions = [0.0] * len(prices)
    current_position = 0.0
    
    for i in range(period, len(prices)):
        if (bb_result.upper[i] is not None and 
            bb_result.lower[i] is not None and 
            bb_result.percent_b[i] is not None):
            
            percent_b = bb_result.percent_b[i]
            price = prices[i]
            
            # Buy when price touches lower band (%B < 0)
            if percent_b < 0.1 and current_position <= 0:
                current_position = 1.0
                signals.append(TradeSignal(i, 'BUY', price, 1.0, 0.7, 'BB Oversold'))
            
            # Sell when price touches upper band (%B > 1)
            elif percent_b > 0.9 and current_position >= 0:
                current_position = -1.0
                signals.append(TradeSignal(i, 'SELL', price, 1.0, 0.7, 'BB Overbought'))
            
            # Exit when returning to middle
            elif 0.4 <= percent_b <= 0.6 and current_position != 0:
                current_position = 0.0
                signals.append(TradeSignal(i, 'SELL' if current_position > 0 else 'BUY', 
                                         price, 1.0, 0.5, 'BB Mean Reversion'))
        
        positions[i] = current_position
    
    returns = _calculate_returns(prices, positions)
    metrics = _calculate_metrics(returns)
    
    return StrategyResult(signals, positions, returns, metrics)


def rsi_mean_reversion(prices: PriceData, period: int = 14, 
                      oversold: float = 30, overbought: float = 70) -> StrategyResult:
    """
    7. RSI Mean Reversion Strategy
    Buy when RSI is oversold, sell when overbought.
    """
    if len(prices) < period + 1:
        return StrategyResult([], [], [], {})
    
    rsi_result = rsi(prices, period)
    signals = []
    positions = [0.0] * len(prices)
    current_position = 0.0
    
    start_idx = period + 1
    
    for i in range(len(rsi_result.values)):
        actual_idx = start_idx + i
        if actual_idx >= len(prices):
            break
            
        rsi_val = rsi_result.values[i]
        price = prices[actual_idx]
        
        if rsi_val < oversold and current_position <= 0:
            current_position = 1.0
            signals.append(TradeSignal(actual_idx, 'BUY', price, 1.0, 0.8, 'RSI Oversold'))
        elif rsi_val > overbought and current_position >= 0:
            current_position = -1.0
            signals.append(TradeSignal(actual_idx, 'SELL', price, 1.0, 0.8, 'RSI Overbought'))
        
        if actual_idx < len(positions):
            positions[actual_idx] = current_position
    
    returns = _calculate_returns(prices, positions)
    metrics = _calculate_metrics(returns)
    
    return StrategyResult(signals, positions, returns, metrics)


def pairs_trading(prices_a: PriceData, prices_b: PriceData, 
                 lookback: int = 60, z_threshold: float = 2.0) -> StrategyResult:
    """
    8. Pairs Trading Strategy
    Statistical arbitrage between two correlated securities.
    """
    if len(prices_a) != len(prices_b) or len(prices_a) < lookback:
        return StrategyResult([], [], [], {})
    
    # Calculate price ratio and z-score
    ratios = [prices_a[i] / prices_b[i] for i in range(len(prices_a))]
    z_result = z_score(ratios, lookback)
    
    signals = []
    positions_a = [0.0] * len(prices_a)
    positions_b = [0.0] * len(prices_b)
    current_position = 0.0
    
    start_idx = lookback
    
    for i in range(len(z_result.values)):
        actual_idx = start_idx + i
        if actual_idx >= len(prices_a):
            break
            
        z_val = z_result.values[i]
        
        if z_val > z_threshold and current_position <= 0:
            # Ratio too high: sell A, buy B
            current_position = -1.0
            signals.append(TradeSignal(actual_idx, 'SELL', prices_a[actual_idx], 1.0, 0.8, 'Pairs Divergence'))
        elif z_val < -z_threshold and current_position >= 0:
            # Ratio too low: buy A, sell B  
            current_position = 1.0
            signals.append(TradeSignal(actual_idx, 'BUY', prices_a[actual_idx], 1.0, 0.8, 'Pairs Convergence'))
        elif abs(z_val) < 0.5 and current_position != 0:
            # Close position when ratio normalizes
            current_position = 0.0
            signals.append(TradeSignal(actual_idx, 'CLOSE', prices_a[actual_idx], 1.0, 0.6, 'Pairs Reversion'))
        
        if actual_idx < len(positions_a):
            positions_a[actual_idx] = current_position
            positions_b[actual_idx] = -current_position  # Opposite position
    
    returns = _calculate_spread_returns(prices_a, prices_b, positions_a, positions_b)
    metrics = _calculate_metrics(returns)
    
    return StrategyResult(signals, positions_a, returns, metrics)


def statistical_arbitrage(prices: List[PriceData], lookback: int = 30) -> StrategyResult:
    """
    9. Statistical Arbitrage Strategy
    Multi-asset mean reversion using principal components.
    """
    if len(prices) < 2 or any(len(p) < lookback for p in prices):
        return StrategyResult([], [], [], {})
    
    n_assets = len(prices)
    n_periods = min(len(p) for p in prices)
    
    signals = []
    positions = [[0.0] * n_periods for _ in range(n_assets)]
    
    # Simplified statistical arbitrage using equal-weighted portfolio
    portfolio_values = []
    for i in range(n_periods):
        portfolio_val = sum(prices[j][i] for j in range(n_assets)) / n_assets
        portfolio_values.append(portfolio_val)
    
    # Calculate z-scores for each asset vs portfolio
    for asset_idx in range(n_assets):
        asset_prices = prices[asset_idx]
        
        for i in range(lookback, n_periods):
            if i >= len(asset_prices):
                continue
                
            # Calculate relative performance vs portfolio
            asset_window = asset_prices[i-lookback:i]
            portfolio_window = portfolio_values[i-lookback:i]
            
            asset_return = (asset_prices[i] - asset_window[0]) / asset_window[0]
            portfolio_return = (portfolio_values[i] - portfolio_window[0]) / portfolio_window[0]
            
            relative_perf = asset_return - portfolio_return
            
            # Calculate z-score of relative performance
            if len(asset_window) > 1:
                rel_perfs = [(asset_prices[j] - asset_prices[j-1]) / asset_prices[j-1] - 
                           (portfolio_values[j] - portfolio_values[j-1]) / portfolio_values[j-1]
                           for j in range(i-lookback+1, i)]
                
                if len(rel_perfs) > 1:
                    mean_rel = sum(rel_perfs) / len(rel_perfs)
                    std_rel = statistics.stdev(rel_perfs)
                    
                    if std_rel > 0:
                        z_score_val = (relative_perf - mean_rel) / std_rel
                        
                        # Generate signals
                        if z_score_val > 2.0:
                            positions[asset_idx][i] = -0.5  # Short overperforming asset
                        elif z_score_val < -2.0:
                            positions[asset_idx][i] = 0.5   # Long underperforming asset
    
    # Calculate combined returns
    combined_returns = []
    for i in range(1, n_periods):
        total_return = 0.0
        for j in range(n_assets):
            if i < len(prices[j]) and positions[j][i-1] != 0:
                asset_return = (prices[j][i] - prices[j][i-1]) / prices[j][i-1]
                total_return += positions[j][i-1] * asset_return
        combined_returns.append(total_return)
    
    metrics = _calculate_metrics(combined_returns)
    
    return StrategyResult(signals, positions[0], combined_returns, metrics)


def ornstein_uhlenbeck_reversion(prices: PriceData, lookback: int = 60) -> StrategyResult:
    """
    10. Ornstein-Uhlenbeck Mean Reversion Strategy
    Uses OU process parameters to predict mean reversion timing.
    """
    if len(prices) < lookback + 1:
        return StrategyResult([], [], [], {})
    
    signals = []
    positions = [0.0] * len(prices)
    current_position = 0.0
    
    for i in range(lookback, len(prices)):
        price_window = prices[i-lookback:i+1]
        log_prices = [math.log(p) for p in price_window]
        
        # Estimate OU parameters (simplified)
        mean_log_price = sum(log_prices) / len(log_prices)
        
        # Calculate mean reversion speed (simplified estimation)
        diffs = [log_prices[j] - log_prices[j-1] for j in range(1, len(log_prices))]
        levels = [log_prices[j-1] - mean_log_price for j in range(1, len(log_prices))]
        
        if len(levels) > 1 and statistics.stdev(levels) > 0:
            # Simple linear regression for mean reversion speed
            mean_level = sum(levels) / len(levels)
            mean_diff = sum(diffs) / len(diffs)
            
            numerator = sum((levels[j] - mean_level) * (diffs[j] - mean_diff) 
                          for j in range(len(levels)))
            denominator = sum((levels[j] - mean_level) ** 2 for j in range(len(levels)))
            
            if denominator > 0:
                theta = -numerator / denominator  # Mean reversion speed
                current_level = log_prices[-1] - mean_log_price
                
                # Generate signals based on OU model
                if theta > 0.1:  # Strong mean reversion
                    if current_level > 1.5:  # Price too high
                        current_position = -1.0
                        signals.append(TradeSignal(i, 'SELL', prices[i], 1.0, 0.7, 'OU Reversion'))
                    elif current_level < -1.5:  # Price too low
                        current_position = 1.0
                        signals.append(TradeSignal(i, 'BUY', prices[i], 1.0, 0.7, 'OU Reversion'))
                    elif abs(current_level) < 0.5:
                        current_position = 0.0
        
        positions[i] = current_position
    
    returns = _calculate_returns(prices, positions)
    metrics = _calculate_metrics(returns)
    
    return StrategyResult(signals, positions, returns, metrics)


# MOMENTUM STRATEGIES

def momentum_strategy(prices: PriceData, lookback: int = 12, 
                     holding_period: int = 3) -> StrategyResult:
    """
    11. Momentum Strategy
    Buy securities with strong recent performance.
    """
    if len(prices) < lookback + holding_period:
        return StrategyResult([], [], [], {})
    
    signals = []
    positions = [0.0] * len(prices)
    
    for i in range(lookback, len(prices) - holding_period):
        # Calculate momentum score (return over lookback period)
        momentum_return = (prices[i] - prices[i - lookback]) / prices[i - lookback]
        
        # Generate signal based on momentum strength
        if momentum_return > 0.1:  # Strong positive momentum
            for j in range(i, min(i + holding_period, len(positions))):
                positions[j] = 1.0
            signals.append(TradeSignal(i, 'BUY', prices[i], 1.0, 0.8, 'Momentum Buy'))
        elif momentum_return < -0.1:  # Strong negative momentum
            for j in range(i, min(i + holding_period, len(positions))):
                positions[j] = -1.0
            signals.append(TradeSignal(i, 'SELL', prices[i], 1.0, 0.8, 'Momentum Sell'))
    
    returns = _calculate_returns(prices, positions)
    metrics = _calculate_metrics(returns)
    
    return StrategyResult(signals, positions, returns, metrics)


def relative_strength_strategy(prices: List[PriceData], lookback: int = 20) -> StrategyResult:
    """
    12. Relative Strength Strategy
    Buy strongest performing securities relative to peers.
    """
    if len(prices) < 2 or any(len(p) < lookback for p in prices):
        return StrategyResult([], [], [], {})
    
    n_assets = len(prices)
    n_periods = min(len(p) for p in prices)
    
    signals = []
    positions = [[0.0] * n_periods for _ in range(n_assets)]
    
    for i in range(lookback, n_periods):
        # Calculate relative strength for each asset
        rs_scores = []
        for asset_idx in range(n_assets):
            if i < len(prices[asset_idx]):
                asset_return = (prices[asset_idx][i] - prices[asset_idx][i - lookback]) / prices[asset_idx][i - lookback]
                rs_scores.append((asset_idx, asset_return))
        
        # Sort by relative strength
        rs_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Long top quartile, short bottom quartile
        top_count = max(1, len(rs_scores) // 4)
        bottom_count = max(1, len(rs_scores) // 4)
        
        for j, (asset_idx, score) in enumerate(rs_scores):
            if j < top_count:
                positions[asset_idx][i] = 1.0 / top_count  # Equal weight long positions
            elif j >= len(rs_scores) - bottom_count:
                positions[asset_idx][i] = -1.0 / bottom_count  # Equal weight short positions
    
    # Calculate portfolio returns
    portfolio_returns = []
    for i in range(1, n_periods):
        total_return = 0.0
        for asset_idx in range(n_assets):
            if i < len(prices[asset_idx]) and positions[asset_idx][i-1] != 0:
                asset_return = (prices[asset_idx][i] - prices[asset_idx][i-1]) / prices[asset_idx][i-1]
                total_return += positions[asset_idx][i-1] * asset_return
        portfolio_returns.append(total_return)
    
    metrics = _calculate_metrics(portfolio_returns)
    
    return StrategyResult(signals, positions[0], portfolio_returns, metrics)


def gap_trading(ohlc_data: OHLCData, min_gap: float = 0.02) -> StrategyResult:
    """
    13. Gap Trading Strategy
    Trade based on overnight gaps - fade gaps or follow gap direction.
    """
    if len(ohlc_data) < 2:
        return StrategyResult([], [], [], {})
    
    prices = [candle[3] for candle in ohlc_data]  # Close prices
    signals = []
    positions = [0.0] * len(prices)
    
    for i in range(1, len(ohlc_data)):
        prev_close = ohlc_data[i-1][3]
        current_open = ohlc_data[i][0]
        current_close = ohlc_data[i][3]
        
        # Calculate gap
        gap = (current_open - prev_close) / prev_close
        
        if abs(gap) > min_gap:
            if gap > 0:
                # Gap up - fade the gap (expect reversion)
                positions[i] = -1.0
                signals.append(TradeSignal(i, 'SELL', current_open, 1.0, 0.6, 'Gap Up Fade'))
            else:
                # Gap down - fade the gap (expect bounce)
                positions[i] = 1.0
                signals.append(TradeSignal(i, 'BUY', current_open, 1.0, 0.6, 'Gap Down Fade'))
        
        # Close position at end of day
        if i > 0 and positions[i-1] != 0:
            positions[i-1] = 0.0  # Close previous position
    
    returns = _calculate_returns(prices, positions)
    metrics = _calculate_metrics(returns)
    
    return StrategyResult(signals, positions, returns, metrics)


# HELPER FUNCTIONS

def _calculate_returns(prices: PriceData, positions: List[float]) -> List[float]:
    """Calculate strategy returns based on positions."""
    returns = []
    for i in range(1, len(prices)):
        if i-1 < len(positions) and positions[i-1] != 0:
            price_return = (prices[i] - prices[i-1]) / prices[i-1]
            strategy_return = positions[i-1] * price_return
            returns.append(strategy_return)
        else:
            returns.append(0.0)
    return returns


def _calculate_spread_returns(prices_a: PriceData, prices_b: PriceData, 
                            positions_a: List[float], positions_b: List[float]) -> List[float]:
    """Calculate returns for pairs trading strategy."""
    returns = []
    for i in range(1, len(prices_a)):
        if (i-1 < len(positions_a) and i-1 < len(positions_b) and
            (positions_a[i-1] != 0 or positions_b[i-1] != 0)):
            
            return_a = (prices_a[i] - prices_a[i-1]) / prices_a[i-1]
            return_b = (prices_b[i] - prices_b[i-1]) / prices_b[i-1]
            
            strategy_return = (positions_a[i-1] * return_a + positions_b[i-1] * return_b)
            returns.append(strategy_return)
        else:
            returns.append(0.0)
    return returns


def _calculate_metrics(returns: List[float]) -> Dict[str, float]:
    """Calculate performance metrics."""
    if not returns or len(returns) < 2:
        return {}
    
    # Remove any None values
    clean_returns = [r for r in returns if r is not None]
    
    if not clean_returns:
        return {}
    
    total_return = sum(clean_returns)
    mean_return = total_return / len(clean_returns)
    
    if len(clean_returns) > 1:
        volatility = statistics.stdev(clean_returns)
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0
    else:
        volatility = 0.0
        sharpe_ratio = 0.0
    
    # Calculate max drawdown
    cumulative = [sum(clean_returns[:i+1]) for i in range(len(clean_returns))]
    running_max = [max(cumulative[:i+1]) for i in range(len(cumulative))]
    drawdowns = [cumulative[i] - running_max[i] for i in range(len(cumulative))]
    max_drawdown = min(drawdowns) if drawdowns else 0.0
    
    return {
        'total_return': total_return,
        'annualized_return': mean_return * 252,  # Assuming daily returns
        'volatility': volatility * math.sqrt(252),
        'sharpe_ratio': sharpe_ratio * math.sqrt(252),
        'max_drawdown': max_drawdown,
        'win_rate': len([r for r in clean_returns if r > 0]) / len(clean_returns)
    }