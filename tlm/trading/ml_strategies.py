"""
Machine learning trading strategies for quantitative finance.

Implements advanced ML-based trading algorithms including Weight of Evidence,
Bayesian Networks, reinforcement learning, and ensemble methods.
"""

from typing import List, Dict, Tuple, Optional, NamedTuple, Union
import math
from ..pure.ops import asum, mean, median, std as stdev, var as variance

# Type definitions
PriceData = List[float]
Features = List[List[float]]  # 2D feature matrix
Labels = List[int]  # Binary labels (0, 1)
Probabilities = List[float]

class MLSignal(NamedTuple):
    """ML-based trading signal."""
    timestamp: int
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0-1
    features: List[float]
    prediction: float

class WOEResult(NamedTuple):
    """Weight of Evidence calculation result."""
    woe_values: Dict[str, float]  # bin -> WOE value
    iv_score: float  # Information Value
    bins: List[Tuple[float, float]]  # bin boundaries

class BayesianNode(NamedTuple):
    """Node in Bayesian Network."""
    name: str
    parents: List[str]
    probabilities: Dict[str, float]  # conditional probability table


# WEIGHT OF EVIDENCE (WOE) STRATEGY

def calculate_woe(feature_values: List[float], labels: List[int], 
                 n_bins: int = 5) -> WOEResult:
    """
    Calculate Weight of Evidence for a continuous feature.
    
    WOE measures the strength of a feature in separating good/bad outcomes.
    Used extensively in credit scoring and risk modeling.
    
    Args:
        feature_values: Continuous feature values
        labels: Binary labels (0=bad, 1=good)
        n_bins: Number of bins for discretization
        
    Returns:
        WOEResult with WOE values, IV score, and bin boundaries
    """
    if len(feature_values) != len(labels) or len(feature_values) == 0:
        return WOEResult({}, 0.0, [])
    
    # Create bins
    sorted_values = sorted(feature_values)
    bin_size = len(sorted_values) // n_bins
    bins = []
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_values)
        
        if start_idx < len(sorted_values):
            min_val = sorted_values[start_idx]
            max_val = sorted_values[end_idx - 1] if end_idx <= len(sorted_values) else sorted_values[-1]
            bins.append((min_val, max_val))
    
    # Calculate WOE for each bin
    total_good = sum(labels)
    total_bad = len(labels) - total_good
    
    if total_good == 0 or total_bad == 0:
        return WOEResult({}, 0.0, bins)
    
    woe_values = {}
    iv_score = 0.0
    
    for i, (min_val, max_val) in enumerate(bins):
        # Count good and bad in this bin
        bin_good = 0
        bin_bad = 0
        
        for j, val in enumerate(feature_values):
            if min_val <= val <= max_val:
                if labels[j] == 1:
                    bin_good += 1
                else:
                    bin_bad += 1
        
        if bin_good > 0 and bin_bad > 0:
            # Calculate WOE
            good_rate = bin_good / total_good
            bad_rate = bin_bad / total_bad
            woe = math.log(good_rate / bad_rate)
            
            # Calculate Information Value contribution
            iv_contrib = (good_rate - bad_rate) * woe
            iv_score += iv_contrib
            
            woe_values[f'bin_{i}'] = woe
        else:
            woe_values[f'bin_{i}'] = 0.0
    
    return WOEResult(woe_values, iv_score, bins)


def woe_trading_strategy(prices: PriceData, volumes: List[float], 
                        lookback: int = 20) -> List[MLSignal]:
    """
    14. Weight of Evidence Trading Strategy
    Uses WOE analysis to predict price movements based on technical features.
    """
    if len(prices) != len(volumes) or len(prices) < lookback * 2:
        return []
    
    signals = []
    
    for i in range(lookback, len(prices) - lookback):
        # Extract features for current window
        price_window = prices[i-lookback:i]
        volume_window = volumes[i-lookback:i]
        future_window = prices[i:i+lookback]
        
        # Calculate technical features
        features = []
        
        # Price momentum
        momentum = (prices[i] - prices[i-10]) / prices[i-10] if i >= 10 else 0
        features.append(momentum)
        
        # Volume ratio
        avg_volume = asum(volume_window) / len(volume_window)
        volume_ratio = volumes[i] / avg_volume if avg_volume > 0 else 1
        features.append(volume_ratio)
        
        # Volatility
        if len(price_window) > 1:
            returns = [(price_window[j] - price_window[j-1]) / price_window[j-1] 
                      for j in range(1, len(price_window))]
            volatility = stdev(returns) if len(returns) > 1 else 0
        else:
            volatility = 0
        features.append(volatility)
        
        # Create label (future return positive/negative)
        future_return = (future_window[-1] - future_window[0]) / future_window[0]
        label = 1 if future_return > 0 else 0
        
        # For demonstration, use simple WOE-based prediction
        # In practice, you'd train on historical data
        if len(features) >= 3:
            # Simple WOE-based scoring
            woe_score = 0.0
            
            # Momentum WOE (simplified)
            if momentum > 0.05:
                woe_score += 0.3
            elif momentum < -0.05:
                woe_score -= 0.3
            
            # Volume WOE (simplified)
            if volume_ratio > 1.5:
                woe_score += 0.2
            elif volume_ratio < 0.5:
                woe_score -= 0.2
            
            # Volatility WOE (simplified)
            if volatility > 0.02:
                woe_score -= 0.1  # High volatility is risky
            
            # Generate signal
            confidence = min(abs(woe_score), 1.0)
            
            if woe_score > 0.2:
                action = 'BUY'
            elif woe_score < -0.2:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            signals.append(MLSignal(i, action, confidence, features, woe_score))
    
    return signals


# BAYESIAN NETWORK STRATEGY

def create_market_bayesian_network() -> List[BayesianNode]:
    """
    Create a simple Bayesian Network for market analysis.
    
    Network structure:
    Market Sentiment -> Price Direction
    Volume Surge -> Price Direction
    Technical Signal -> Price Direction
    """
    nodes = []
    
    # Market Sentiment node (prior)
    sentiment_node = BayesianNode(
        name='market_sentiment',
        parents=[],
        probabilities={
            'bullish': 0.4,
            'neutral': 0.4,
            'bearish': 0.2
        }
    )
    nodes.append(sentiment_node)
    
    # Volume Surge node (prior)
    volume_node = BayesianNode(
        name='volume_surge',
        parents=[],
        probabilities={
            'high': 0.2,
            'normal': 0.6,
            'low': 0.2
        }
    )
    nodes.append(volume_node)
    
    # Technical Signal node (prior)  
    technical_node = BayesianNode(
        name='technical_signal',
        parents=[],
        probabilities={
            'buy': 0.3,
            'hold': 0.4,
            'sell': 0.3
        }
    )
    nodes.append(technical_node)
    
    # Price Direction node (depends on all others)
    price_direction_node = BayesianNode(
        name='price_direction',
        parents=['market_sentiment', 'volume_surge', 'technical_signal'],
        probabilities={
            # P(up | bullish, high, buy) = 0.8
            'bullish,high,buy,up': 0.8,
            'bullish,high,buy,down': 0.2,
            # P(up | bullish, high, hold) = 0.6
            'bullish,high,hold,up': 0.6,
            'bullish,high,hold,down': 0.4,
            # ... (simplified for demo)
            # In practice, you'd learn these from data
            'bearish,low,sell,up': 0.1,
            'bearish,low,sell,down': 0.9,
        }
    )
    nodes.append(price_direction_node)
    
    return nodes


def bayesian_inference(network: List[BayesianNode], evidence: Dict[str, str]) -> Dict[str, float]:
    """
    Perform Bayesian inference on the network given evidence.
    
    Simplified implementation - in practice you'd use junction tree algorithm.
    """
    # This is a simplified version - real Bayesian inference is complex
    
    # Find the target node (price_direction)
    target_probs = {'up': 0.5, 'down': 0.5}  # Default prior
    
    for node in network:
        if node.name == 'price_direction':
            # Simple evidence-based adjustment
            adjustment = 1.0
            
            if 'market_sentiment' in evidence:
                if evidence['market_sentiment'] == 'bullish':
                    adjustment *= 1.2
                elif evidence['market_sentiment'] == 'bearish':
                    adjustment *= 0.8
            
            if 'volume_surge' in evidence:
                if evidence['volume_surge'] == 'high':
                    adjustment *= 1.1
                elif evidence['volume_surge'] == 'low':
                    adjustment *= 0.9
            
            if 'technical_signal' in evidence:
                if evidence['technical_signal'] == 'buy':
                    adjustment *= 1.3
                elif evidence['technical_signal'] == 'sell':
                    adjustment *= 0.7
            
            # Update probabilities
            up_prob = 0.5 * adjustment
            down_prob = 1.0 - up_prob
            
            # Normalize
            total = up_prob + down_prob
            target_probs = {
                'up': up_prob / total,
                'down': down_prob / total
            }
            break
    
    return target_probs


def bayesian_network_strategy(prices: PriceData, volumes: List[float], 
                             lookback: int = 10) -> List[MLSignal]:
    """
    15. Bayesian Network Trading Strategy
    Uses probabilistic inference to make trading decisions.
    """
    if len(prices) != len(volumes) or len(prices) < lookback:
        return []
    
    network = create_market_bayesian_network()
    signals = []
    
    for i in range(lookback, len(prices)):
        # Gather evidence from market data
        evidence = {}
        
        # Market sentiment from price trend
        recent_return = (prices[i] - prices[i-lookback]) / prices[i-lookback]
        if recent_return > 0.05:
            evidence['market_sentiment'] = 'bullish'
        elif recent_return < -0.05:
            evidence['market_sentiment'] = 'bearish'
        else:
            evidence['market_sentiment'] = 'neutral'
        
        # Volume analysis
        recent_volume = asum(volumes[i-5:i]) / 5 if i >= 5 else volumes[i]
        historical_volume = asum(volumes[i-lookback:i-5]) / (lookback-5) if i >= lookback else recent_volume
        
        if recent_volume > historical_volume * 1.5:
            evidence['volume_surge'] = 'high'
        elif recent_volume < historical_volume * 0.5:
            evidence['volume_surge'] = 'low'
        else:
            evidence['volume_surge'] = 'normal'
        
        # Technical signal (simplified RSI-like)
        price_window = prices[i-14:i] if i >= 14 else prices[max(0, i-10):i]
        if len(price_window) > 1:
            gains = [max(0, price_window[j] - price_window[j-1]) for j in range(1, len(price_window))]
            losses = [max(0, price_window[j-1] - price_window[j]) for j in range(1, len(price_window))]
            
            avg_gain = asum(gains) / len(gains) if gains else 0
            avg_loss = asum(losses) / len(losses) if losses else 0.01
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            if rsi > 70:
                evidence['technical_signal'] = 'sell'
            elif rsi < 30:
                evidence['technical_signal'] = 'buy'
            else:
                evidence['technical_signal'] = 'hold'
        else:
            evidence['technical_signal'] = 'hold'
        
        # Perform Bayesian inference
        posterior = bayesian_inference(network, evidence)
        
        # Generate trading signal
        up_prob = posterior.get('up', 0.5)
        confidence = abs(up_prob - 0.5) * 2  # Scale to 0-1
        
        if up_prob > 0.6:
            action = 'BUY'
        elif up_prob < 0.4:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        features = [recent_return, recent_volume / historical_volume, rsi if 'rsi' in locals() else 50]
        
        signals.append(MLSignal(i, action, confidence, features, up_prob))
    
    return signals


# REINFORCEMENT LEARNING STRATEGY

class QLearningTrader:
    """
    Q-Learning based trading agent.
    
    States: Market conditions (discretized)
    Actions: BUY, SELL, HOLD
    Rewards: Portfolio returns
    """
    
    def __init__(self, n_states: int = 100, n_actions: int = 3, 
                 learning_rate: float = 0.1, discount: float = 0.95):
        self.n_states = n_states
        self.n_actions = n_actions  # 0: HOLD, 1: BUY, 2: SELL
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = 0.1  # Exploration rate
        
        # Initialize Q-table
        self.q_table = [[0.0 for _ in range(n_actions)] for _ in range(n_states)]
        self.action_names = ['HOLD', 'BUY', 'SELL']
    
    def discretize_state(self, features: List[float]) -> int:
        """Convert continuous features to discrete state."""
        if not features or len(features) < 3:
            return 0
        
        # Normalize and discretize features
        momentum = max(-1, min(1, features[0]))  # Clamp to [-1, 1]
        volume_ratio = max(0, min(3, features[1]))  # Clamp to [0, 3]
        rsi = max(0, min(100, features[2])) / 100  # Normalize to [0, 1]
        
        # Create state index (simplified discretization)
        momentum_bin = int((momentum + 1) * 10) % 10  # 0-9
        volume_bin = int(volume_ratio * 3) % 4  # 0-3
        rsi_bin = int(rsi * 2) % 3  # 0-2
        
        state = momentum_bin * 12 + volume_bin * 3 + rsi_bin
        return min(state, self.n_states - 1)
    
    def choose_action(self, state: int, explore: bool = True) -> int:
        """Choose action using epsilon-greedy policy."""
        if explore and hash(str(state)) % 100 < int(self.epsilon * 100):
            # Random exploration
            return hash(str(state)) % self.n_actions
        else:
            # Exploit best action
            return self.q_table[state].index(max(self.q_table[state]))
    
    def update_q_table(self, state: int, action: int, reward: float, next_state: int):
        """Update Q-table using Q-learning update rule."""
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q


def reinforcement_learning_trader(prices: PriceData, volumes: List[float], 
                                train_episodes: int = 100) -> List[MLSignal]:
    """
    16. Reinforcement Learning Trading Strategy
    Uses Q-learning to learn optimal trading actions.
    """
    if len(prices) != len(volumes) or len(prices) < 50:
        return []
    
    trader = QLearningTrader()
    signals = []
    
    # Training phase
    for episode in range(train_episodes):
        position = 0  # 0: no position, 1: long, -1: short
        portfolio_value = 1.0
        
        for i in range(20, len(prices) - 10):
            # Calculate features
            momentum = (prices[i] - prices[i-10]) / prices[i-10]
            volume_ratio = volumes[i] / (asum(volumes[i-10:i]) / 10)
            
            # Simple RSI calculation
            gains = [max(0, prices[j] - prices[j-1]) for j in range(i-13, i)]
            losses = [max(0, prices[j-1] - prices[j]) for j in range(i-13, i)]
            avg_gain = asum(gains) / len(gains) if gains else 0
            avg_loss = asum(losses) / len(losses) if losses else 0.01
            # Ensure avg_loss is never zero to prevent division by zero
            if avg_loss == 0:
                avg_loss = 0.01
            rsi = 100 - (100 / (1 + avg_gain / avg_loss))
            
            features = [momentum, volume_ratio, rsi]
            state = trader.discretize_state(features)
            
            # Choose action
            action = trader.choose_action(state, explore=(episode < train_episodes * 0.8))
            
            # Execute action and calculate reward
            next_return = (prices[i+1] - prices[i]) / prices[i]
            
            reward = 0.0
            if action == 1:  # BUY
                reward = next_return if position <= 0 else 0
                position = 1
            elif action == 2:  # SELL
                reward = -next_return if position >= 0 else 0
                position = -1
            else:  # HOLD
                reward = next_return * position if position != 0 else 0
            
            # Calculate next state
            if i + 1 < len(prices) - 1:
                next_momentum = (prices[i+1] - prices[i-9]) / prices[i-9]
                next_volume_ratio = volumes[i+1] / (asum(volumes[i-9:i+1]) / 10)
                next_rsi = rsi  # Simplified
                next_features = [next_momentum, next_volume_ratio, next_rsi]
                next_state = trader.discretize_state(next_features)
            else:
                next_state = state
            
            # Update Q-table
            trader.update_q_table(state, action, reward, next_state)
    
    # Testing/deployment phase
    position = 0
    
    for i in range(20, len(prices) - 1):
        momentum = (prices[i] - prices[i-10]) / prices[i-10]
        volume_ratio = volumes[i] / (asum(volumes[i-10:i]) / 10)
        
        gains = [max(0, prices[j] - prices[j-1]) for j in range(i-13, i)]
        losses = [max(0, prices[j-1] - prices[j]) for j in range(i-13, i)]
        avg_gain = asum(gains) / len(gains) if gains else 0
        avg_loss = asum(losses) / len(losses) if losses else 0.01
        # Ensure avg_loss is never zero to prevent division by zero
        if avg_loss == 0:
            avg_loss = 0.01
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        
        features = [momentum, volume_ratio, rsi]
        state = trader.discretize_state(features)
        
        # Choose best action (no exploration)
        action = trader.choose_action(state, explore=False)
        action_name = trader.action_names[action]
        
        # Calculate confidence based on Q-values
        q_values = trader.q_table[state]
        max_q = max(q_values)
        min_q = min(q_values)
        confidence = (max_q - min_q) / (abs(max_q) + abs(min_q) + 1e-6) if max_q != min_q else 0.5
        
        signals.append(MLSignal(i, action_name, confidence, features, max_q))
    
    return signals


# ENSEMBLE STRATEGY

def ensemble_strategy(prices: PriceData, volumes: List[float], 
                     ohlc_data: Optional[List[Tuple[float, float, float, float]]] = None) -> List[MLSignal]:
    """
    17. Ensemble Trading Strategy
    Combines multiple ML strategies using voting/averaging.
    """
    if len(prices) != len(volumes) or len(prices) < 50:
        return []
    
    # Generate signals from different strategies
    woe_signals = woe_trading_strategy(prices, volumes)
    bayesian_signals = bayesian_network_strategy(prices, volumes)
    rl_signals = reinforcement_learning_trader(prices, volumes, train_episodes=20)  # Reduced for speed
    
    # Align signals by timestamp
    all_timestamps = set()
    if woe_signals:
        all_timestamps.update(s.timestamp for s in woe_signals)
    if bayesian_signals:
        all_timestamps.update(s.timestamp for s in bayesian_signals)
    if rl_signals:
        all_timestamps.update(s.timestamp for s in rl_signals)
    
    ensemble_signals = []
    
    for timestamp in sorted(all_timestamps):
        # Collect signals for this timestamp
        timestamp_signals = []
        
        for signal_list in [woe_signals, bayesian_signals, rl_signals]:
            for signal in signal_list:
                if signal.timestamp == timestamp:
                    timestamp_signals.append(signal)
                    break
        
        if len(timestamp_signals) >= 2:  # Need at least 2 strategies to agree
            # Vote on action
            action_votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            total_confidence = 0.0
            total_prediction = 0.0
            combined_features = []
            
            for signal in timestamp_signals:
                action_votes[signal.action] += signal.confidence
                total_confidence += signal.confidence
                total_prediction += signal.prediction * signal.confidence
                if signal.features:
                    combined_features.extend(signal.features)
            
            # Determine ensemble action
            best_action = max(action_votes, key=action_votes.get)
            ensemble_confidence = total_confidence / len(timestamp_signals)
            ensemble_prediction = total_prediction / total_confidence if total_confidence > 0 else 0
            
            # Only generate signal if confidence is high enough
            if ensemble_confidence > 0.4:
                ensemble_signals.append(MLSignal(
                    timestamp, 
                    best_action, 
                    ensemble_confidence, 
                    combined_features[:10],  # Limit feature size
                    ensemble_prediction
                ))
    
    return ensemble_signals


# PATTERN RECOGNITION STRATEGIES

def head_shoulders_pattern(prices: PriceData, window: int = 20) -> List[MLSignal]:
    """
    18. Head and Shoulders Pattern Recognition
    Detects reversal patterns using peak/trough analysis.
    """
    if len(prices) < window * 3:
        return []
    
    signals = []
    
    for i in range(window, len(prices) - window):
        # Look for local maxima and minima
        window_prices = prices[i-window:i+window+1]
        center_idx = window
        
        # Check if current point is a local maximum
        if (window_prices[center_idx] > max(window_prices[:center_idx]) and 
            window_prices[center_idx] > max(window_prices[center_idx+1:])):
            
            # Look for head-shoulders pattern
            # Find left shoulder, head, right shoulder
            left_shoulder = None
            right_shoulder = None
            
            # Search for shoulders
            for j in range(5, window-5):
                left_val = window_prices[j]
                right_val = window_prices[center_idx + (center_idx - j)]
                
                if (left_val > max(window_prices[max(0, j-5):j]) and
                    left_val > max(window_prices[j+1:j+6]) and
                    right_val > max(window_prices[max(0, center_idx + (center_idx - j) - 5):center_idx + (center_idx - j)]) and
                    right_val > max(window_prices[center_idx + (center_idx - j)+1:center_idx + (center_idx - j)+6])):
                    
                    left_shoulder = left_val
                    right_shoulder = right_val
                    break
            
            # Check head-shoulders criteria
            if (left_shoulder and right_shoulder and
                window_prices[center_idx] > left_shoulder * 1.05 and
                window_prices[center_idx] > right_shoulder * 1.05 and
                abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder) < 0.1):
                
                # Head-shoulders pattern detected
                confidence = min(1.0, (window_prices[center_idx] - max(left_shoulder, right_shoulder)) / window_prices[center_idx])
                
                signals.append(MLSignal(
                    i, 'SELL', confidence, 
                    [left_shoulder, window_prices[center_idx], right_shoulder],
                    -confidence
                ))
    
    return signals


def double_top_bottom(prices: PriceData, tolerance: float = 0.02) -> List[MLSignal]:
    """
    19. Double Top/Bottom Pattern Recognition
    Identifies reversal patterns with two similar peaks/troughs.
    """
    if len(prices) < 40:
        return []
    
    signals = []
    
    # Find local extrema
    extrema = []
    for i in range(10, len(prices) - 10):
        is_peak = all(prices[i] >= prices[j] for j in range(i-10, i+11))
        is_trough = all(prices[i] <= prices[j] for j in range(i-10, i+11))
        
        if is_peak:
            extrema.append((i, prices[i], 'peak'))
        elif is_trough:
            extrema.append((i, prices[i], 'trough'))
    
    # Look for double patterns
    for i in range(len(extrema) - 1):
        for j in range(i + 1, len(extrema)):
            idx1, price1, type1 = extrema[i]
            idx2, price2, type2 = extrema[j]
            
            if type1 == type2 and abs(idx2 - idx1) > 20:  # Same type, sufficient distance
                price_similarity = abs(price1 - price2) / max(price1, price2)
                
                if price_similarity < tolerance:
                    if type1 == 'peak':
                        # Double top - bearish reversal
                        signals.append(MLSignal(
                            idx2, 'SELL', 1.0 - price_similarity,
                            [price1, price2, abs(idx2 - idx1)],
                            -(1.0 - price_similarity)
                        ))
                    else:
                        # Double bottom - bullish reversal
                        signals.append(MLSignal(
                            idx2, 'BUY', 1.0 - price_similarity,
                            [price1, price2, abs(idx2 - idx1)],
                            1.0 - price_similarity
                        ))
    
    return signals


def triangle_breakout(ohlc_data: List[Tuple[float, float, float, float]], 
                     min_touches: int = 4) -> List[MLSignal]:
    """
    20. Triangle Breakout Pattern Recognition
    Detects converging trendlines and breakout signals.
    """
    if len(ohlc_data) < 50:
        return []
    
    signals = []
    prices = [candle[3] for candle in ohlc_data]  # Close prices
    highs = [candle[1] for candle in ohlc_data]   # High prices
    lows = [candle[2] for candle in ohlc_data]    # Low prices
    
    # Look for triangle patterns
    for i in range(30, len(ohlc_data) - 10):
        # Get recent highs and lows
        recent_highs = []
        recent_lows = []
        
        for j in range(i-30, i):
            # Find local highs and lows
            if j >= 5 and j < len(highs) - 5:
                if all(highs[j] >= highs[k] for k in range(j-5, j+6)):
                    recent_highs.append((j, highs[j]))
                if all(lows[j] <= lows[k] for k in range(j-5, j+6)):
                    recent_lows.append((j, lows[j]))
        
        if len(recent_highs) >= min_touches//2 and len(recent_lows) >= min_touches//2:
            # Calculate trendlines
            high_slope = _calculate_trendline_slope(recent_highs)
            low_slope = _calculate_trendline_slope(recent_lows)
            
            # Check for converging trendlines (triangle)
            if high_slope < 0 and low_slope > 0:  # Converging triangle
                # Check for breakout
                current_high = highs[i]
                current_low = lows[i]
                
                # Calculate expected trendline values
                high_trend_value = recent_highs[-1][1] + high_slope * (i - recent_highs[-1][0])
                low_trend_value = recent_lows[-1][1] + low_slope * (i - recent_lows[-1][0])
                
                if current_high > high_trend_value:
                    # Upward breakout
                    confidence = min(1.0, (current_high - high_trend_value) / high_trend_value)
                    signals.append(MLSignal(
                        i, 'BUY', confidence,
                        [high_slope, low_slope, current_high - high_trend_value],
                        confidence
                    ))
                elif current_low < low_trend_value:
                    # Downward breakout
                    confidence = min(1.0, (low_trend_value - current_low) / low_trend_value)
                    signals.append(MLSignal(
                        i, 'SELL', confidence,
                        [high_slope, low_slope, low_trend_value - current_low],
                        -confidence
                    ))
    
    return signals


def _calculate_trendline_slope(points: List[Tuple[int, float]]) -> float:
    """Calculate slope of trendline through points using linear regression."""
    if len(points) < 2:
        return 0.0
    
    n = len(points)
    sum_x = sum(p[0] for p in points)
    sum_y = sum(p[1] for p in points)
    sum_xy = sum(p[0] * p[1] for p in points)
    sum_x2 = sum(p[0] ** 2 for p in points)
    
    denominator = n * sum_x2 - sum_x ** 2
    if denominator == 0:
        return 0.0
    
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return slope