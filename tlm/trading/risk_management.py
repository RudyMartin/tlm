"""
Risk management and performance metrics for trading algorithms.

Implements comprehensive risk management techniques, portfolio optimization,
missing data analysis (MCAR/MAR/MNAR), and performance metrics including
AUC-ROC variants and trading-specific measures.
"""

from typing import List, Dict, Tuple, Optional, NamedTuple, Union
import math
from ..pure.ops import asum, mean, median, std as stdev, var as variance

# Type definitions
PriceData = List[float]
Returns = List[float]
Weights = List[float]
CovarianceMatrix = List[List[float]]

class RiskMetrics(NamedTuple):
    """Container for risk metrics."""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    beta: float
    alpha: float

class MissingDataAnalysis(NamedTuple):
    """Missing data pattern analysis."""
    missing_percentage: float
    mcar_test_p: float  # Little's MCAR test p-value (approximated)
    pattern_type: str   # 'MCAR', 'MAR', 'MNAR'
    recommendations: List[str]

class ROCMetrics(NamedTuple):
    """ROC analysis results."""
    auc_roc: float
    auc_pr: float  # Precision-Recall AUC
    gini_coefficient: float
    ks_statistic: float
    sensitivity: float
    specificity: float
    optimal_threshold: float


# MISSING DATA ANALYSIS (MCAR/MAR/MNAR)

def analyze_missing_data(data: List[List[Optional[float]]], 
                        feature_names: Optional[List[str]] = None) -> MissingDataAnalysis:
    """
    Analyze missing data patterns to determine if data is MCAR, MAR, or MNAR.
    
    MCAR: Missing Completely At Random - missingness independent of observed/unobserved data
    MAR: Missing At Random - missingness depends on observed data
    MNAR: Missing Not At Random - missingness depends on unobserved data
    
    Args:
        data: 2D list with potential None values
        feature_names: Optional names for features
        
    Returns:
        MissingDataAnalysis with pattern classification and recommendations
    """
    if not data or not data[0]:
        return MissingDataAnalysis(0.0, 1.0, 'MCAR', [])
    
    n_rows = len(data)
    n_cols = len(data[0])
    total_cells = n_rows * n_cols
    
    # Calculate missing data statistics
    missing_count = 0
    missing_pattern = {}
    col_missing_counts = [0] * n_cols
    
    for i, row in enumerate(data):
        pattern = tuple(1 if x is None else 0 for x in row)
        missing_pattern[pattern] = missing_pattern.get(pattern, 0) + 1
        
        for j, value in enumerate(row):
            if value is None:
                missing_count += 1
                col_missing_counts[j] += 1
    
    missing_percentage = (missing_count / total_cells) * 100
    
    if missing_percentage == 0:
        return MissingDataAnalysis(0.0, 1.0, 'MCAR', ['No missing data detected'])
    
    # Simplified MCAR test (approximation of Little's test)
    expected_patterns = _calculate_expected_missing_patterns(col_missing_counts, n_rows)
    mcar_test_stat = _calculate_mcar_test_statistic(missing_pattern, expected_patterns)
    mcar_p_value = _chi_square_p_value(mcar_test_stat, len(missing_pattern) - 1)
    
    # Classify missing data pattern
    if mcar_p_value > 0.05:
        pattern_type = 'MCAR'
        recommendations = [
            'Data appears to be Missing Completely At Random',
            'Listwise deletion or simple imputation methods are appropriate',
            'Consider mean/median imputation for numerical features'
        ]
    elif missing_percentage < 20:
        pattern_type = 'MAR'
        recommendations = [
            'Data appears to be Missing At Random',
            'Use multiple imputation methods',
            'Consider model-based imputation using observed features',
            'Avoid listwise deletion to prevent bias'
        ]
    else:
        pattern_type = 'MNAR'
        recommendations = [
            'Data may be Missing Not At Random',
            'Investigate reasons for missingness',
            'Consider selection models or pattern mixture models',
            'Domain expertise required for proper handling'
        ]
    
    return MissingDataAnalysis(missing_percentage, mcar_p_value, pattern_type, recommendations)


def impute_missing_data(data: List[List[Optional[float]]], 
                       method: str = 'mean') -> List[List[float]]:
    """
    Impute missing data using specified method.
    
    Args:
        data: 2D list with potential None values
        method: 'mean', 'median', 'mode', 'forward_fill', 'backward_fill'
        
    Returns:
        2D list with imputed values
    """
    if not data or not data[0]:
        return []
    
    n_rows = len(data)
    n_cols = len(data[0])
    imputed_data = [row[:] for row in data]  # Deep copy
    
    for col in range(n_cols):
        col_values = [row[col] for row in data if row[col] is not None]
        
        if not col_values:
            continue  # Skip columns with all missing values
        
        if method == 'mean':
            impute_value = asum(col_values) / len(col_values)
        elif method == 'median':
            impute_value = median(col_values)
        elif method == 'mode':
            # For mode, find most frequent value
            from collections import Counter
            impute_value = Counter(col_values).most_common(1)[0][0]
        else:
            impute_value = asum(col_values) / len(col_values)  # Default to mean
        
        # Apply imputation
        for row in range(n_rows):
            if imputed_data[row][col] is None:
                if method == 'forward_fill' and row > 0:
                    imputed_data[row][col] = imputed_data[row-1][col]
                elif method == 'backward_fill' and row < n_rows - 1:
                    # Look ahead for next valid value
                    for next_row in range(row + 1, n_rows):
                        if data[next_row][col] is not None:
                            imputed_data[row][col] = data[next_row][col]
                            break
                    else:
                        imputed_data[row][col] = impute_value
                else:
                    imputed_data[row][col] = impute_value
    
    return imputed_data


# PERFORMANCE METRICS (AUC-ROC VARIANTS)

def calculate_roc_metrics(y_true: List[int], y_scores: List[float], 
                         thresholds: Optional[List[float]] = None) -> ROCMetrics:
    """
    Calculate comprehensive ROC-based performance metrics.
    
    Args:
        y_true: True binary labels (0, 1)
        y_scores: Predicted scores/probabilities
        thresholds: Optional thresholds to evaluate
        
    Returns:
        ROCMetrics with AUC-ROC, AUC-PR, Gini, KS statistic, etc.
    """
    if len(y_true) != len(y_scores):
        return ROCMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    if thresholds is None:
        # Create thresholds from unique scores
        unique_scores = sorted(set(y_scores))
        thresholds = unique_scores + [min(unique_scores) - 0.001, max(unique_scores) + 0.001]
    
    # Calculate ROC curve points
    roc_points = []
    pr_points = []
    
    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        
        tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
        fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
        tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
        fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
        
        # ROC metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 1 - Specificity
        
        # Precision-Recall metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        roc_points.append((fpr, tpr))
        pr_points.append((tpr, precision))  # (recall, precision)
    
    # Sort points
    roc_points.sort()
    pr_points.sort()
    
    # Calculate AUC-ROC using trapezoidal rule
    auc_roc = 0.0
    for i in range(1, len(roc_points)):
        auc_roc += (roc_points[i][0] - roc_points[i-1][0]) * \
                   (roc_points[i][1] + roc_points[i-1][1]) / 2
    
    # Calculate AUC-PR using trapezoidal rule
    auc_pr = 0.0
    for i in range(1, len(pr_points)):
        auc_pr += (pr_points[i][0] - pr_points[i-1][0]) * \
                  (pr_points[i][1] + pr_points[i-1][1]) / 2
    
    # Gini coefficient (2 * AUC - 1)
    gini_coefficient = 2 * auc_roc - 1
    
    # Kolmogorov-Smirnov statistic (max distance between TPR and FPR)
    ks_statistic = max(tpr - fpr for fpr, tpr in roc_points)
    
    # Find optimal threshold (maximizes Youden's J statistic)
    best_threshold = 0.5
    best_j = 0.0
    
    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        
        tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
        fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
        tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
        fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        j_statistic = sensitivity + specificity - 1
        
        if j_statistic > best_j:
            best_j = j_statistic
            best_threshold = threshold
    
    # Calculate final metrics at optimal threshold
    y_pred_optimal = [1 if score >= best_threshold else 0 for score in y_scores]
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred_optimal[i] == 1)
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred_optimal[i] == 1)
    tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred_optimal[i] == 0)
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred_optimal[i] == 0)
    
    final_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    final_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return ROCMetrics(
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        gini_coefficient=gini_coefficient,
        ks_statistic=ks_statistic,
        sensitivity=final_sensitivity,
        specificity=final_specificity,
        optimal_threshold=best_threshold
    )


# POSITION SIZING METHODS

def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate Kelly Criterion for optimal position sizing.
    
    Kelly % = (bp - q) / b
    where b = odds (avg_win/avg_loss), p = win_rate, q = 1-p
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    
    b = avg_win / avg_loss  # Odds
    p = win_rate
    q = 1 - p
    
    kelly_pct = (b * p - q) / b
    
    # Apply safety constraint (max 25% position)
    return max(0.0, min(kelly_pct, 0.25))


def fixed_fractional(capital: float, risk_per_trade: float = 0.02) -> float:
    """
    Calculate fixed fractional position size.
    
    Args:
        capital: Total capital available
        risk_per_trade: Risk percentage per trade (default 2%)
        
    Returns:
        Position size
    """
    return capital * risk_per_trade


def volatility_position_sizing(prices: PriceData, target_volatility: float = 0.15,
                              lookback: int = 20) -> List[float]:
    """
    Calculate position sizes based on volatility targeting.
    
    Position size inversely proportional to realized volatility.
    """
    if len(prices) < lookback + 1:
        return [1.0] * len(prices)
    
    position_sizes = [1.0] * lookback  # Initial positions
    
    for i in range(lookback, len(prices)):
        # Calculate realized volatility
        returns = []
        for j in range(i - lookback, i):
            if j > 0:
                ret = (prices[j] - prices[j-1]) / prices[j-1]
                returns.append(ret)
        
        if len(returns) > 1:
            realized_vol = stdev(returns) * math.sqrt(252)  # Annualized
            
            # Scale position inversely with volatility
            position_size = target_volatility / max(realized_vol, 0.01)
            position_size = max(0.1, min(position_size, 2.0))  # Bounds
        else:
            position_size = 1.0
        
        position_sizes.append(position_size)
    
    return position_sizes


# RISK METRICS CALCULATION

def sharpe_ratio(returns: Returns, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio (excess return per unit of volatility)."""
    if len(returns) < 2:
        return 0.0
    
    mean_return = asum(returns) / len(returns)
    excess_return = mean_return - risk_free_rate / 252  # Daily risk-free rate
    volatility = stdev(returns)
    
    if volatility == 0:
        return 0.0
    
    return (excess_return / volatility) * math.sqrt(252)  # Annualized


def sortino_ratio(returns: Returns, risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio (excess return per unit of downside deviation)."""
    if len(returns) < 2:
        return 0.0
    
    mean_return = asum(returns) / len(returns)
    excess_return = mean_return - risk_free_rate / 252
    
    # Calculate downside deviation
    negative_returns = [r for r in returns if r < 0]
    if len(negative_returns) < 2:
        return float('inf') if excess_return > 0 else 0.0
    
    downside_deviation = stdev(negative_returns)
    
    if downside_deviation == 0:
        return float('inf') if excess_return > 0 else 0.0
    
    return (excess_return / downside_deviation) * math.sqrt(252)


def maximum_drawdown(returns: Returns) -> float:
    """Calculate maximum drawdown."""
    if not returns:
        return 0.0
    
    cumulative_returns = []
    cumulative = 0.0
    
    for ret in returns:
        cumulative += ret
        cumulative_returns.append(cumulative)
    
    max_dd = 0.0
    peak = cumulative_returns[0]
    
    for cum_ret in cumulative_returns:
        if cum_ret > peak:
            peak = cum_ret
        
        drawdown = (peak - cum_ret) / (1 + peak) if peak != 0 else 0
        max_dd = max(max_dd, drawdown)
    
    return max_dd


def value_at_risk(returns: Returns, confidence_level: float = 0.05) -> float:
    """Calculate Value at Risk (VaR) at given confidence level."""
    if len(returns) < 2:
        return 0.0
    
    sorted_returns = sorted(returns)
    var_index = int(confidence_level * len(sorted_returns))
    
    return -sorted_returns[var_index] if var_index < len(sorted_returns) else 0.0


def conditional_value_at_risk(returns: Returns, confidence_level: float = 0.05) -> float:
    """Calculate Conditional Value at Risk (CVaR/Expected Shortfall)."""
    if len(returns) < 2:
        return 0.0
    
    var = value_at_risk(returns, confidence_level)
    tail_losses = [r for r in returns if -r >= var]
    
    if not tail_losses:
        return var
    
    return -asum(tail_losses) / len(tail_losses)


def calculate_risk_metrics(returns: Returns, benchmark_returns: Optional[Returns] = None,
                          risk_free_rate: float = 0.02) -> RiskMetrics:
    """
    Calculate comprehensive risk metrics for a return series.
    
    Args:
        returns: Strategy returns
        benchmark_returns: Optional benchmark returns for beta/alpha calculation
        risk_free_rate: Annual risk-free rate
        
    Returns:
        RiskMetrics object with all calculated metrics
    """
    if not returns:
        return RiskMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    sharpe = sharpe_ratio(returns, risk_free_rate)
    sortino = sortino_ratio(returns, risk_free_rate)
    max_dd = maximum_drawdown(returns)
    
    # Calmar ratio (annual return / max drawdown)
    annual_return = (asum(returns) / len(returns)) * 252
    calmar = annual_return / max_dd if max_dd != 0 else 0.0
    
    var_95 = value_at_risk(returns, 0.05)
    cvar_95 = conditional_value_at_risk(returns, 0.05)
    
    # Beta and Alpha (if benchmark provided)
    beta = 0.0
    alpha = 0.0
    
    if benchmark_returns and len(benchmark_returns) == len(returns) and len(returns) > 1:
        # Calculate beta using covariance
        returns_mean = asum(returns) / len(returns)
        benchmark_mean = asum(benchmark_returns) / len(benchmark_returns)
        
        covariance = asum((returns[i] - returns_mean) * (benchmark_returns[i] - benchmark_mean)
                         for i in range(len(returns))) / len(returns)
        
        benchmark_variance = asum((r - benchmark_mean) ** 2 for r in benchmark_returns) / len(benchmark_returns)
        
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0.0
        
        # Jensen's Alpha
        alpha = (returns_mean - risk_free_rate / 252) - beta * (benchmark_mean - risk_free_rate / 252)
        alpha *= 252  # Annualize
    
    return RiskMetrics(
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        var_95=var_95,
        cvar_95=cvar_95,
        beta=beta,
        alpha=alpha
    )


# PORTFOLIO OPTIMIZATION

def modern_portfolio_theory(expected_returns: List[float], 
                           covariance_matrix: CovarianceMatrix,
                           target_return: Optional[float] = None) -> Weights:
    """
    Optimize portfolio weights using Modern Portfolio Theory.
    
    Simplified implementation - finds minimum variance portfolio or
    efficient frontier point for target return.
    """
    n_assets = len(expected_returns)
    
    if (len(covariance_matrix) != n_assets or 
        any(len(row) != n_assets for row in covariance_matrix)):
        return [1.0 / n_assets] * n_assets  # Equal weights fallback
    
    # Simplified optimization using equal risk contribution as approximation
    if target_return is None:
        # Minimum variance portfolio (simplified)
        inv_vol_weights = []
        for i in range(n_assets):
            volatility = math.sqrt(covariance_matrix[i][i])
            inv_vol_weights.append(1.0 / max(volatility, 0.001))
        
        total_weight = asum(inv_vol_weights)
        return [w / total_weight for w in inv_vol_weights]
    
    # Target return portfolio (simplified mean-variance optimization)
    # This is a basic approximation - full MVO requires quadratic programming
    mean_return = asum(expected_returns) / len(expected_returns)
    
    weights = []
    for i, exp_ret in enumerate(expected_returns):
        # Weight based on distance from target return
        distance = abs(exp_ret - target_return)
        weight = 1.0 / (1.0 + distance)
        weights.append(weight)
    
    total_weight = asum(weights)
    return [w / total_weight for w in weights]


def black_litterman(market_caps: List[float], expected_returns: List[float],
                   investor_views: Dict[int, Tuple[float, float]]) -> Weights:
    """
    Black-Litterman portfolio optimization with investor views.
    
    Args:
        market_caps: Market capitalizations for each asset
        expected_returns: Historical expected returns
        investor_views: Dict of {asset_index: (view_return, confidence)}
        
    Returns:
        Optimized portfolio weights
    """
    n_assets = len(market_caps)
    
    # Start with market cap weights
    total_cap = asum(market_caps)
    market_weights = [cap / total_cap for cap in market_caps]
    
    # Adjust weights based on investor views (simplified)
    adjusted_weights = market_weights[:]
    
    for asset_idx, (view_return, confidence) in investor_views.items():
        if 0 <= asset_idx < n_assets:
            expected_ret = expected_returns[asset_idx]
            
            # Adjust weight based on view vs expected return
            adjustment = (view_return - expected_ret) * confidence
            adjusted_weights[asset_idx] *= (1.0 + adjustment)
    
    # Normalize weights
    total_weight = asum(adjusted_weights)
    return [w / total_weight for w in adjusted_weights] if total_weight > 0 else market_weights


def risk_parity(covariance_matrix: CovarianceMatrix) -> Weights:
    """
    Calculate risk parity weights (equal risk contribution).
    
    Each asset contributes equally to portfolio risk.
    """
    n_assets = len(covariance_matrix)
    
    if any(len(row) != n_assets for row in covariance_matrix):
        return [1.0 / n_assets] * n_assets
    
    # Simplified risk parity using inverse volatility
    volatilities = [math.sqrt(covariance_matrix[i][i]) for i in range(n_assets)]
    inv_vol_weights = [1.0 / max(vol, 0.001) for vol in volatilities]
    
    total_weight = asum(inv_vol_weights)
    return [w / total_weight for w in inv_vol_weights]


# HELPER FUNCTIONS

def _calculate_expected_missing_patterns(col_missing_counts: List[int], 
                                       n_rows: int) -> Dict[Tuple[int, ...], float]:
    """Calculate expected missing patterns under MCAR assumption."""
    n_cols = len(col_missing_counts)
    missing_probs = [count / n_rows for count in col_missing_counts]
    
    expected_patterns = {}
    
    # Generate all possible patterns (simplified - just compute a few common ones)
    for i in range(2 ** min(n_cols, 8)):  # Limit to avoid exponential explosion
        pattern = tuple((i >> j) & 1 for j in range(n_cols))
        
        # Calculate expected count under independence
        prob = 1.0
        for j, missing_bit in enumerate(pattern):
            if missing_bit:
                prob *= missing_probs[j]
            else:
                prob *= (1 - missing_probs[j])
        
        expected_patterns[pattern] = prob * n_rows
    
    return expected_patterns


def _calculate_mcar_test_statistic(observed_patterns: Dict[Tuple[int, ...], int],
                                 expected_patterns: Dict[Tuple[int, ...], float]) -> float:
    """Calculate test statistic for MCAR test."""
    chi_square = 0.0
    
    for pattern in set(observed_patterns.keys()) | set(expected_patterns.keys()):
        observed = observed_patterns.get(pattern, 0)
        expected = expected_patterns.get(pattern, 0.1)  # Small positive value
        
        if expected > 0:
            chi_square += ((observed - expected) ** 2) / expected
    
    return chi_square


def _chi_square_p_value(test_statistic: float, degrees_freedom: int) -> float:
    """Approximate chi-square p-value (simplified implementation)."""
    if degrees_freedom <= 0:
        return 1.0
    
    # Very rough approximation using normal approximation for large df
    if degrees_freedom > 30:
        z_score = (test_statistic - degrees_freedom) / math.sqrt(2 * degrees_freedom)
        return max(0.001, min(0.999, 0.5 * (1 - math.erf(z_score / math.sqrt(2)))))
    
    # Simple threshold-based approximation for small df
    critical_values = {1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49, 5: 11.07}
    critical = critical_values.get(degrees_freedom, degrees_freedom * 2.0)
    
    if test_statistic < critical:
        return 0.1  # Not significant
    else:
        return 0.01  # Significant