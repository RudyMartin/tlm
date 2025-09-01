"""
Time series decomposition algorithms for TLM signal processing.

Provides classical and advanced methods to decompose time series into
trend, seasonal, and remainder (noise) components - essential for
understanding underlying patterns in data.
"""

from typing import List, Tuple, Optional, Dict, Union
import math
from collections import namedtuple

# Type definitions
TimeSeries = List[float]
DecompositionResult = namedtuple('DecompositionResult', ['trend', 'seasonal', 'remainder', 'observed'])

def additive_decompose(series: TimeSeries, period: int, 
                      trend_method: str = 'moving_average') -> DecompositionResult:
    """
    Classical additive decomposition: Y(t) = Trend(t) + Seasonal(t) + Remainder(t)
    
    Args:
        series: Time series data
        period: Seasonal period length
        trend_method: Method for trend extraction ('moving_average', 'linear')
    
    Returns:
        DecompositionResult with trend, seasonal, remainder components
    """
    n = len(series)
    
    if n < 2 * period:
        raise ValueError(f"Series length {n} must be at least 2 * period ({2 * period})")
    
    # Step 1: Extract trend
    if trend_method == 'moving_average':
        trend = _centered_moving_average(series, period)
    elif trend_method == 'linear':
        trend = _linear_trend_detrend(series)[0]
    else:
        raise ValueError(f"Unknown trend method: {trend_method}")
    
    # Step 2: Detrend the series
    detrended = [series[i] - trend[i] if trend[i] is not None else None 
                 for i in range(n)]
    
    # Step 3: Extract seasonal component
    seasonal = _extract_seasonal_additive(detrended, period)
    
    # Step 4: Calculate remainder
    remainder = []
    for i in range(n):
        if trend[i] is not None and seasonal[i] is not None:
            remainder.append(series[i] - trend[i] - seasonal[i])
        else:
            remainder.append(None)
    
    return DecompositionResult(trend=trend, seasonal=seasonal, 
                             remainder=remainder, observed=series)


def multiplicative_decompose(series: TimeSeries, period: int,
                            trend_method: str = 'moving_average') -> DecompositionResult:
    """
    Classical multiplicative decomposition: Y(t) = Trend(t) × Seasonal(t) × Remainder(t)
    
    Args:
        series: Time series data (must be positive)
        period: Seasonal period length
        trend_method: Method for trend extraction
    
    Returns:
        DecompositionResult with multiplicative components
    """
    n = len(series)
    
    if n < 2 * period:
        raise ValueError(f"Series length {n} must be at least 2 * period ({2 * period})")
    
    # Check for non-positive values
    if any(x <= 0 for x in series if x is not None):
        raise ValueError("Multiplicative decomposition requires positive values")
    
    # Step 1: Extract trend
    if trend_method == 'moving_average':
        trend = _centered_moving_average(series, period)
    else:
        trend = _linear_trend_detrend(series)[0]
    
    # Step 2: Detrend the series (divide by trend)
    detrended = [series[i] / trend[i] if trend[i] is not None and trend[i] != 0 else None 
                 for i in range(n)]
    
    # Step 3: Extract seasonal component (multiplicative)
    seasonal = _extract_seasonal_multiplicative(detrended, period)
    
    # Step 4: Calculate remainder
    remainder = []
    for i in range(n):
        if (trend[i] is not None and seasonal[i] is not None and 
            trend[i] != 0 and seasonal[i] != 0):
            remainder.append(series[i] / (trend[i] * seasonal[i]))
        else:
            remainder.append(None)
    
    return DecompositionResult(trend=trend, seasonal=seasonal, 
                             remainder=remainder, observed=series)


def seasonal_decompose(series: TimeSeries, model: str = 'additive', 
                      period: Optional[int] = None, extrapolate_trend: int = 0) -> DecompositionResult:
    """
    General seasonal decomposition function (similar to R's decompose()).
    
    Args:
        series: Time series data
        model: 'additive' or 'multiplicative'
        period: Seasonal period (auto-detected if None)
        extrapolate_trend: Number of periods to extrapolate trend
    
    Returns:
        DecompositionResult
    """
    if period is None:
        period = _detect_period(series)
    
    if model == 'additive':
        result = additive_decompose(series, period)
    elif model == 'multiplicative':
        result = multiplicative_decompose(series, period)
    else:
        raise ValueError(f"Unknown model: {model}. Use 'additive' or 'multiplicative'")
    
    # Extrapolate trend if requested
    if extrapolate_trend > 0:
        result = _extrapolate_trend(result, extrapolate_trend, period)
    
    return result


def moving_average_trend(series: TimeSeries, window: int) -> List[Optional[float]]:
    """
    Extract trend using moving average smoothing.
    
    Args:
        series: Time series data
        window: Moving average window size
    
    Returns:
        Smoothed trend series
    """
    return _simple_moving_average(series, window)


def stl_decompose(series: TimeSeries, seasonal_len: int, trend_len: int,
                 robust: bool = False, seasonal_deg: int = 1, 
                 trend_deg: int = 1, seasonal_jump: int = 1,
                 trend_jump: int = 1, inner_iter: int = 2, 
                 outer_iter: int = 0) -> DecompositionResult:
    """
    STL: Seasonal and Trend decomposition using Loess.
    
    This is a simplified implementation of the STL algorithm.
    
    Args:
        series: Time series data
        seasonal_len: Length of seasonal smoother (should be odd)
        trend_len: Length of trend smoother (should be odd)  
        robust: Use robust fitting
        seasonal_deg: Degree of seasonal smoother (0 or 1)
        trend_deg: Degree of trend smoother (0 or 1)
        seasonal_jump: Seasonal smoother jump
        trend_jump: Trend smoother jump
        inner_iter: Number of inner iterations
        outer_iter: Number of outer iterations (for robust fitting)
    
    Returns:
        DecompositionResult
    """
    n = len(series)
    period = _detect_period(series) if seasonal_len is None else seasonal_len
    
    # Initialize components
    seasonal = [0.0] * n
    trend = [0.0] * n
    remainder = [0.0] * n
    weights = [1.0] * n
    
    # STL iteration
    for outer in range(outer_iter + 1):
        for inner in range(inner_iter):
            # Step 1: Seasonal smoothing
            # Detrend: Y - T
            detrended = [series[i] - trend[i] for i in range(n)]
            
            # Smooth seasonal component
            seasonal = _seasonal_smoother_loess(detrended, period, seasonal_len, 
                                              seasonal_deg, weights)
            
            # Step 2: Trend smoothing  
            # Deseasonalize: Y - S
            deseasoned = [series[i] - seasonal[i] for i in range(n)]
            
            # Smooth trend component
            trend = _trend_smoother_loess(deseasoned, trend_len, trend_deg, weights)
        
        # Calculate remainder
        remainder = [series[i] - trend[i] - seasonal[i] for i in range(n)]
        
        # Update weights for robust fitting
        if robust and outer < outer_iter:
            weights = _calculate_robust_weights(remainder)
    
    return DecompositionResult(trend=trend, seasonal=seasonal, 
                             remainder=remainder, observed=series)


def robust_stl_decompose(series: TimeSeries, seasonal_len: int, 
                        trend_len: int) -> DecompositionResult:
    """
    Robust STL decomposition with automatic parameter selection.
    
    Args:
        series: Time series data
        seasonal_len: Seasonal smoother length
        trend_len: Trend smoother length
    
    Returns:
        DecompositionResult
    """
    return stl_decompose(series, seasonal_len, trend_len, robust=True, 
                        outer_iter=1, inner_iter=5)


def trend_strength(decomposition: DecompositionResult) -> float:
    """
    Calculate strength of trend component.
    
    Trend strength = 1 - Var(Remainder) / Var(Y - Seasonal)
    
    Args:
        decomposition: Result from decomposition function
    
    Returns:
        Trend strength between 0 and 1
    """
    # Remove None values
    observed = [x for x in decomposition.observed if x is not None]
    seasonal = [x for x in decomposition.seasonal if x is not None]
    remainder = [x for x in decomposition.remainder if x is not None]
    
    if len(observed) != len(seasonal) or len(observed) != len(remainder):
        # Align series
        aligned = _align_series([decomposition.observed, decomposition.seasonal, 
                               decomposition.remainder])
        observed, seasonal, remainder = aligned
    
    # Deseasonalized series
    deseason = [observed[i] - seasonal[i] for i in range(len(observed))]
    
    var_remainder = _variance(remainder)
    var_deseason = _variance(deseason)
    
    if var_deseason == 0:
        return 0.0
    
    strength = max(0.0, 1.0 - var_remainder / var_deseason)
    return min(1.0, strength)


def seasonal_strength(decomposition: DecompositionResult) -> float:
    """
    Calculate strength of seasonal component.
    
    Seasonal strength = 1 - Var(Remainder) / Var(Y - Trend)
    
    Args:
        decomposition: Result from decomposition function
    
    Returns:
        Seasonal strength between 0 and 1
    """
    # Align series
    aligned = _align_series([decomposition.observed, decomposition.trend, 
                           decomposition.remainder])
    observed, trend, remainder = aligned
    
    # Detrended series
    detrend = [observed[i] - trend[i] for i in range(len(observed))]
    
    var_remainder = _variance(remainder)
    var_detrend = _variance(detrend)
    
    if var_detrend == 0:
        return 0.0
    
    strength = max(0.0, 1.0 - var_remainder / var_detrend)
    return min(1.0, strength)


def remainder_strength(decomposition: DecompositionResult) -> float:
    """
    Calculate strength of remainder (noise) component.
    
    Args:
        decomposition: Result from decomposition function
    
    Returns:
        Remainder strength between 0 and 1
    """
    aligned = _align_series([decomposition.observed, decomposition.remainder])
    observed, remainder = aligned
    
    var_remainder = _variance(remainder)
    var_observed = _variance(observed)
    
    if var_observed == 0:
        return 0.0
    
    return var_remainder / var_observed


# Helper functions

def _centered_moving_average(series: TimeSeries, period: int) -> List[Optional[float]]:
    """Centered moving average for trend extraction."""
    n = len(series)
    trend = [None] * n
    
    if period % 2 == 0:
        # Even period: use 2x(period) MA
        k = period // 2
        for i in range(k, n - k):
            # First MA
            ma1_start = max(0, i - k)
            ma1_end = min(n, i + k + 1)
            ma1 = sum(series[ma1_start:ma1_end]) / (ma1_end - ma1_start)
            
            # Second MA (offset by 1)
            ma2_start = max(0, i - k + 1)  
            ma2_end = min(n, i + k + 2)
            ma2 = sum(series[ma2_start:ma2_end]) / (ma2_end - ma2_start)
            
            trend[i] = (ma1 + ma2) / 2
    else:
        # Odd period: simple centered MA
        k = period // 2
        for i in range(k, n - k):
            start = i - k
            end = i + k + 1
            trend[i] = sum(series[start:end]) / period
    
    return trend


def _simple_moving_average(series: TimeSeries, window: int) -> List[Optional[float]]:
    """Simple moving average."""
    n = len(series)
    result = [None] * n
    
    for i in range(window - 1, n):
        start = i - window + 1
        result[i] = sum(series[start:i + 1]) / window
    
    return result


def _extract_seasonal_additive(detrended: List[Optional[float]], 
                              period: int) -> List[Optional[float]]:
    """Extract seasonal component for additive model."""
    n = len(detrended)
    seasonal = [None] * n
    
    # Calculate seasonal averages for each season
    seasonal_means = [0.0] * period
    seasonal_counts = [0] * period
    
    for i, val in enumerate(detrended):
        if val is not None:
            season = i % period
            seasonal_means[season] += val
            seasonal_counts[season] += 1
    
    # Average and center
    for i in range(period):
        if seasonal_counts[i] > 0:
            seasonal_means[i] /= seasonal_counts[i]
    
    # Center seasonal component (sum should be 0)
    seasonal_sum = sum(seasonal_means)
    seasonal_mean = seasonal_sum / period
    seasonal_means = [s - seasonal_mean for s in seasonal_means]
    
    # Replicate across full series
    for i in range(n):
        seasonal[i] = seasonal_means[i % period]
    
    return seasonal


def _extract_seasonal_multiplicative(detrended: List[Optional[float]], 
                                   period: int) -> List[Optional[float]]:
    """Extract seasonal component for multiplicative model."""
    n = len(detrended)
    seasonal = [None] * n
    
    # Calculate seasonal averages for each season
    seasonal_means = [1.0] * period
    seasonal_counts = [0] * period
    
    for i, val in enumerate(detrended):
        if val is not None:
            season = i % period
            if seasonal_counts[season] == 0:
                seasonal_means[season] = val
            else:
                seasonal_means[season] *= val
            seasonal_counts[season] += 1
    
    # Geometric average
    for i in range(period):
        if seasonal_counts[i] > 0:
            seasonal_means[i] = seasonal_means[i] ** (1.0 / seasonal_counts[i])
    
    # Center seasonal component (product should be 1)
    seasonal_product = 1.0
    for s in seasonal_means:
        seasonal_product *= s
    seasonal_geometric_mean = seasonal_product ** (1.0 / period)
    
    seasonal_means = [s / seasonal_geometric_mean for s in seasonal_means]
    
    # Replicate across full series
    for i in range(n):
        seasonal[i] = seasonal_means[i % period]
    
    return seasonal


def _linear_trend_detrend(series: TimeSeries) -> Tuple[List[float], float, float]:
    """Linear trend estimation using least squares."""
    n = len(series)
    t = list(range(n))
    
    # Calculate means
    t_mean = sum(t) / n
    y_mean = sum(series) / n
    
    # Calculate slope
    numerator = sum((t[i] - t_mean) * (series[i] - y_mean) for i in range(n))
    denominator = sum((t[i] - t_mean) ** 2 for i in range(n))
    
    slope = numerator / denominator if denominator != 0 else 0
    intercept = y_mean - slope * t_mean
    
    # Generate trend
    trend = [intercept + slope * i for i in range(n)]
    
    return trend, slope, intercept


def _detect_period(series: TimeSeries, max_period: Optional[int] = None) -> int:
    """Auto-detect seasonal period using autocorrelation."""
    n = len(series)
    if max_period is None:
        max_period = min(n // 2, 50)  # Reasonable upper bound
    
    autocorrs = []
    for lag in range(1, max_period + 1):
        autocorr = _autocorrelation(series, lag)
        autocorrs.append((lag, autocorr))
    
    # Find lag with maximum autocorrelation
    best_lag = max(autocorrs, key=lambda x: abs(x[1]))[0]
    return best_lag


def _autocorrelation(series: TimeSeries, lag: int) -> float:
    """Calculate autocorrelation at given lag."""
    n = len(series)
    if lag >= n:
        return 0.0
    
    mean_val = sum(series) / n
    
    numerator = sum((series[i] - mean_val) * (series[i + lag] - mean_val) 
                   for i in range(n - lag))
    denominator = sum((series[i] - mean_val) ** 2 for i in range(n))
    
    return numerator / denominator if denominator != 0 else 0.0


def _seasonal_smoother_loess(series: TimeSeries, period: int, 
                           seasonal_len: int, degree: int, 
                           weights: List[float]) -> List[float]:
    """Simplified seasonal smoother (placeholder for full LOESS)."""
    # This is a simplified version - full LOESS is complex
    return _simple_moving_average_with_weights(series, seasonal_len, weights)


def _trend_smoother_loess(series: TimeSeries, trend_len: int, 
                         degree: int, weights: List[float]) -> List[float]:
    """Simplified trend smoother (placeholder for full LOESS).""" 
    # This is a simplified version - full LOESS is complex
    return _simple_moving_average_with_weights(series, trend_len, weights)


def _simple_moving_average_with_weights(series: TimeSeries, window: int, 
                                      weights: List[float]) -> List[float]:
    """Weighted moving average."""
    n = len(series)
    result = [0.0] * n
    k = window // 2
    
    for i in range(n):
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for j in range(max(0, i - k), min(n, i + k + 1)):
            weighted_sum += series[j] * weights[j]
            weight_sum += weights[j]
        
        result[i] = weighted_sum / weight_sum if weight_sum > 0 else series[i]
    
    return result


def _calculate_robust_weights(residuals: List[float]) -> List[float]:
    """Calculate robust weights based on residuals."""
    n = len(residuals)
    abs_residuals = [abs(r) for r in residuals]
    
    # Calculate median absolute deviation
    sorted_abs = sorted(abs_residuals)
    median_abs = sorted_abs[n // 2] if n > 0 else 0
    mad = median_abs * 1.4826  # Scale factor for normal distribution
    
    # Bi-square weights
    weights = []
    for r in abs_residuals:
        if mad == 0:
            weights.append(1.0)
        else:
            u = r / (6 * mad)  # 6 * MAD cutoff
            if u < 1:
                w = (1 - u * u) ** 2
            else:
                w = 0.0
            weights.append(w)
    
    return weights


def _variance(series: List[float]) -> float:
    """Calculate variance."""
    n = len(series)
    if n == 0:
        return 0.0
    
    mean_val = sum(series) / n
    return sum((x - mean_val) ** 2 for x in series) / n


def _align_series(series_list: List[List[Optional[float]]]) -> List[List[float]]:
    """Align multiple series by removing positions where any series has None."""
    n = len(series_list[0])
    aligned = [[] for _ in series_list]
    
    for i in range(n):
        if all(series[i] is not None for series in series_list):
            for j, series in enumerate(series_list):
                aligned[j].append(series[i])
    
    return aligned


def _extrapolate_trend(result: DecompositionResult, periods: int, 
                      period_len: int) -> DecompositionResult:
    """Extrapolate trend component."""
    # Simple linear extrapolation of trend
    trend = result.trend
    n = len(trend)
    
    # Find last few non-None trend values
    valid_trends = [(i, t) for i, t in enumerate(trend) if t is not None]
    if len(valid_trends) < 2:
        return result
    
    # Linear extrapolation based on last trend points
    (i1, t1), (i2, t2) = valid_trends[-2:]
    slope = (t2 - t1) / (i2 - i1) if i2 != i1 else 0
    
    # Extend trend
    extended_trend = trend.copy()
    for i in range(n, n + periods * period_len):
        extended_trend.append(t2 + slope * (i - i2))
    
    # Extend other components as needed
    return DecompositionResult(
        trend=extended_trend,
        seasonal=result.seasonal,
        remainder=result.remainder,
        observed=result.observed
    )