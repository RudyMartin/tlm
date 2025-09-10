"""
Trend analysis functions for TLM signal processing.

Provides methods for detecting, modeling, and analyzing trends in time series data.
Includes linear/polynomial fitting, changepoint detection, and trend metrics.
"""

from typing import List, Tuple, Optional, Dict, Union
import math
import statistics

# Type definitions
TimeSeries = List[float]
TrendModel = Dict[str, Union[float, List[float]]]

def linear_trend(series: TimeSeries, return_residuals: bool = False) -> Union[TrendModel, Tuple[TrendModel, TimeSeries]]:
    """
    Fit linear trend to time series using least squares.
    
    Args:
        series: Input time series
        return_residuals: Whether to return residuals
        
    Returns:
        Dictionary with slope, intercept, r_squared, or tuple with residuals
    """
    n = len(series)
    if n < 2:
        model = {'slope': 0.0, 'intercept': series[0] if series else 0.0, 'r_squared': 0.0}
        if return_residuals:
            return model, []
        return model
    
    # Time points
    x = list(range(n))
    x_mean = (n - 1) / 2.0
    y_mean = sum(series) / n
    
    # Calculate slope using least squares
    numerator = sum((x[i] - x_mean) * (series[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        slope = 0.0
    else:
        slope = numerator / denominator
    
    intercept = y_mean - slope * x_mean
    
    # Calculate R-squared
    y_pred = [intercept + slope * i for i in range(n)]
    ss_res = sum((series[i] - y_pred[i]) ** 2 for i in range(n))
    ss_tot = sum((series[i] - y_mean) ** 2 for i in range(n))
    
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    model = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared
    }
    
    if return_residuals:
        residuals = [series[i] - y_pred[i] for i in range(n)]
        return model, residuals
    
    return model


def polynomial_trend(series: TimeSeries, degree: int = 2, 
                    return_residuals: bool = False) -> Union[TrendModel, Tuple[TrendModel, TimeSeries]]:
    """
    Fit polynomial trend to time series.
    
    Args:
        series: Input time series
        degree: Degree of polynomial (1=linear, 2=quadratic, etc.)
        return_residuals: Whether to return residuals
        
    Returns:
        Dictionary with coefficients and r_squared, or tuple with residuals
    """
    n = len(series)
    if n < degree + 1:
        # Fallback to constant trend
        mean_val = statistics.mean(series) if series else 0.0
        model = {'coefficients': [mean_val] + [0.0] * degree, 'r_squared': 0.0, 'degree': degree}
        if return_residuals:
            residuals = [series[i] - mean_val for i in range(n)]
            return model, residuals
        return model
    
    # Use simplified polynomial fitting for common cases
    if degree == 1:
        linear_model = linear_trend(series)
        model = {
            'coefficients': [linear_model['intercept'], linear_model['slope']],
            'r_squared': linear_model['r_squared'],
            'degree': 1
        }
    elif degree == 2:
        model = _quadratic_fit(series)
    else:
        # Fallback to linear for higher degrees
        linear_model = linear_trend(series)
        model = {
            'coefficients': [linear_model['intercept'], linear_model['slope']] + [0.0] * (degree - 1),
            'r_squared': linear_model['r_squared'],
            'degree': degree
        }
    
    if return_residuals:
        y_pred = [_evaluate_polynomial(model['coefficients'], i) for i in range(n)]
        residuals = [series[i] - y_pred[i] for i in range(n)]
        return model, residuals
    
    return model


def local_linear_trend(series: TimeSeries, window: int = 10, 
                      method: str = 'rolling') -> TimeSeries:
    """
    Calculate local linear trend using rolling window.
    
    Args:
        series: Input time series
        window: Size of rolling window
        method: 'rolling' or 'expanding'
        
    Returns:
        Local trend values
    """
    n = len(series)
    if window < 2:
        window = 2
    
    trends = []
    
    for i in range(n):
        if method == 'rolling':
            start = max(0, i - window // 2)
            end = min(n, start + window)
            if end - start < 2:
                trends.append(0.0)
                continue
        else:  # expanding
            start = 0
            end = i + 1
        
        window_series = series[start:end]
        if len(window_series) >= 2:
            trend_model = linear_trend(window_series)
            trends.append(trend_model['slope'])
        else:
            trends.append(0.0)
    
    return trends


def changepoint_detection(series: TimeSeries, method: str = 'variance', 
                         min_segment_length: int = 5) -> List[int]:
    """
    Detect changepoints in time series using statistical methods.
    
    Args:
        series: Input time series
        method: Detection method ('variance', 'mean', 'trend')
        min_segment_length: Minimum length between changepoints
        
    Returns:
        List of changepoint indices
    """
    n = len(series)
    if n < 2 * min_segment_length:
        return []
    
    if method == 'variance':
        return _variance_changepoints(series, min_segment_length)
    elif method == 'mean':
        return _mean_changepoints(series, min_segment_length)
    elif method == 'trend':
        return _trend_changepoints(series, min_segment_length)
    else:
        raise ValueError(f"Unknown method: {method}")


def trend_changepoints(series: TimeSeries, threshold: float = 0.1) -> List[int]:
    """
    Detect trend changepoints by analyzing slope changes.
    
    Args:
        series: Input time series
        threshold: Minimum slope change to detect
        
    Returns:
        List of changepoint indices
    """
    if len(series) < 6:
        return []
    
    window_size = max(3, len(series) // 20)
    slopes = local_linear_trend(series, window_size)
    
    changepoints = []
    
    for i in range(window_size, len(slopes) - window_size):
        # Compare slopes before and after
        before_slope = statistics.mean(slopes[i-window_size:i])
        after_slope = statistics.mean(slopes[i:i+window_size])
        
        slope_change = abs(after_slope - before_slope)
        
        if slope_change > threshold:
            # Check if this is a local maximum of slope change
            is_peak = True
            for j in range(max(0, i-2), min(len(slopes), i+3)):
                if j != i:
                    other_change = abs(statistics.mean(slopes[j-window_size:j]) - 
                                     statistics.mean(slopes[j:j+window_size])) if j >= window_size and j < len(slopes) - window_size else 0
                    if other_change > slope_change:
                        is_peak = False
                        break
            
            if is_peak:
                changepoints.append(i)
    
    # Remove nearby changepoints
    filtered_changepoints = []
    for cp in changepoints:
        if not filtered_changepoints or cp - filtered_changepoints[-1] > window_size:
            filtered_changepoints.append(cp)
    
    return filtered_changepoints


def trend_slope(series: TimeSeries, method: str = 'linear') -> float:
    """
    Calculate overall trend slope.
    
    Args:
        series: Input time series
        method: 'linear', 'theil_sen', or 'percentile'
        
    Returns:
        Trend slope value
    """
    if len(series) < 2:
        return 0.0
    
    if method == 'linear':
        model = linear_trend(series)
        return model['slope']
    elif method == 'theil_sen':
        return _theil_sen_slope(series)
    elif method == 'percentile':
        return _percentile_slope(series)
    else:
        raise ValueError(f"Unknown method: {method}")


def trend_acceleration(series: TimeSeries, window: int = 5) -> TimeSeries:
    """
    Calculate trend acceleration (second derivative).
    
    Args:
        series: Input time series
        window: Smoothing window size
        
    Returns:
        Trend acceleration values
    """
    if len(series) < 3:
        return [0.0] * len(series)
    
    # Calculate local slopes
    slopes = local_linear_trend(series, window)
    
    # Calculate acceleration as slope of slopes
    acceleration = []
    for i in range(len(slopes)):
        if i == 0:
            acc = 0.0
        elif i == 1:
            acc = slopes[i] - slopes[i-1]
        else:
            # Use centered difference
            acc = (slopes[i] - slopes[i-2]) / 2.0
        
        acceleration.append(acc)
    
    return acceleration


def trend_volatility(series: TimeSeries, window: int = 20, 
                    method: str = 'std') -> TimeSeries:
    """
    Calculate trend volatility using rolling statistics.
    
    Args:
        series: Input time series
        window: Rolling window size
        method: 'std', 'mad', or 'range'
        
    Returns:
        Trend volatility values
    """
    n = len(series)
    volatility = []
    
    for i in range(n):
        start = max(0, i - window + 1)
        end = i + 1
        window_data = series[start:end]
        
        if len(window_data) < 2:
            vol = 0.0
        elif method == 'std':
            vol = statistics.stdev(window_data)
        elif method == 'mad':
            median = statistics.median(window_data)
            vol = statistics.median([abs(x - median) for x in window_data])
        elif method == 'range':
            vol = max(window_data) - min(window_data)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        volatility.append(vol)
    
    return volatility


def detrend(series: TimeSeries, method: str = 'linear') -> TimeSeries:
    """
    Remove trend from time series.
    
    Args:
        series: Input time series
        method: 'linear', 'polynomial', or 'local'
        
    Returns:
        Detrended time series
    """
    if method == 'linear':
        model, residuals = linear_trend(series, return_residuals=True)
        return residuals
    elif method == 'polynomial':
        model, residuals = polynomial_trend(series, degree=2, return_residuals=True)
        return residuals
    elif method == 'local':
        # Use local linear detrending
        window = max(5, len(series) // 10)
        detrended = []
        
        for i in range(len(series)):
            start = max(0, i - window // 2)
            end = min(len(series), start + window)
            local_series = series[start:end]
            
            if len(local_series) >= 2:
                local_model = linear_trend(local_series)
                local_idx = i - start
                trend_val = local_model['intercept'] + local_model['slope'] * local_idx
                detrended.append(series[i] - trend_val)
            else:
                detrended.append(series[i])
        
        return detrended
    else:
        raise ValueError(f"Unknown method: {method}")


# Helper functions

def _quadratic_fit(series: TimeSeries) -> TrendModel:
    """Fit quadratic polynomial using least squares."""
    n = len(series)
    x = list(range(n))
    
    # Build normal equations for ax^2 + bx + c
    sum_x = sum(x)
    sum_x2 = sum(xi**2 for xi in x)
    sum_x3 = sum(xi**3 for xi in x)
    sum_x4 = sum(xi**4 for xi in x)
    sum_y = sum(series)
    sum_xy = sum(x[i] * series[i] for i in range(n))
    sum_x2y = sum(x[i]**2 * series[i] for i in range(n))
    
    # Solve 3x3 system (simplified approach)
    # For demonstration, use a simplified method
    if n >= 3:
        # Use three-point method for quadratic
        y0, y1, y2 = series[0], series[n//2], series[-1]
        x0, x1, x2 = 0, n//2, n-1
        
        # Solve for a, b, c in ax^2 + bx + c
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if abs(denom) > 1e-10:
            a = (x2*(y1 - y0) + x1*(y0 - y2) + x0*(y2 - y1)) / denom
            b = ((x2**2)*(y0 - y1) + (x1**2)*(y2 - y0) + (x0**2)*(y1 - y2)) / denom
            c = (x1*x2*(x1 - x2)*y0 + x2*x0*(x2 - x0)*y1 + x0*x1*(x0 - x1)*y2) / denom
        else:
            a, b, c = 0.0, 0.0, sum(series) / n
    else:
        a, b, c = 0.0, 0.0, sum(series) / n
    
    # Calculate R-squared
    y_pred = [a*i**2 + b*i + c for i in range(n)]
    y_mean = sum(series) / n
    ss_res = sum((series[i] - y_pred[i])**2 for i in range(n))
    ss_tot = sum((series[i] - y_mean)**2 for i in range(n))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    return {
        'coefficients': [c, b, a],  # [constant, linear, quadratic]
        'r_squared': r_squared,
        'degree': 2
    }


def _evaluate_polynomial(coefficients: List[float], x: float) -> float:
    """Evaluate polynomial at point x."""
    result = 0.0
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** i)
    return result


def _variance_changepoints(series: TimeSeries, min_length: int) -> List[int]:
    """Detect changepoints based on variance changes."""
    n = len(series)
    changepoints = []
    
    for i in range(min_length, n - min_length):
        left_segment = series[:i]
        right_segment = series[i:]
        
        if len(left_segment) >= 2 and len(right_segment) >= 2:
            left_var = statistics.variance(left_segment)
            right_var = statistics.variance(right_segment)
            
            # Simple variance ratio test
            var_ratio = max(left_var, right_var) / (min(left_var, right_var) + 1e-10)
            
            if var_ratio > 2.0:  # Threshold for significant change
                changepoints.append(i)
    
    return changepoints


def _mean_changepoints(series: TimeSeries, min_length: int) -> List[int]:
    """Detect changepoints based on mean changes."""
    n = len(series)
    changepoints = []
    
    overall_std = statistics.stdev(series) if len(series) > 1 else 1.0
    
    for i in range(min_length, n - min_length):
        left_mean = sum(series[:i]) / i
        right_mean = sum(series[i:]) / (n - i)
        
        # Test for significant mean difference
        mean_diff = abs(left_mean - right_mean)
        threshold = 1.5 * overall_std  # Conservative threshold
        
        if mean_diff > threshold:
            changepoints.append(i)
    
    return changepoints


def _trend_changepoints(series: TimeSeries, min_length: int) -> List[int]:
    """Detect changepoints based on trend changes."""
    n = len(series)
    changepoints = []
    
    for i in range(min_length, n - min_length):
        if i - min_length >= 0:
            left_slope = trend_slope(series[max(0, i-min_length):i])
            right_slope = trend_slope(series[i:min(n, i+min_length)])
            
            slope_change = abs(left_slope - right_slope)
            
            if slope_change > 0.05:  # Threshold for significant slope change
                changepoints.append(i)
    
    return changepoints


def _theil_sen_slope(series: TimeSeries) -> float:
    """Calculate Theil-Sen robust slope estimator."""
    n = len(series)
    if n < 2:
        return 0.0
    
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if j != i:
                slope = (series[j] - series[i]) / (j - i)
                slopes.append(slope)
    
    return statistics.median(slopes) if slopes else 0.0


def _percentile_slope(series: TimeSeries, percentile: float = 90.0) -> float:
    """Calculate slope between percentile points."""
    if len(series) < 2:
        return 0.0
    
    sorted_indices = sorted(range(len(series)), key=lambda i: series[i])
    low_idx = int(len(sorted_indices) * (100 - percentile) / 200)
    high_idx = int(len(sorted_indices) * (100 + percentile) / 200)
    
    low_idx = max(0, min(low_idx, len(sorted_indices) - 1))
    high_idx = max(0, min(high_idx, len(sorted_indices) - 1))
    
    if high_idx == low_idx:
        return 0.0
    
    x1, y1 = sorted_indices[low_idx], series[sorted_indices[low_idx]]
    x2, y2 = sorted_indices[high_idx], series[sorted_indices[high_idx]]
    
    return (y2 - y1) / (x2 - x1) if x2 != x1 else 0.0