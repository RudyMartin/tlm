"""
Signal filtering and noise analysis functions for TLM.

Provides functions for noise detection, filtering, outlier detection,
and signal enhancement - essential for cleaning and preprocessing
time series data before analysis.
"""

from typing import List, Tuple, Optional, Dict
import math
import statistics

# Type definitions
TimeSeries = List[float]

def white_noise_test(series: TimeSeries, max_lags: int = 20) -> Dict[str, float]:
    """
    Test if series is white noise using Ljung-Box test.
    
    Args:
        series: Input time series
        max_lags: Maximum lags to test
    
    Returns:
        Dictionary with test statistic and p-value approximation
    """
    from .spectral import autocorrelation_function
    
    n = len(series)
    max_lags = min(max_lags, n // 4)
    
    # Get autocorrelations
    acf = autocorrelation_function(series, max_lags)
    
    # Ljung-Box statistic
    lb_stat = 0.0
    for k in range(1, max_lags + 1):
        if k < len(acf):
            lb_stat += (acf[k] ** 2) / (n - k)
    
    lb_stat *= n * (n + 2)
    
    # Approximate p-value using chi-square distribution
    # This is a simplified approximation
    degrees_freedom = max_lags
    p_value = _chi_square_survival(lb_stat, degrees_freedom)
    
    return {
        'statistic': lb_stat,
        'p_value': p_value,
        'is_white_noise': p_value > 0.05  # 5% significance level
    }


def noise_variance_estimation(series: TimeSeries, method: str = 'mad') -> float:
    """
    Estimate noise variance in the series.
    
    Args:
        series: Input time series
        method: 'mad' (median absolute deviation) or 'diff' (differencing)
    
    Returns:
        Estimated noise variance
    """
    if method == 'mad':
        return _mad_variance_estimate(series)
    elif method == 'diff':
        return _diff_variance_estimate(series)
    else:
        raise ValueError(f"Unknown method: {method}")


def signal_to_noise_ratio(series: TimeSeries, signal_estimate: Optional[TimeSeries] = None) -> float:
    """
    Calculate signal-to-noise ratio.
    
    Args:
        series: Input time series
        signal_estimate: Estimated signal component (if None, use smoothed version)
    
    Returns:
        Signal-to-noise ratio in dB
    """
    if signal_estimate is None:
        # Use moving average as signal estimate
        signal_estimate = moving_average_filter(series, window=5)
    
    # Ensure same length
    min_len = min(len(series), len(signal_estimate))
    series = series[:min_len]
    signal_estimate = signal_estimate[:min_len]
    
    # Calculate noise as residual
    noise = [series[i] - signal_estimate[i] for i in range(min_len)]
    
    # Calculate power
    signal_power = sum(s ** 2 for s in signal_estimate) / len(signal_estimate)
    noise_power = sum(n ** 2 for n in noise) / len(noise)
    
    if noise_power == 0:
        return float('inf')
    
    snr_ratio = signal_power / noise_power
    return 10 * math.log10(snr_ratio) if snr_ratio > 0 else float('-inf')


def estimate_noise_level(series: TimeSeries, method: str = 'robust') -> float:
    """
    Estimate noise level in the series.
    
    Args:
        series: Input time series  
        method: 'robust', 'std', or 'iqr'
    
    Returns:
        Estimated noise level (standard deviation)
    """
    if method == 'robust':
        return _robust_noise_estimate(series)
    elif method == 'std':
        # Use differenced series to estimate noise
        diffs = [series[i+1] - series[i] for i in range(len(series) - 1)]
        return statistics.stdev(diffs) / math.sqrt(2) if len(diffs) > 1 else 0.0
    elif method == 'iqr':
        return _iqr_noise_estimate(series)
    else:
        raise ValueError(f"Unknown method: {method}")


def moving_average_filter(series: TimeSeries, window: int, mode: str = 'centered') -> TimeSeries:
    """
    Apply moving average filter for smoothing.
    
    Args:
        series: Input time series
        window: Window size
        mode: 'centered', 'forward', or 'backward'
    
    Returns:
        Filtered time series
    """
    n = len(series)
    result = []
    
    if mode == 'centered':
        k = window // 2
        for i in range(n):
            start = max(0, i - k)
            end = min(n, i + k + 1)
            avg = sum(series[start:end]) / (end - start)
            result.append(avg)
    
    elif mode == 'forward':
        for i in range(n):
            end = min(n, i + window)
            avg = sum(series[i:end]) / (end - i)
            result.append(avg)
    
    elif mode == 'backward':
        for i in range(n):
            start = max(0, i - window + 1)
            avg = sum(series[start:i + 1]) / (i - start + 1)
            result.append(avg)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return result


def exponential_smoothing(series: TimeSeries, alpha: float = 0.3) -> TimeSeries:
    """
    Apply exponential smoothing filter.
    
    Args:
        series: Input time series
        alpha: Smoothing parameter (0 < alpha < 1)
    
    Returns:
        Exponentially smoothed series
    """
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    result = []
    s = series[0]  # Initialize with first value
    
    for x in series:
        s = alpha * x + (1 - alpha) * s
        result.append(s)
    
    return result


def savitzky_golay_filter(series: TimeSeries, window_length: int = 5, 
                         poly_order: int = 2) -> TimeSeries:
    """
    Apply Savitzky-Golay smoothing filter.
    
    Fits local polynomials to smooth the data while preserving features.
    
    Args:
        series: Input time series
        window_length: Length of filter window (must be odd)
        poly_order: Order of polynomial fit
    
    Returns:
        Smoothed time series
    """
    if window_length % 2 == 0:
        window_length += 1  # Make odd
    
    if poly_order >= window_length:
        raise ValueError("poly_order must be less than window_length")
    
    n = len(series)
    result = []
    half_window = window_length // 2
    
    # Generate Savitzky-Golay coefficients
    coeffs = _savgol_coeffs(window_length, poly_order)
    
    for i in range(n):
        # Determine window bounds
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        
        # Adjust coefficients for boundary conditions
        if start == 0 or end == n:
            # Use simple polynomial fit for boundaries
            window_data = series[start:end]
            smoothed_val = _local_polynomial_smooth(window_data, poly_order, 
                                                  i - start if start == 0 else half_window)
        else:
            # Use pre-computed coefficients
            window_data = series[start:end]
            smoothed_val = sum(c * d for c, d in zip(coeffs, window_data))
        
        result.append(smoothed_val)
    
    return result


def median_filter(series: TimeSeries, window: int = 3) -> TimeSeries:
    """
    Apply median filter for removing impulse noise.
    
    Args:
        series: Input time series
        window: Window size (should be odd)
    
    Returns:
        Median filtered series
    """
    if window % 2 == 0:
        window += 1  # Make odd
    
    n = len(series)
    result = []
    k = window // 2
    
    for i in range(n):
        start = max(0, i - k)
        end = min(n, i + k + 1)
        window_data = series[start:end]
        result.append(statistics.median(window_data))
    
    return result


def outlier_detection_seasonal(series: TimeSeries, period: int, 
                             threshold: float = 3.0) -> List[bool]:
    """
    Detect outliers accounting for seasonal patterns.
    
    Args:
        series: Input time series
        period: Seasonal period
        threshold: Number of standard deviations for outlier threshold
    
    Returns:
        Boolean list indicating outliers
    """
    from .decomposition import additive_decompose
    
    # Decompose series
    try:
        decomp = additive_decompose(series, period)
        residuals = [r for r in decomp.remainder if r is not None]
    except:
        # Fallback to simple residuals
        residuals = series
    
    # Calculate robust statistics
    median_res = statistics.median(residuals)
    mad = statistics.median([abs(r - median_res) for r in residuals])
    
    # Modified z-score using MAD
    mad_scaled = mad * 1.4826  # Scale factor for normal distribution
    
    outliers = []
    for i, val in enumerate(series):
        if i < len(decomp.remainder) and decomp.remainder[i] is not None:
            residual = decomp.remainder[i]
            if mad_scaled > 0:
                modified_z = abs(residual - median_res) / mad_scaled
                is_outlier = modified_z > threshold
            else:
                is_outlier = False
        else:
            is_outlier = False
        
        outliers.append(is_outlier)
    
    return outliers


def outlier_detection_spectral(series: TimeSeries, threshold: float = 2.0) -> List[bool]:
    """
    Detect outliers using spectral analysis.
    
    Args:
        series: Input time series
        threshold: Threshold for spectral outlier detection
    
    Returns:
        Boolean list indicating outliers
    """
    from .spectral import periodogram
    
    # Compute power spectral density
    try:
        frequencies, psd = periodogram(series)
        
        # Find dominant frequency
        max_power_idx = psd.index(max(psd))
        dominant_freq = frequencies[max_power_idx]
        
        # Reconstruct signal using dominant frequency
        n = len(series)
        reconstructed = []
        for i in range(n):
            # Simple sinusoidal reconstruction
            val = sum(math.sin(2 * math.pi * dominant_freq * i + phase) 
                     for phase in [0, math.pi/2])  # Two phase components
            reconstructed.append(val)
        
        # Calculate residuals
        residuals = [series[i] - reconstructed[i] for i in range(n)]
        
    except:
        # Fallback to simple differencing
        residuals = [series[i+1] - series[i] for i in range(len(series) - 1)]
        residuals.append(0.0)  # Pad to same length
    
    # Detect outliers in residuals
    if len(residuals) > 0:
        std_res = statistics.stdev(residuals) if len(residuals) > 1 else 0
        mean_res = statistics.mean(residuals)
        
        outliers = [abs(r - mean_res) > threshold * std_res for r in residuals]
    else:
        outliers = [False] * len(series)
    
    return outliers


def adaptive_filter(series: TimeSeries, mu: float = 0.01, 
                   filter_length: int = 5) -> TimeSeries:
    """
    Apply adaptive filtering (LMS algorithm).
    
    Args:
        series: Input time series
        mu: Learning rate
        filter_length: Length of adaptive filter
    
    Returns:
        Filtered series
    """
    n = len(series)
    weights = [0.0] * filter_length
    output = []
    
    for i in range(n):
        # Get input vector
        x = []
        for j in range(filter_length):
            if i - j >= 0:
                x.append(series[i - j])
            else:
                x.append(0.0)
        
        # Filter output
        y = sum(w * xi for w, xi in zip(weights, x))
        output.append(y)
        
        # Error signal (assuming desired signal is delayed input)
        d = series[i - 1] if i > 0 else series[i]
        error = d - y
        
        # Update weights (LMS algorithm)
        for j in range(filter_length):
            weights[j] += mu * error * x[j]
    
    return output


# Helper functions

def _mad_variance_estimate(series: TimeSeries) -> float:
    """Estimate variance using median absolute deviation."""
    median_val = statistics.median(series)
    mad = statistics.median([abs(x - median_val) for x in series])
    # Convert MAD to standard deviation estimate
    return (mad * 1.4826) ** 2


def _diff_variance_estimate(series: TimeSeries) -> float:
    """Estimate variance using first differences."""
    diffs = [series[i+1] - series[i] for i in range(len(series) - 1)]
    if len(diffs) <= 1:
        return 0.0
    return statistics.variance(diffs) / 2  # Divide by 2 for white noise


def _robust_noise_estimate(series: TimeSeries) -> float:
    """Robust noise estimation using wavelet-based method."""
    # Use high-pass filtering to isolate noise
    n = len(series)
    if n <= 1:
        return 0.0
    
    # Simple high-pass filter (difference operator)
    high_freq = [series[i+1] - series[i] for i in range(n - 1)]
    
    # Robust scale estimate
    if len(high_freq) == 0:
        return 0.0
    
    mad = statistics.median([abs(x) for x in high_freq])
    return mad * 1.4826 / math.sqrt(2)  # Scale for white noise


def _iqr_noise_estimate(series: TimeSeries) -> float:
    """Estimate noise using interquartile range."""
    sorted_series = sorted(series)
    n = len(sorted_series)
    
    if n <= 1:
        return 0.0
    
    q1 = sorted_series[n // 4]
    q3 = sorted_series[3 * n // 4]
    iqr = q3 - q1
    
    # Convert IQR to standard deviation estimate
    return iqr / 1.35  # For normal distribution


def _chi_square_survival(x: float, df: int) -> float:
    """Approximate chi-square survival function (1 - CDF)."""
    # Very rough approximation for p-value
    if df <= 0:
        return 1.0
    
    # Use normal approximation for large df
    if df > 30:
        z = (x - df) / math.sqrt(2 * df)
        return 0.5 * (1 - math.erf(z / math.sqrt(2)))
    
    # Simple approximation for small df
    if x <= 0:
        return 1.0
    elif x > 20:
        return 0.0
    else:
        # Linear interpolation (very rough)
        return max(0.0, min(1.0, 1.0 - x / (2 * df)))


def _savgol_coeffs(window_length: int, poly_order: int) -> List[float]:
    """Generate Savitzky-Golay filter coefficients."""
    # This is a simplified implementation
    # Full implementation would use matrix operations
    half_window = window_length // 2
    
    # For second-order polynomial (most common case)
    if poly_order == 2:
        # Pre-computed coefficients for common window sizes
        if window_length == 5:
            return [-0.086, 0.343, 0.486, 0.343, -0.086]
        elif window_length == 7:
            return [-0.095, 0.143, 0.286, 0.333, 0.286, 0.143, -0.095]
    
    # Fallback to simple moving average
    return [1.0 / window_length] * window_length


def _local_polynomial_smooth(data: List[float], degree: int, pos: int) -> float:
    """Fit local polynomial and evaluate at position."""
    # Simplified polynomial fitting
    n = len(data)
    if n <= 1:
        return data[0] if data else 0.0
    
    if degree == 0:
        return statistics.mean(data)
    elif degree == 1:
        # Linear fit
        x = list(range(n))
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(data)
        
        numerator = sum((x[i] - x_mean) * (data[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return y_mean
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        return intercept + slope * pos
    else:
        # Higher order - fallback to mean
        return statistics.mean(data)