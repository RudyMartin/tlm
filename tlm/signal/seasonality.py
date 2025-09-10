"""
Seasonality detection and analysis functions for TLM signal processing.

Provides comprehensive methods for detecting seasonal patterns, estimating periods,
and decomposing multiple seasonal components in time series data.
"""

from typing import List, Tuple, Optional, Dict, Union, NamedTuple
import math
from ..pure.ops import mean, median, std as stdev, var as variance

# Type definitions
TimeSeries = List[float]
SeasonalComponent = List[float]
SeasonalPeriods = List[int]

class SeasonalityResult(NamedTuple):
    """Result of seasonality detection."""
    is_seasonal: bool
    primary_period: Optional[int] 
    strength: float
    periods: List[Tuple[int, float]]  # (period, strength) pairs


def detect_seasonality(series: TimeSeries, max_period: Optional[int] = None,
                      min_period: int = 2, method: str = 'autocorr') -> SeasonalityResult:
    """
    Detect seasonality in time series data.
    
    Args:
        series: Input time series
        max_period: Maximum period to test (default: len(series)//2)
        min_period: Minimum period to test
        method: Detection method ('autocorr', 'fft', 'combined')
        
    Returns:
        SeasonalityResult with detection results
    """
    n = len(series)
    if n < 2 * min_period:
        return SeasonalityResult(False, None, 0.0, [])
    
    if max_period is None:
        max_period = min(n // 2, 50)  # Reasonable upper limit
    
    max_period = min(max_period, n // 2)
    
    if method == 'autocorr':
        return _detect_seasonality_autocorr(series, min_period, max_period)
    elif method == 'fft':
        return _detect_seasonality_fft(series, min_period, max_period)
    elif method == 'combined':
        return _detect_seasonality_combined(series, min_period, max_period)
    else:
        raise ValueError(f"Unknown method: {method}")


def seasonal_periods(series: TimeSeries, n_periods: int = 3, 
                    method: str = 'autocorr') -> List[Tuple[int, float]]:
    """
    Find multiple seasonal periods in descending order of strength.
    
    Args:
        series: Input time series
        n_periods: Number of periods to return
        method: Detection method
        
    Returns:
        List of (period, strength) tuples
    """
    result = detect_seasonality(series, method=method)
    periods = sorted(result.periods, key=lambda x: x[1], reverse=True)
    return periods[:n_periods]


def seasonal_strength_test(series: TimeSeries, period: int, 
                          method: str = 'f_test') -> Dict[str, float]:
    """
    Test strength of specific seasonal period.
    
    Args:
        series: Input time series
        period: Period to test
        method: Test method ('f_test', 'autocorr', 'kruskal')
        
    Returns:
        Dictionary with strength metrics
    """
    n = len(series)
    if period >= n // 2:
        return {'strength': 0.0, 'statistic': 0.0, 'p_value': 1.0}
    
    if method == 'f_test':
        return _seasonal_f_test(series, period)
    elif method == 'autocorr':
        from .spectral import autocorrelation_function
        acf = autocorrelation_function(series, period)
        strength = abs(acf[period]) if len(acf) > period else 0.0
        return {'strength': strength, 'statistic': strength, 'p_value': 0.05 if strength > 0.2 else 0.5}
    elif method == 'kruskal':
        return _seasonal_kruskal_test(series, period)
    else:
        raise ValueError(f"Unknown method: {method}")


def seasonal_autocorrelation(series: TimeSeries, max_lag: Optional[int] = None,
                           seasonal_lags: Optional[List[int]] = None) -> Dict[int, float]:
    """
    Calculate autocorrelation at seasonal lags.
    
    Args:
        series: Input time series
        max_lag: Maximum lag to consider
        seasonal_lags: Specific seasonal lags to test
        
    Returns:
        Dictionary mapping lags to autocorrelation values
    """
    from .spectral import autocorrelation_function
    
    if max_lag is None:
        max_lag = min(len(series) // 2, 50)
    
    acf = autocorrelation_function(series, max_lag)
    
    if seasonal_lags is None:
        # Common seasonal periods
        seasonal_lags = [4, 7, 12, 24, 52, 365]  # quarterly, weekly, monthly, etc.
        seasonal_lags = [lag for lag in seasonal_lags if lag < len(acf)]
    
    seasonal_acf = {}
    for lag in seasonal_lags:
        if lag < len(acf):
            seasonal_acf[lag] = acf[lag]
    
    return seasonal_acf


def seasonal_decomposition_strength(series: TimeSeries, period: int,
                                  method: str = 'stl') -> float:
    """
    Measure strength of seasonal decomposition.
    
    Args:
        series: Input time series
        period: Seasonal period
        method: Decomposition method
        
    Returns:
        Seasonal strength (0-1)
    """
    if period >= len(series) // 2:
        return 0.0
    
    # Extract seasonal component
    seasonal = _extract_seasonal_component(series, period, method)
    
    if not seasonal:
        return 0.0
    
    # Calculate strength as proportion of variance explained
    series_var = variance(series) if len(series) > 1 else 0.0
    
    if series_var == 0:
        return 0.0
    
    seasonal_var = variance(seasonal) if len(seasonal) > 1 else 0.0
    strength = seasonal_var / series_var
    
    return min(1.0, strength)


def multiple_seasonal_decompose(series: TimeSeries, periods: List[int],
                               method: str = 'additive') -> Dict[str, List[float]]:
    """
    Decompose series with multiple seasonal components.
    
    Args:
        series: Input time series
        periods: List of seasonal periods
        method: 'additive' or 'multiplicative'
        
    Returns:
        Dictionary with trend, seasonal components, and residual
    """
    n = len(series)
    result = {
        'trend': [0.0] * n,
        'residual': series[:]
    }
    
    # Add seasonal components
    for i, period in enumerate(periods):
        if period < n // 2:
            seasonal = _extract_seasonal_component(result['residual'], period, 'simple')
            
            # Store seasonal component
            result[f'seasonal_{period}'] = seasonal
            
            # Remove from residual
            if method == 'additive':
                result['residual'] = [result['residual'][j] - seasonal[j] for j in range(n)]
            else:  # multiplicative
                result['residual'] = [result['residual'][j] / (seasonal[j] + 1e-10) for j in range(n)]
    
    # Extract trend from residual
    from .trends import linear_trend
    trend_model = linear_trend(result['residual'])
    result['trend'] = [trend_model['intercept'] + trend_model['slope'] * i for i in range(n)]
    
    # Final residual
    if method == 'additive':
        result['residual'] = [result['residual'][i] - result['trend'][i] for i in range(n)]
    else:
        result['residual'] = [result['residual'][i] / (result['trend'][i] + 1e-10) for i in range(n)]
    
    return result


def hierarchical_seasonal_decompose(series: TimeSeries, 
                                   periods: List[int]) -> Dict[str, List[float]]:
    """
    Hierarchical seasonal decomposition (high to low frequency).
    
    Args:
        series: Input time series
        periods: Seasonal periods in decreasing order
        
    Returns:
        Dictionary with hierarchical seasonal components
    """
    n = len(series)
    result = {'original': series[:]}
    current = series[:]
    
    # Sort periods by decreasing length (lowest frequency first)
    sorted_periods = sorted([p for p in periods if p < n // 2], reverse=True)
    
    for period in sorted_periods:
        # Extract seasonal component from current series
        seasonal = _extract_seasonal_component(current, period, 'stl')
        result[f'seasonal_{period}'] = seasonal
        
        # Remove seasonal component
        current = [current[i] - seasonal[i] for i in range(n)]
    
    # Extract trend from remaining series
    from .trends import linear_trend
    trend_model = linear_trend(current)
    trend = [trend_model['intercept'] + trend_model['slope'] * i for i in range(n)]
    result['trend'] = trend
    
    # Final residual
    result['residual'] = [current[i] - trend[i] for i in range(n)]
    
    return result


def seasonal_naive_forecast(series: TimeSeries, period: int, 
                           horizon: int = 1) -> List[float]:
    """
    Simple seasonal naive forecasting.
    
    Args:
        series: Input time series
        period: Seasonal period
        horizon: Forecast horizon
        
    Returns:
        Forecasted values
    """
    n = len(series)
    if period >= n:
        # No seasonal pattern, use last value
        return [series[-1]] * horizon
    
    forecasts = []
    for h in range(horizon):
        # Use value from same season in previous cycle
        seasonal_index = (n + h) % period
        historical_index = n - period + seasonal_index
        
        if historical_index >= 0:
            forecasts.append(series[historical_index])
        else:
            forecasts.append(series[seasonal_index % n])
    
    return forecasts


def seasonal_difference(series: TimeSeries, period: int, 
                       order: int = 1) -> TimeSeries:
    """
    Apply seasonal differencing to remove seasonality.
    
    Args:
        series: Input time series
        period: Seasonal period
        order: Order of differencing
        
    Returns:
        Seasonally differenced series
    """
    current = series[:]
    
    for _ in range(order):
        if len(current) <= period:
            break
        
        differenced = []
        for i in range(period, len(current)):
            differenced.append(current[i] - current[i - period])
        
        current = differenced
    
    return current


# Helper functions

def _detect_seasonality_autocorr(series: TimeSeries, min_period: int, 
                                max_period: int) -> SeasonalityResult:
    """Detect seasonality using autocorrelation."""
    from .spectral import autocorrelation_function
    
    acf = autocorrelation_function(series, max_period)
    
    periods_strengths = []
    for period in range(min_period, min(len(acf), max_period + 1)):
        strength = abs(acf[period])
        periods_strengths.append((period, strength))
    
    if not periods_strengths:
        return SeasonalityResult(False, None, 0.0, [])
    
    # Find periods with significant autocorrelation
    significant_periods = [(p, s) for p, s in periods_strengths if s > 0.2]
    
    if significant_periods:
        # Primary period is the one with highest autocorrelation
        primary_period, primary_strength = max(significant_periods, key=lambda x: x[1])
        is_seasonal = primary_strength > 0.3
    else:
        primary_period, primary_strength = max(periods_strengths, key=lambda x: x[1])
        is_seasonal = False
    
    return SeasonalityResult(
        is_seasonal=is_seasonal,
        primary_period=primary_period,
        strength=primary_strength,
        periods=significant_periods if significant_periods else periods_strengths[:5]
    )


def _detect_seasonality_fft(series: TimeSeries, min_period: int, 
                           max_period: int) -> SeasonalityResult:
    """Detect seasonality using FFT."""
    from .spectral import periodogram
    
    try:
        frequencies, psd = periodogram(series)
        
        periods_strengths = []
        for i, freq in enumerate(frequencies[1:], 1):  # Skip DC component
            if freq > 0:
                period = 1.0 / freq
                if min_period <= period <= max_period:
                    periods_strengths.append((int(round(period)), psd[i]))
        
        if not periods_strengths:
            return SeasonalityResult(False, None, 0.0, [])
        
        # Sort by power and find significant peaks
        periods_strengths.sort(key=lambda x: x[1], reverse=True)
        
        # Primary period
        primary_period, primary_power = periods_strengths[0]
        
        # Check if significant
        total_power = sum(psd)
        primary_strength = primary_power / total_power if total_power > 0 else 0.0
        is_seasonal = primary_strength > 0.1
        
        return SeasonalityResult(
            is_seasonal=is_seasonal,
            primary_period=primary_period,
            strength=primary_strength,
            periods=periods_strengths[:5]
        )
    
    except:
        # Fallback to autocorrelation
        return _detect_seasonality_autocorr(series, min_period, max_period)


def _detect_seasonality_combined(series: TimeSeries, min_period: int, 
                                max_period: int) -> SeasonalityResult:
    """Combine autocorrelation and FFT methods."""
    acf_result = _detect_seasonality_autocorr(series, min_period, max_period)
    fft_result = _detect_seasonality_fft(series, min_period, max_period)
    
    # Combine results by averaging strengths for matching periods
    combined_periods = {}
    
    for period, strength in acf_result.periods:
        combined_periods[period] = strength
    
    for period, strength in fft_result.periods:
        if period in combined_periods:
            combined_periods[period] = (combined_periods[period] + strength) / 2
        else:
            combined_periods[period] = strength * 0.5  # Lower weight for FFT-only
    
    sorted_periods = sorted(combined_periods.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_periods:
        primary_period, primary_strength = sorted_periods[0]
        is_seasonal = primary_strength > 0.25
    else:
        primary_period, primary_strength = None, 0.0
        is_seasonal = False
    
    return SeasonalityResult(
        is_seasonal=is_seasonal,
        primary_period=primary_period,
        strength=primary_strength,
        periods=sorted_periods[:5]
    )


def _seasonal_f_test(series: TimeSeries, period: int) -> Dict[str, float]:
    """F-test for seasonal effects."""
    n = len(series)
    if period >= n:
        return {'strength': 0.0, 'statistic': 0.0, 'p_value': 1.0}
    
    # Group data by seasonal periods
    seasonal_groups = [[] for _ in range(period)]
    for i, val in enumerate(series):
        seasonal_groups[i % period].append(val)
    
    # Remove empty groups
    seasonal_groups = [group for group in seasonal_groups if group]
    
    if len(seasonal_groups) < 2:
        return {'strength': 0.0, 'statistic': 0.0, 'p_value': 1.0}
    
    # Calculate F-statistic
    group_means = [mean(group) for group in seasonal_groups]
    overall_mean = mean(series)
    
    # Between-group sum of squares
    ss_between = sum(len(group) * (mean - overall_mean)**2 
                    for group, mean in zip(seasonal_groups, group_means))
    
    # Within-group sum of squares  
    ss_within = sum(sum((val - mean(group))**2 for val in group)
                   for group in seasonal_groups)
    
    df_between = len(seasonal_groups) - 1
    df_within = n - len(seasonal_groups)
    
    if df_within <= 0 or ss_within == 0:
        return {'strength': 0.0, 'statistic': 0.0, 'p_value': 1.0}
    
    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within
    
    f_stat = ms_between / ms_within if ms_within > 0 else 0.0
    
    # Approximate p-value (very rough)
    p_value = 0.01 if f_stat > 5.0 else (0.05 if f_stat > 2.0 else 0.5)
    
    # Strength as effect size
    eta_squared = ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else 0.0
    
    return {
        'strength': eta_squared,
        'statistic': f_stat,
        'p_value': p_value
    }


def _seasonal_kruskal_test(series: TimeSeries, period: int) -> Dict[str, float]:
    """Kruskal-Wallis test for seasonal effects (non-parametric)."""
    n = len(series)
    if period >= n:
        return {'strength': 0.0, 'statistic': 0.0, 'p_value': 1.0}
    
    # Group data by seasonal periods
    seasonal_groups = [[] for _ in range(period)]
    for i, val in enumerate(series):
        seasonal_groups[i % period].append(val)
    
    # Remove empty groups
    seasonal_groups = [group for group in seasonal_groups if group]
    
    if len(seasonal_groups) < 2:
        return {'strength': 0.0, 'statistic': 0.0, 'p_value': 1.0}
    
    # Rank all values
    all_values = [(val, i % period) for i, val in enumerate(series)]
    all_values.sort(key=lambda x: x[0])
    
    # Assign ranks (handling ties by averaging)
    ranks = {}
    for rank, (val, group) in enumerate(all_values, 1):
        if val not in ranks:
            ranks[val] = []
        ranks[val].append((rank, group))
    
    # Calculate average ranks for ties
    for val in ranks:
        tie_ranks = [r[0] for r in ranks[val]]
        avg_rank = sum(tie_ranks) / len(tie_ranks)
        for i in range(len(ranks[val])):
            ranks[val][i] = (avg_rank, ranks[val][i][1])
    
    # Sum ranks for each group
    group_rank_sums = [0.0] * period
    group_sizes = [0] * period
    
    for val in ranks:
        for avg_rank, group in ranks[val]:
            group_rank_sums[group] += avg_rank
            group_sizes[group] += 1
    
    # Calculate H statistic
    h_stat = 0.0
    for i in range(period):
        if group_sizes[i] > 0:
            mean_rank = group_rank_sums[i] / group_sizes[i]
            overall_mean_rank = (n + 1) / 2.0
            h_stat += group_sizes[i] * (mean_rank - overall_mean_rank)**2
    
    h_stat *= 12.0 / (n * (n + 1))
    
    # Approximate p-value
    p_value = 0.01 if h_stat > 7.815 else (0.05 if h_stat > 5.991 else 0.5)
    
    # Strength measure
    strength = min(1.0, h_stat / 10.0)
    
    return {
        'strength': strength,
        'statistic': h_stat,
        'p_value': p_value
    }


def _extract_seasonal_component(series: TimeSeries, period: int, 
                               method: str = 'simple') -> List[float]:
    """Extract seasonal component using specified method."""
    n = len(series)
    
    if method == 'simple':
        # Simple seasonal averages
        seasonal_sums = [0.0] * period
        seasonal_counts = [0] * period
        
        for i, val in enumerate(series):
            seasonal_sums[i % period] += val
            seasonal_counts[i % period] += 1
        
        # Calculate seasonal averages
        seasonal_means = []
        for i in range(period):
            if seasonal_counts[i] > 0:
                seasonal_means.append(seasonal_sums[i] / seasonal_counts[i])
            else:
                seasonal_means.append(0.0)
        
        # Adjust to sum to zero
        overall_mean = sum(seasonal_means) / len(seasonal_means)
        seasonal_means = [s - overall_mean for s in seasonal_means]
        
        # Expand to full series
        seasonal_component = [seasonal_means[i % period] for i in range(n)]
        
    elif method == 'stl':
        # Simplified STL-like approach
        seasonal_component = _simple_stl_seasonal(series, period)
        
    else:
        # Fallback to simple
        seasonal_component = _extract_seasonal_component(series, period, 'simple')
    
    return seasonal_component


def _simple_stl_seasonal(series: TimeSeries, period: int) -> List[float]:
    """Simplified STL seasonal extraction."""
    n = len(series)
    
    # Start with simple seasonal means
    seasonal_component = _extract_seasonal_component(series, period, 'simple')
    
    # Iterate to refine (simplified)
    for iteration in range(3):
        # Remove current seasonal estimate
        deseasonalized = [series[i] - seasonal_component[i] for i in range(n)]
        
        # Smooth the deseasonalized series (trend)
        from .filtering import moving_average_filter
        trend = moving_average_filter(deseasonalized, min(period, n // 4))
        
        # Remove trend to get seasonal + noise
        detrended = [series[i] - trend[i] for i in range(n)]
        
        # Update seasonal component
        seasonal_sums = [0.0] * period
        seasonal_counts = [0] * period
        
        for i, val in enumerate(detrended):
            seasonal_sums[i % period] += val
            seasonal_counts[i % period] += 1
        
        seasonal_means = []
        for i in range(period):
            if seasonal_counts[i] > 0:
                seasonal_means.append(seasonal_sums[i] / seasonal_counts[i])
            else:
                seasonal_means.append(0.0)
        
        # Adjust to sum to zero
        overall_mean = sum(seasonal_means) / len(seasonal_means)
        seasonal_means = [s - overall_mean for s in seasonal_means]
        
        seasonal_component = [seasonal_means[i % period] for i in range(n)]
    
    return seasonal_component