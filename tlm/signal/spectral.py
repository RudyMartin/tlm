"""
Spectral analysis functions for TLM signal processing.

Provides Fourier-based frequency domain analysis for detecting
periodicities, dominant frequencies, and spectral characteristics
of time series data.
"""

from typing import List, Tuple, Optional, Dict
import math
import cmath

# Type definitions  
TimeSeries = List[float]
ComplexSeries = List[complex]
FrequencySeries = List[float]

def fft_decompose(series: TimeSeries) -> Tuple[ComplexSeries, FrequencySeries]:
    """
    Discrete Fourier Transform using pure Python FFT implementation.
    
    Args:
        series: Input time series
    
    Returns:
        Tuple of (complex_frequencies, frequency_bins)
    """
    n = len(series)
    
    # Pad to nearest power of 2 for efficiency
    padded_n = _next_power_of_2(n)
    padded_series = series + [0.0] * (padded_n - n)
    
    # Compute FFT
    fft_result = _fft(padded_series)
    
    # Generate frequency bins
    freq_bins = [i / padded_n for i in range(padded_n)]
    
    return fft_result, freq_bins


def periodogram(series: TimeSeries, window: Optional[str] = None) -> Tuple[FrequencySeries, FrequencySeries]:
    """
    Compute periodogram (power spectral density estimate).
    
    Args:
        series: Input time series
        window: Window function ('hanning', 'hamming', 'blackman', None)
    
    Returns:
        Tuple of (frequencies, power_spectral_density)
    """
    n = len(series)
    
    # Apply window function
    if window:
        windowed = _apply_window(series, window)
        window_correction = _window_correction_factor(window, n)
    else:
        windowed = series
        window_correction = 1.0
    
    # Compute FFT
    fft_result, freq_bins = fft_decompose(windowed)
    
    # Calculate power spectral density
    psd = []
    for i in range(n // 2 + 1):  # Only positive frequencies
        magnitude_squared = abs(fft_result[i]) ** 2
        # Scale by window correction and normalize
        psd_val = magnitude_squared * window_correction / n
        psd.append(psd_val)
    
    # Positive frequencies only
    frequencies = freq_bins[:len(psd)]
    
    return frequencies, psd


def power_spectral_density(series: TimeSeries, method: str = 'periodogram', 
                          window: str = 'hanning', overlap: float = 0.5,
                          nperseg: Optional[int] = None) -> Tuple[FrequencySeries, FrequencySeries]:
    """
    Estimate power spectral density using various methods.
    
    Args:
        series: Input time series
        method: 'periodogram', 'welch', 'multitaper'
        window: Window function name
        overlap: Overlap fraction for Welch method
        nperseg: Segment length for Welch method
    
    Returns:
        Tuple of (frequencies, psd)
    """
    if method == 'periodogram':
        return periodogram(series, window)
    elif method == 'welch':
        return _welch_psd(series, window, overlap, nperseg)
    else:
        raise ValueError(f"Unknown method: {method}")


def autocorrelation_function(series: TimeSeries, max_lags: Optional[int] = None) -> List[float]:
    """
    Compute sample autocorrelation function.
    
    Args:
        series: Input time series
        max_lags: Maximum number of lags (default: n-1)
    
    Returns:
        Autocorrelation values for lags 0, 1, 2, ..., max_lags
    """
    n = len(series)
    if max_lags is None:
        max_lags = n - 1
    
    max_lags = min(max_lags, n - 1)
    
    # Center the series
    mean_val = sum(series) / n
    centered = [x - mean_val for x in series]
    
    # Compute autocorrelations
    autocorrs = []
    
    # Lag 0 (always 1.0)
    c0 = sum(x * x for x in centered) / n
    autocorrs.append(1.0)
    
    # Other lags
    for lag in range(1, max_lags + 1):
        ck = sum(centered[i] * centered[i + lag] for i in range(n - lag)) / n
        autocorr = ck / c0 if c0 != 0 else 0.0
        autocorrs.append(autocorr)
    
    return autocorrs


def partial_autocorrelation_function(series: TimeSeries, max_lags: Optional[int] = None) -> List[float]:
    """
    Compute partial autocorrelation function using Yule-Walker equations.
    
    Args:
        series: Input time series
        max_lags: Maximum number of lags
    
    Returns:
        Partial autocorrelation values
    """
    if max_lags is None:
        max_lags = min(len(series) // 4, 40)
    
    # Get autocorrelation function
    acf = autocorrelation_function(series, max_lags)
    
    # Compute partial autocorrelations using Durbin-Levinson recursion
    pacf = [1.0]  # PACF at lag 0 is always 1
    
    if max_lags == 0:
        return pacf
    
    # Initialize
    phi = [[0.0] * (max_lags + 1) for _ in range(max_lags + 1)]
    
    for k in range(1, max_lags + 1):
        # Calculate phi_kk (partial autocorrelation at lag k)
        numerator = acf[k] - sum(phi[k-1][j] * acf[k-j] for j in range(1, k))
        denominator = 1 - sum(phi[k-1][j] * acf[j] for j in range(1, k))
        
        phi_kk = numerator / denominator if abs(denominator) > 1e-10 else 0.0
        phi[k][k] = phi_kk
        pacf.append(phi_kk)
        
        # Update other coefficients
        for j in range(1, k):
            phi[k][j] = phi[k-1][j] - phi_kk * phi[k-1][k-j]
    
    return pacf


def dominant_frequencies(series: TimeSeries, n_peaks: int = 5) -> List[Tuple[float, float]]:
    """
    Find dominant frequencies in the spectrum.
    
    Args:
        series: Input time series
        n_peaks: Number of dominant peaks to return
    
    Returns:
        List of (frequency, power) tuples, sorted by power
    """
    frequencies, psd = periodogram(series)
    
    # Find peaks
    peaks = []
    for i in range(1, len(psd) - 1):
        if psd[i] > psd[i-1] and psd[i] > psd[i+1]:
            peaks.append((frequencies[i], psd[i]))
    
    # Sort by power and return top n
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks[:n_peaks]


def spectral_centroid(series: TimeSeries) -> float:
    """
    Calculate spectral centroid (center of mass of spectrum).
    
    Args:
        series: Input time series
    
    Returns:
        Spectral centroid frequency
    """
    frequencies, psd = periodogram(series)
    
    # Calculate weighted average frequency
    total_power = sum(psd)
    if total_power == 0:
        return 0.0
    
    centroid = sum(f * p for f, p in zip(frequencies, psd)) / total_power
    return centroid


def spectral_bandwidth(series: TimeSeries) -> float:
    """
    Calculate spectral bandwidth (spread around centroid).
    
    Args:
        series: Input time series
    
    Returns:
        Spectral bandwidth
    """
    frequencies, psd = periodogram(series)
    centroid = spectral_centroid(series)
    
    total_power = sum(psd)
    if total_power == 0:
        return 0.0
    
    # Calculate weighted variance around centroid
    bandwidth_sq = sum(((f - centroid) ** 2) * p for f, p in zip(frequencies, psd)) / total_power
    return math.sqrt(bandwidth_sq)


def spectral_rolloff(series: TimeSeries, rolloff_percent: float = 85.0) -> float:
    """
    Calculate spectral rolloff frequency.
    
    Args:
        series: Input time series
        rolloff_percent: Percentage of total power (default 85%)
    
    Returns:
        Frequency below which rolloff_percent of power is contained
    """
    frequencies, psd = periodogram(series)
    
    total_power = sum(psd)
    if total_power == 0:
        return 0.0
    
    threshold = total_power * (rolloff_percent / 100.0)
    cumulative_power = 0.0
    
    for i, power in enumerate(psd):
        cumulative_power += power
        if cumulative_power >= threshold:
            return frequencies[i]
    
    return frequencies[-1] if frequencies else 0.0


def spectral_flatness(series: TimeSeries) -> float:
    """
    Calculate spectral flatness (Wiener entropy).
    
    Measures how noise-like vs tonal a signal is.
    
    Args:
        series: Input time series
    
    Returns:
        Spectral flatness between 0 and 1
    """
    _, psd = periodogram(series)
    
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    psd_safe = [max(p, epsilon) for p in psd]
    
    # Geometric mean
    log_sum = sum(math.log(p) for p in psd_safe)
    geometric_mean = math.exp(log_sum / len(psd_safe))
    
    # Arithmetic mean
    arithmetic_mean = sum(psd_safe) / len(psd_safe)
    
    if arithmetic_mean == 0:
        return 0.0
    
    return geometric_mean / arithmetic_mean


def cross_spectrum(series1: TimeSeries, series2: TimeSeries) -> Tuple[FrequencySeries, ComplexSeries]:
    """
    Compute cross-power spectral density between two series.
    
    Args:
        series1: First time series
        series2: Second time series
    
    Returns:
        Tuple of (frequencies, cross_spectrum)
    """
    # Make series same length
    min_len = min(len(series1), len(series2))
    s1 = series1[:min_len]
    s2 = series2[:min_len]
    
    # Compute FFTs
    fft1, frequencies = fft_decompose(s1)
    fft2, _ = fft_decompose(s2)
    
    # Cross spectrum = FFT1 * conjugate(FFT2)
    cross_spec = [fft1[i] * fft2[i].conjugate() for i in range(len(fft1))]
    
    return frequencies, cross_spec


def coherence(series1: TimeSeries, series2: TimeSeries) -> Tuple[FrequencySeries, FrequencySeries]:
    """
    Compute magnitude squared coherence between two series.
    
    Args:
        series1: First time series
        series2: Second time series
    
    Returns:
        Tuple of (frequencies, coherence)
    """
    # Cross spectrum
    frequencies, cross_spec = cross_spectrum(series1, series2)
    
    # Auto spectra
    _, psd1 = periodogram(series1)
    _, psd2 = periodogram(series2)
    
    # Make same length as cross spectrum
    min_len = min(len(cross_spec), len(psd1), len(psd2))
    cross_spec = cross_spec[:min_len]
    psd1 = psd1[:min_len]
    psd2 = psd2[:min_len]
    frequencies = frequencies[:min_len]
    
    # Coherence = |cross_spectrum|^2 / (psd1 * psd2)
    coherence_vals = []
    for i in range(min_len):
        if psd1[i] * psd2[i] > 0:
            coh = (abs(cross_spec[i]) ** 2) / (psd1[i] * psd2[i])
        else:
            coh = 0.0
        coherence_vals.append(coh)
    
    return frequencies, coherence_vals


# Helper functions

def _next_power_of_2(n: int) -> int:
    """Find next power of 2 greater than or equal to n."""
    return 2 ** math.ceil(math.log2(n)) if n > 1 else 1


def _fft(x: List[float]) -> ComplexSeries:
    """
    Cooley-Tukey FFT algorithm (radix-2).
    
    Args:
        x: Input sequence (length must be power of 2)
    
    Returns:
        Complex FFT result
    """
    n = len(x)
    
    if n <= 1:
        return [complex(val) for val in x]
    
    # Divide
    even = _fft([x[i] for i in range(0, n, 2)])
    odd = _fft([x[i] for i in range(1, n, 2)])
    
    # Conquer
    result = [0] * n
    for k in range(n // 2):
        t = cmath.exp(-2j * cmath.pi * k / n) * odd[k]
        result[k] = even[k] + t
        result[k + n // 2] = even[k] - t
    
    return result


def _apply_window(series: TimeSeries, window_type: str) -> TimeSeries:
    """Apply window function to series."""
    n = len(series)
    
    if window_type == 'hanning':
        window = [0.5 * (1 - math.cos(2 * math.pi * i / (n - 1))) for i in range(n)]
    elif window_type == 'hamming':
        window = [0.54 - 0.46 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]
    elif window_type == 'blackman':
        window = [0.42 - 0.5 * math.cos(2 * math.pi * i / (n - 1)) + 
                 0.08 * math.cos(4 * math.pi * i / (n - 1)) for i in range(n)]
    else:
        raise ValueError(f"Unknown window type: {window_type}")
    
    return [series[i] * window[i] for i in range(n)]


def _window_correction_factor(window_type: str, n: int) -> float:
    """Calculate correction factor for window function."""
    if window_type == 'hanning':
        return 8.0 / 3.0
    elif window_type == 'hamming':
        return 25.0 / 9.0
    elif window_type == 'blackman':
        return 3.0 / 0.42
    else:
        return 1.0


def _welch_psd(series: TimeSeries, window: str, overlap: float, 
              nperseg: Optional[int]) -> Tuple[FrequencySeries, FrequencySeries]:
    """
    Welch's method for PSD estimation with overlapping segments.
    
    Args:
        series: Input time series
        window: Window function
        overlap: Overlap fraction
        nperseg: Segment length
    
    Returns:
        Tuple of (frequencies, psd)
    """
    n = len(series)
    
    if nperseg is None:
        nperseg = n // 8
    
    nperseg = min(nperseg, n)
    step = int(nperseg * (1 - overlap))
    
    # Generate overlapping segments
    segments = []
    for i in range(0, n - nperseg + 1, step):
        segment = series[i:i + nperseg]
        segments.append(segment)
    
    if not segments:
        return periodogram(series, window)
    
    # Compute periodogram for each segment and average
    freq_sum = None
    psd_sum = None
    
    for segment in segments:
        freqs, psd = periodogram(segment, window)
        
        if freq_sum is None:
            freq_sum = freqs
            psd_sum = psd
        else:
            psd_sum = [psd_sum[i] + psd[i] for i in range(len(psd))]
    
    # Average
    n_segments = len(segments)
    psd_avg = [p / n_segments for p in psd_sum]
    
    return freq_sum, psd_avg