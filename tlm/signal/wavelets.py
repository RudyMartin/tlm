"""
Wavelet analysis functions for TLM signal processing.

Provides multi-scale decomposition using wavelets for time-frequency
analysis, denoising, and feature extraction from time series.
"""

from typing import List, Tuple, Dict, Optional
import math

# Type definitions
TimeSeries = List[float]
WaveletCoeffs = List[float]

def discrete_wavelet_transform(series: TimeSeries, wavelet: str = 'haar', 
                              levels: Optional[int] = None) -> Tuple[List[WaveletCoeffs], WaveletCoeffs]:
    """
    Discrete Wavelet Transform using specified wavelet.
    
    Args:
        series: Input time series
        wavelet: Wavelet type ('haar', 'db4', 'biorthogonal')
        levels: Number of decomposition levels (auto if None)
    
    Returns:
        Tuple of (detail_coefficients, approximation_coefficients)
    """
    if levels is None:
        levels = min(6, int(math.log2(len(series))))
    
    # Get wavelet filters
    if wavelet == 'haar':
        h, g = _haar_filters()
    elif wavelet == 'db4':
        h, g = _daubechies4_filters()
    else:
        raise ValueError(f"Unsupported wavelet: {wavelet}")
    
    # Multi-level decomposition
    details = []
    current = series[:]
    
    for level in range(levels):
        if len(current) < len(h):
            break
            
        # Convolve and downsample
        approx = _convolve_downsample(current, h)
        detail = _convolve_downsample(current, g)
        
        details.append(detail)
        current = approx
    
    return details, current


def continuous_wavelet_transform(series: TimeSeries, scales: List[float], 
                               wavelet: str = 'morlet') -> List[List[complex]]:
    """
    Continuous Wavelet Transform for time-frequency analysis.
    
    Args:
        series: Input time series
        scales: List of scales to analyze
        wavelet: Mother wavelet ('morlet', 'mexican_hat')
    
    Returns:
        2D array of complex wavelet coefficients [scale][time]
    """
    n = len(series)
    coeffs = []
    
    for scale in scales:
        scale_coeffs = []
        
        for t in range(n):
            coeff = 0.0 + 0.0j
            
            for tau in range(n):
                if wavelet == 'morlet':
                    psi = _morlet_wavelet((t - tau) / scale)
                elif wavelet == 'mexican_hat':
                    psi = _mexican_hat_wavelet((t - tau) / scale)
                else:
                    raise ValueError(f"Unsupported wavelet: {wavelet}")
                
                coeff += series[tau] * psi.conjugate()
            
            coeff /= math.sqrt(scale)  # Normalization
            scale_coeffs.append(coeff)
        
        coeffs.append(scale_coeffs)
    
    return coeffs


def wavelet_decompose(series: TimeSeries, wavelet: str = 'haar', 
                     mode: str = 'symmetric') -> Dict[str, WaveletCoeffs]:
    """
    Single-level wavelet decomposition.
    
    Args:
        series: Input time series
        wavelet: Wavelet type
        mode: Boundary condition handling
    
    Returns:
        Dictionary with 'approximation' and 'detail' coefficients
    """
    details, approx = discrete_wavelet_transform(series, wavelet, levels=1)
    
    return {
        'approximation': approx,
        'detail': details[0] if details else []
    }


def wavelet_reconstruct(coeffs: Dict[str, WaveletCoeffs], 
                       wavelet: str = 'haar') -> TimeSeries:
    """
    Reconstruct signal from wavelet coefficients.
    
    Args:
        coeffs: Dictionary with approximation and detail coefficients
        wavelet: Wavelet type used for decomposition
    
    Returns:
        Reconstructed time series
    """
    if wavelet == 'haar':
        h, g = _haar_filters()
    elif wavelet == 'db4':
        h, g = _daubechies4_filters()
    else:
        raise ValueError(f"Unsupported wavelet: {wavelet}")
    
    approx = coeffs['approximation']
    detail = coeffs['detail']
    
    # Upsample and convolve
    approx_up = _upsample_convolve(approx, h)
    detail_up = _upsample_convolve(detail, g)
    
    # Add components
    n = min(len(approx_up), len(detail_up))
    reconstructed = [approx_up[i] + detail_up[i] for i in range(n)]
    
    return reconstructed


def wavelet_denoise(series: TimeSeries, wavelet: str = 'db4', 
                   threshold_mode: str = 'soft', threshold: Optional[float] = None) -> TimeSeries:
    """
    Denoise signal using wavelet thresholding.
    
    Args:
        series: Noisy input time series
        wavelet: Wavelet type for decomposition
        threshold_mode: 'soft' or 'hard' thresholding
        threshold: Threshold value (auto-estimated if None)
    
    Returns:
        Denoised time series
    """
    # Multi-level wavelet decomposition
    details, approx = discrete_wavelet_transform(series, wavelet)
    
    # Estimate threshold if not provided
    if threshold is None:
        # Use universal threshold based on noise level
        if details:
            noise_level = _estimate_noise_level_wavelet(details[-1])  # Finest detail
            n = len(series)
            threshold = noise_level * math.sqrt(2 * math.log(n))
    
    # Apply thresholding to detail coefficients
    thresholded_details = []
    for detail in details:
        if threshold_mode == 'soft':
            thresholded = [_soft_threshold(x, threshold) for x in detail]
        elif threshold_mode == 'hard':
            thresholded = [_hard_threshold(x, threshold) for x in detail]
        else:
            raise ValueError(f"Unknown threshold mode: {threshold_mode}")
        
        thresholded_details.append(thresholded)
    
    # Reconstruct signal
    reconstructed = _wavelet_reconstruct_multilevel(thresholded_details, approx, wavelet)
    
    return reconstructed


def wavelet_coherence(series1: TimeSeries, series2: TimeSeries, 
                     scales: List[float], wavelet: str = 'morlet') -> List[List[float]]:
    """
    Compute wavelet coherence between two time series.
    
    Args:
        series1: First time series
        series2: Second time series  
        scales: Scales for analysis
        wavelet: Mother wavelet
    
    Returns:
        2D coherence matrix [scale][time]
    """
    # Compute CWT for both series
    cwt1 = continuous_wavelet_transform(series1, scales, wavelet)
    cwt2 = continuous_wavelet_transform(series2, scales, wavelet)
    
    coherence = []
    
    for i, scale in enumerate(scales):
        scale_coherence = []
        
        for t in range(len(series1)):
            # Cross-wavelet transform
            cross_wt = cwt1[i][t] * cwt2[i][t].conjugate()
            
            # Auto-wavelet transforms
            auto_wt1 = abs(cwt1[i][t]) ** 2
            auto_wt2 = abs(cwt2[i][t]) ** 2
            
            # Coherence
            if auto_wt1 * auto_wt2 > 0:
                coh = abs(cross_wt) / math.sqrt(auto_wt1 * auto_wt2)
            else:
                coh = 0.0
            
            scale_coherence.append(coh)
        
        coherence.append(scale_coherence)
    
    return coherence


def wavelet_cross_correlation(series1: TimeSeries, series2: TimeSeries,
                             scales: List[float], max_lag: int = 10) -> Dict[float, List[float]]:
    """
    Compute scale-dependent cross-correlation using wavelets.
    
    Args:
        series1: First time series
        series2: Second time series
        scales: Scales for analysis  
        max_lag: Maximum lag to compute
    
    Returns:
        Dictionary mapping scales to cross-correlation arrays
    """
    correlations = {}
    
    for scale in scales:
        # Decompose both series at this scale
        cwt1 = continuous_wavelet_transform(series1, [scale], 'morlet')[0]
        cwt2 = continuous_wavelet_transform(series2, [scale], 'morlet')[0]
        
        # Extract real parts for correlation
        real1 = [c.real for c in cwt1]
        real2 = [c.real for c in cwt2]
        
        # Compute cross-correlation
        corr = _cross_correlation(real1, real2, max_lag)
        correlations[scale] = corr
    
    return correlations


# Helper functions

def _haar_filters() -> Tuple[List[float], List[float]]:
    """Haar wavelet filters."""
    sqrt2 = math.sqrt(2)
    h = [1/sqrt2, 1/sqrt2]  # Low-pass (scaling)
    g = [1/sqrt2, -1/sqrt2]  # High-pass (wavelet)
    return h, g


def _daubechies4_filters() -> Tuple[List[float], List[float]]:
    """Daubechies-4 wavelet filters."""
    sqrt2 = math.sqrt(2)
    h = [(1 + math.sqrt(3))/(4*sqrt2), (3 + math.sqrt(3))/(4*sqrt2),
         (3 - math.sqrt(3))/(4*sqrt2), (1 - math.sqrt(3))/(4*sqrt2)]
    
    # High-pass filter (alternating signs)
    g = [h[3], -h[2], h[1], -h[0]]
    
    return h, g


def _morlet_wavelet(t: float, omega0: float = 6.0) -> complex:
    """Complex Morlet wavelet."""
    if abs(t) > 5:  # Truncate for efficiency
        return 0.0 + 0.0j
    
    norm = math.pi ** (-0.25)
    gaussian = math.exp(-t*t / 2)
    oscillation = complex(math.cos(omega0 * t), math.sin(omega0 * t))
    
    return norm * gaussian * oscillation


def _mexican_hat_wavelet(t: float) -> float:
    """Mexican hat (Ricker) wavelet."""
    if abs(t) > 5:  # Truncate for efficiency
        return 0.0
    
    norm = 2 / (math.sqrt(3) * math.pi**0.25)
    gaussian = math.exp(-t*t / 2)
    polynomial = (1 - t*t)
    
    return norm * polynomial * gaussian


def _convolve_downsample(signal: List[float], filt: List[float]) -> List[float]:
    """Convolve signal with filter and downsample by 2."""
    n = len(signal)
    m = len(filt)
    result = []
    
    for i in range(0, n, 2):  # Downsample by 2
        conv_sum = 0.0
        for j in range(m):
            if i - j >= 0 and i - j < n:
                conv_sum += signal[i - j] * filt[j]
        result.append(conv_sum)
    
    return result


def _upsample_convolve(coeffs: List[float], filt: List[float]) -> List[float]:
    """Upsample coefficients and convolve with filter."""
    # Upsample by inserting zeros
    upsampled = []
    for c in coeffs:
        upsampled.extend([c, 0.0])
    
    # Convolve with filter
    n = len(upsampled)
    m = len(filt)
    result = []
    
    for i in range(n):
        conv_sum = 0.0
        for j in range(m):
            if i - j >= 0 and i - j < n:
                conv_sum += upsampled[i - j] * filt[j]
        result.append(conv_sum)
    
    return result


def _soft_threshold(x: float, threshold: float) -> float:
    """Soft thresholding function."""
    if x > threshold:
        return x - threshold
    elif x < -threshold:
        return x + threshold
    else:
        return 0.0


def _hard_threshold(x: float, threshold: float) -> float:
    """Hard thresholding function."""
    return x if abs(x) > threshold else 0.0


def _estimate_noise_level_wavelet(detail_coeffs: List[float]) -> float:
    """Estimate noise level from finest detail coefficients."""
    if not detail_coeffs:
        return 0.0
    
    # Use median absolute deviation
    median_val = sorted(detail_coeffs)[len(detail_coeffs) // 2]
    mad = sorted([abs(x - median_val) for x in detail_coeffs])[len(detail_coeffs) // 2]
    
    return mad / 0.6745  # Convert MAD to standard deviation estimate


def _wavelet_reconstruct_multilevel(details: List[List[float]], 
                                   approx: List[float], wavelet: str) -> List[float]:
    """Reconstruct signal from multi-level wavelet coefficients."""
    # Start with approximation coefficients
    current = approx[:]
    
    # Reconstruct level by level
    for detail in reversed(details):
        coeffs = {'approximation': current, 'detail': detail}
        current = wavelet_reconstruct(coeffs, wavelet)
    
    return current


def _cross_correlation(x: List[float], y: List[float], max_lag: int) -> List[float]:
    """Compute cross-correlation between two series."""
    n = len(x)
    m = len(y)
    min_len = min(n, m)
    
    correlations = []
    
    for lag in range(-max_lag, max_lag + 1):
        corr_sum = 0.0
        count = 0
        
        for i in range(min_len):
            j = i + lag
            if 0 <= j < min_len:
                corr_sum += x[i] * y[j]
                count += 1
        
        correlations.append(corr_sum / count if count > 0 else 0.0)
    
    return correlations