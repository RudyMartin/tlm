"""
Signal processing and time series decomposition module for TLM.

Provides comprehensive functions for analyzing signals, decomposing time series
into trend/seasonal/noise components, and spectral analysis - essential for 
TidyMart analytics, financial analysis, and temporal pattern detection.
"""

from .decomposition import (
    # Classical decomposition
    additive_decompose, multiplicative_decompose,
    seasonal_decompose, moving_average_trend,
    
    # STL decomposition
    stl_decompose, robust_stl_decompose,
    
    # Components analysis
    trend_strength, seasonal_strength, remainder_strength
)

from .spectral import (
    # Fourier analysis
    fft_decompose, periodogram, power_spectral_density,
    autocorrelation_function, partial_autocorrelation_function,
    
    # Frequency domain
    dominant_frequencies, spectral_centroid, spectral_bandwidth
)

from .wavelets import (
    # Wavelet decomposition
    discrete_wavelet_transform, continuous_wavelet_transform,
    wavelet_decompose, wavelet_reconstruct,
    
    # Wavelet analysis
    wavelet_coherence, wavelet_cross_correlation
)

from .filtering import (
    # Noise analysis
    white_noise_test, noise_variance_estimation,
    signal_to_noise_ratio, estimate_noise_level,
    
    # Filtering
    moving_average_filter, exponential_smoothing,
    savitzky_golay_filter, median_filter,
    
    # Outlier detection
    outlier_detection_seasonal, outlier_detection_spectral
)

from .trends import (
    # Trend analysis
    linear_trend, polynomial_trend, local_linear_trend,
    changepoint_detection, trend_changepoints,
    
    # Trend metrics
    trend_slope, trend_acceleration, trend_volatility
)

from .seasonality import (
    # Seasonal analysis
    detect_seasonality, seasonal_periods, seasonal_strength_test,
    seasonal_autocorrelation, seasonal_decomposition_strength,
    
    # Multiple seasonality
    multiple_seasonal_decompose, hierarchical_seasonal_decompose
)

__all__ = [
    # Classical decomposition
    'additive_decompose', 'multiplicative_decompose', 'seasonal_decompose',
    'moving_average_trend', 'trend_strength', 'seasonal_strength', 'remainder_strength',
    
    # STL decomposition
    'stl_decompose', 'robust_stl_decompose',
    
    # Spectral analysis
    'fft_decompose', 'periodogram', 'power_spectral_density',
    'autocorrelation_function', 'partial_autocorrelation_function',
    'dominant_frequencies', 'spectral_centroid', 'spectral_bandwidth',
    
    # Wavelet analysis
    'discrete_wavelet_transform', 'continuous_wavelet_transform',
    'wavelet_decompose', 'wavelet_reconstruct',
    'wavelet_coherence', 'wavelet_cross_correlation',
    
    # Filtering and noise
    'white_noise_test', 'noise_variance_estimation', 'signal_to_noise_ratio',
    'estimate_noise_level', 'moving_average_filter', 'exponential_smoothing',
    'savitzky_golay_filter', 'median_filter',
    'outlier_detection_seasonal', 'outlier_detection_spectral',
    
    # Trend analysis
    'linear_trend', 'polynomial_trend', 'local_linear_trend',
    'changepoint_detection', 'trend_changepoints',
    'trend_slope', 'trend_acceleration', 'trend_volatility',
    
    # Seasonality
    'detect_seasonality', 'seasonal_periods', 'seasonal_strength_test',
    'seasonal_autocorrelation', 'seasonal_decomposition_strength',
    'multiple_seasonal_decompose', 'hierarchical_seasonal_decompose',
]