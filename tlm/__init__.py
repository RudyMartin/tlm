# tlm/__init__.py — NumPy-free API (pure Python, stdlib only)

__version__ = "1.2.0"

# Core list-based ops
from .pure.ops import (
    array, shape, zeros, ones, eye,
    transpose, dot, matmul,
    add, sub, mul, div,
    asum, mean, var, std, amax, amin, argmax,
    exp, log, sqrt, clip, norm, l2_normalize,
    flatten, reshape, concatenate,
    # Random functions (removed - not in GitHub version)
    # seed_rng, random_uniform, random_normal, random_choice,
    # Distance & similarity metrics (from GitHub)
    cosine_similarity, euclidean_distance, manhattan_distance, hamming_distance,
    # Additional math operations (from GitHub)
    power, abs, sign,
    # Statistical functions (from GitHub)
    median, percentile, correlation, covariance_matrix,
    # Array manipulation (from GitHub)
    unique, where, tile, stack, vstack, hstack,
    # New array functions (from GitHub)
    histogram, allclose, searchsorted, diff, gradient, convolve,
    # Numerical validation (from GitHub)
    isfinite, isinf, isnan,
    # Special functions for transformers (from GitHub - but we'll use our attention module's version)
    # positional_encoding,  # Commented out - using attention module's version
)

# Activations / losses / metrics
from .core.activations import sigmoid, relu, leaky_relu, softmax
from .core.losses import (
    # Regression losses
    mse, mae, rmse, msle, mape, smape,
    huber_loss, log_cosh_loss, quantile_loss,
    # Classification losses
    binary_cross_entropy, cross_entropy, sparse_cross_entropy,
    focal_loss, hinge_loss, squared_hinge_loss, f2_loss,
    # Distribution losses
    bernoulli_loss, poisson_loss, gamma_loss, lucas_loss,
    # Probability losses
    kl_divergence, js_divergence, wasserstein_distance,
    # Ranking losses
    contrastive_loss, triplet_loss, margin_ranking_loss,
    # Robust losses
    robust_l1_loss, robust_l2_loss, tukey_loss,
    # Multi-task losses
    multi_task_loss, weighted_loss,
    # Statistical test losses
    durbin_watson_loss
)
from .core.metrics import confusion_matrix, accuracy

# Classification metrics (comprehensive)
from .core.classification_metrics import (
    # Binary metrics
    true_positives, true_negatives, false_positives, false_negatives,
    precision, recall, f1_score, specificity, sensitivity,
    balanced_accuracy, matthews_correlation_coefficient, cohen_kappa,
    # Multi-class metrics
    precision_multi, recall_multi, f1_score_multi,
    macro_avg_precision, macro_avg_recall, macro_avg_f1,
    weighted_avg_precision, weighted_avg_recall, weighted_avg_f1,
    micro_avg_precision, micro_avg_recall, micro_avg_f1,
    # Advanced metrics
    classification_report, roc_auc_binary, precision_recall_curve,
    average_precision_score, log_loss, brier_score,
    # Utility functions
    confusion_matrix_binary, confusion_matrix_metrics,
    support_per_class, class_distribution
)

# Linear models
from .linear_models.logistic_regression import (
    fit as logreg_fit,
    predict as logreg_predict,
    predict_proba as logreg_predict_proba,
)
from .linear_models.softmax_regression import (
    fit as softmax_fit,
    predict as softmax_predict,
    predict_proba as softmax_predict_proba,
)

# Clustering
from .cluster.kmeans import fit as kmeans_fit

# Model selection
from .model_selection.split import train_test_split, kfold, stratified_kfold

# SVM (linear)
from .svm.linear import hinge_loss as svm_hinge_loss, fit as svm_fit, predict as svm_predict

# PCA (power iteration) + whitening
from .decomp.pca_power import fit as pca_power_fit, transform as pca_power_transform, fit_transform as pca_power_fit_transform
from .decomp.whiten import whiten as pca_whiten

# GMM (EM, pure)
from .mixture.gmm_pure import fit as gmm_fit, predict_proba as gmm_predict_proba, predict as gmm_predict

# Anomaly detection
from .anomaly.gaussian import fit as anomaly_fit, score_logpdf as anomaly_score_logpdf, predict as anomaly_predict

# Naive Bayes
from .naive_bayes.multinomial import fit as nb_fit, predict_log_proba as nb_predict_log_proba, predict as nb_predict

# Matrix factorization
from .mf.matrix_factorization import fit_sgd as mf_fit_sgd, predict_full as mf_predict_full, mse as mf_mse

# Attention mechanisms (LOCAL ADDITION)
from .attention.scaled_dot_product import scaled_dot_product_attention, multi_head_attention, positional_encoding

# Network analysis (GITHUB ADDITION)
from .network import (
    # Core data structures
    Graph, WeightedGraph,
    # Centrality measures
    degree_centrality, betweenness_centrality, closeness_centrality,
    eigenvector_centrality, pagerank,
    # Clustering
    clustering_coefficient, global_clustering_coefficient, transitivity,
    # Network topology
    density, diameter, average_path_length, degree_distribution,
    assortativity_coefficient, modularity,
    # Paths and connectivity
    shortest_path, shortest_path_length, is_connected, connected_components,
    # Community detection
    louvain_communities, greedy_modularity_communities
)

# Signal processing and time series analysis (GITHUB ADDITION)
from .signal import (
    # Classical decomposition
    additive_decompose, multiplicative_decompose, seasonal_decompose,
    moving_average_trend, trend_strength, seasonal_strength, remainder_strength,
    # STL decomposition
    stl_decompose, robust_stl_decompose,
    # Spectral analysis
    fft_decompose, periodogram, power_spectral_density,
    autocorrelation_function, partial_autocorrelation_function,
    dominant_frequencies, spectral_centroid, spectral_bandwidth,
    # Wavelet analysis
    discrete_wavelet_transform, continuous_wavelet_transform,
    wavelet_decompose, wavelet_reconstruct,
    wavelet_coherence, wavelet_cross_correlation,
    # Filtering and noise
    white_noise_test, noise_variance_estimation, signal_to_noise_ratio,
    estimate_noise_level, moving_average_filter, exponential_smoothing,
    savitzky_golay_filter, median_filter,
    outlier_detection_seasonal, outlier_detection_spectral,
    # Trend analysis
    linear_trend, polynomial_trend, local_linear_trend,
    changepoint_detection, trend_changepoints,
    trend_slope, trend_acceleration, trend_volatility,
    # Seasonality
    detect_seasonality, seasonal_periods, seasonal_strength_test,
    seasonal_autocorrelation, seasonal_decomposition_strength,
    multiple_seasonal_decompose, hierarchical_seasonal_decompose,
)

# Trading algorithms and quantitative finance (GITHUB ADDITION)
from .trading import (
    # Technical indicators
    simple_moving_average, exponential_moving_average, weighted_moving_average,
    hull_moving_average, kaufman_adaptive_moving_average, z_score,
    rsi, stochastic_oscillator, williams_r, commodity_channel_index,
    macd, momentum, rate_of_change, awesome_oscillator,
    bollinger_bands, average_true_range, keltner_channels,
    on_balance_volume, volume_weighted_average_price, money_flow_index,
    
    # Trading strategies
    moving_average_crossover, breakout_strategy, turtle_trading,
    parabolic_sar_strategy, ichimoku_cloud_strategy,
    bollinger_bands_reversion, rsi_mean_reversion, pairs_trading,
    statistical_arbitrage, ornstein_uhlenbeck_reversion,
    momentum_strategy, relative_strength_strategy, gap_trading,
    
    # ML trading strategies
    woe_trading_strategy, bayesian_network_strategy, reinforcement_learning_trader,
    ensemble_strategy, head_shoulders_pattern, double_top_bottom, triangle_breakout,
    
    # Risk management
    kelly_criterion, fixed_fractional, volatility_position_sizing,
    sharpe_ratio, sortino_ratio, maximum_drawdown, value_at_risk,
    modern_portfolio_theory, black_litterman, risk_parity,
    
    # Performance and fairness metrics
    calculate_roc_metrics, analyze_missing_data, calculate_gini_metrics,
    calculate_fairness_metrics, calculate_diversity_metrics, wealth_gini_coefficient,
)

__all__ = [
    # ops
    'array','shape','zeros','ones','eye','transpose','dot','matmul',
    'add','sub','mul','div','asum','mean','var','std','amax','amin','argmax',
    'exp','log','sqrt','clip','norm','l2_normalize',
    'flatten','reshape','concatenate',
    # random (removed - not in GitHub version)
    # 'seed_rng','random_uniform','random_normal','random_choice',
    # distance & similarity (from GitHub)
    'cosine_similarity','euclidean_distance','manhattan_distance','hamming_distance',
    # additional math (from GitHub)
    'power','abs','sign',
    # statistical (from GitHub)
    'median','percentile','correlation','covariance_matrix',
    # array manipulation (from GitHub)
    'unique','where','tile','stack','vstack','hstack',
    # new array functions (from GitHub)
    'histogram','allclose','searchsorted','diff','gradient','convolve',
    # numerical validation (from GitHub)
    'isfinite','isinf','isnan',
    # core
    'sigmoid','relu','leaky_relu','softmax','confusion_matrix','accuracy',
    # regression losses
    'mse','mae','rmse','msle','mape','smape',
    'huber_loss','log_cosh_loss','quantile_loss',
    # classification losses
    'binary_cross_entropy','cross_entropy','sparse_cross_entropy',
    'focal_loss','hinge_loss','squared_hinge_loss','f2_loss',
    # distribution losses
    'bernoulli_loss','poisson_loss','gamma_loss','lucas_loss',
    # probability losses
    'kl_divergence','js_divergence','wasserstein_distance',
    # ranking losses
    'contrastive_loss','triplet_loss','margin_ranking_loss',
    # robust losses
    'robust_l1_loss','robust_l2_loss','tukey_loss',
    # multi-task losses
    'multi_task_loss','weighted_loss',
    # statistical test losses
    'durbin_watson_loss',
    # classification metrics (binary)
    'true_positives','true_negatives','false_positives','false_negatives',
    'precision','recall','f1_score','specificity','sensitivity',
    'balanced_accuracy','matthews_correlation_coefficient','cohen_kappa',
    # classification metrics (multi-class)
    'precision_multi','recall_multi','f1_score_multi',
    'macro_avg_precision','macro_avg_recall','macro_avg_f1',
    'weighted_avg_precision','weighted_avg_recall','weighted_avg_f1',
    'micro_avg_precision','micro_avg_recall','micro_avg_f1',
    # advanced classification metrics
    'classification_report','roc_auc_binary','precision_recall_curve',
    'average_precision_score','log_loss','brier_score',
    # classification utilities
    'confusion_matrix_binary','confusion_matrix_metrics',
    'support_per_class','class_distribution',
    # linear models & clustering & selection
    'logreg_fit','logreg_predict','logreg_predict_proba',
    'softmax_fit','softmax_predict','softmax_predict_proba',
    'kmeans_fit','train_test_split','kfold','stratified_kfold',
    # svm
    'svm_hinge_loss','svm_fit','svm_predict',
    # pca
    'pca_power_fit','pca_power_transform','pca_power_fit_transform','pca_whiten',
    # gmm
    'gmm_fit','gmm_predict_proba','gmm_predict',
    # anomaly
    'anomaly_fit','anomaly_score_logpdf','anomaly_predict',
    # naive bayes
    'nb_fit','nb_predict_log_proba','nb_predict',
    # matrix factorization
    'mf_fit_sgd','mf_predict_full','mf_mse',
    # attention (LOCAL ADDITION)
    'scaled_dot_product_attention','multi_head_attention','positional_encoding',
    # network analysis (GITHUB ADDITION)
    'Graph','WeightedGraph',
    'degree_centrality','betweenness_centrality','closeness_centrality',
    'eigenvector_centrality','pagerank',
    'clustering_coefficient','global_clustering_coefficient','transitivity',
    'density','diameter','average_path_length','degree_distribution',
    'assortativity_coefficient','modularity',
    'shortest_path','shortest_path_length','is_connected','connected_components',
    'louvain_communities','greedy_modularity_communities',
    # signal processing - decomposition (GITHUB ADDITION)
    'additive_decompose','multiplicative_decompose','seasonal_decompose',
    'moving_average_trend','trend_strength','seasonal_strength','remainder_strength',
    'stl_decompose','robust_stl_decompose',
    # signal processing - spectral (GITHUB ADDITION)
    'fft_decompose','periodogram','power_spectral_density',
    'autocorrelation_function','partial_autocorrelation_function',
    'dominant_frequencies','spectral_centroid','spectral_bandwidth',
    # signal processing - wavelets (GITHUB ADDITION)
    'discrete_wavelet_transform','continuous_wavelet_transform',
    'wavelet_decompose','wavelet_reconstruct',
    'wavelet_coherence','wavelet_cross_correlation',
    # signal processing - filtering (GITHUB ADDITION)
    'white_noise_test','noise_variance_estimation','signal_to_noise_ratio',
    'estimate_noise_level','moving_average_filter','exponential_smoothing',
    'savitzky_golay_filter','median_filter',
    'outlier_detection_seasonal','outlier_detection_spectral',
    # signal processing - trends (GITHUB ADDITION)
    'linear_trend','polynomial_trend','local_linear_trend',
    'changepoint_detection','trend_changepoints',
    'trend_slope','trend_acceleration','trend_volatility',
    # signal processing - seasonality (GITHUB ADDITION)
    'detect_seasonality','seasonal_periods','seasonal_strength_test',
    'seasonal_autocorrelation','seasonal_decomposition_strength',
    'multiple_seasonal_decompose','hierarchical_seasonal_decompose',
    # Trading algorithms and quantitative finance (GITHUB ADDITION)
    # Technical indicators
    'simple_moving_average','exponential_moving_average','weighted_moving_average',
    'hull_moving_average','kaufman_adaptive_moving_average','z_score',
    'rsi','stochastic_oscillator','williams_r','commodity_channel_index',
    'macd','momentum','rate_of_change','awesome_oscillator',
    'bollinger_bands','average_true_range','keltner_channels',
    'on_balance_volume','volume_weighted_average_price','money_flow_index',
    # Trading strategies  
    'moving_average_crossover','breakout_strategy','turtle_trading',
    'parabolic_sar_strategy','ichimoku_cloud_strategy',
    'bollinger_bands_reversion','rsi_mean_reversion','pairs_trading',
    'statistical_arbitrage','ornstein_uhlenbeck_reversion',
    'momentum_strategy','relative_strength_strategy','gap_trading',
    # ML trading strategies
    'woe_trading_strategy','bayesian_network_strategy','reinforcement_learning_trader',
    'ensemble_strategy','head_shoulders_pattern','double_top_bottom','triangle_breakout',
    # Risk management
    'kelly_criterion','fixed_fractional','volatility_position_sizing',
    'sharpe_ratio','sortino_ratio','maximum_drawdown','value_at_risk',
    'modern_portfolio_theory','black_litterman','risk_parity',
    # Performance and fairness metrics
    'calculate_roc_metrics','analyze_missing_data','calculate_gini_metrics',
    'calculate_fairness_metrics','calculate_diversity_metrics','wealth_gini_coefficient',
]