#!/usr/bin/env python3
"""
TLM Test Coverage Analysis by Category
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tlm

def analyze_coverage():
    """Analyze test coverage by category."""
    
    # Get all available functions
    all_functions = [name for name in dir(tlm) if callable(getattr(tlm, name)) and not name.startswith('_')]
    
    # Define comprehensive categories
    categories = {
        'Core Mathematical Operations': {
            'functions': ['add', 'sub', 'mul', 'div', 'asum', 'mean', 'var', 'std', 'amax', 'amin', 
                         'argmax', 'exp', 'log', 'sqrt', 'clip', 'abs', 'power', 'sign'],
            'tested': ['add', 'sub', 'mul', 'div', 'asum', 'mean', 'var', 'std', 'amax', 'amin', 
                      'argmax', 'exp', 'log', 'sqrt', 'clip', 'abs', 'power', 'sign']
        },
        
        'Array Operations': {
            'functions': ['array', 'shape', 'zeros', 'ones', 'eye', 'transpose', 'dot', 'matmul', 
                         'flatten', 'reshape', 'concatenate', 'norm', 'l2_normalize', 'tile', 'stack', 
                         'hstack', 'vstack', 'allclose', 'where', 'isnan', 'isinf', 'isfinite', 
                         'searchsorted', 'unique', 'diff', 'gradient'],
            'tested': ['array', 'shape', 'zeros', 'ones', 'eye', 'transpose', 'dot', 'matmul', 
                      'flatten', 'reshape', 'concatenate', 'norm', 'l2_normalize', 'tile', 'stack',
                      'hstack', 'vstack', 'allclose', 'where', 'isnan', 'isinf', 'isfinite',
                      'searchsorted', 'unique', 'diff', 'gradient']
        },
        
        'Statistics & Probability': {
            'functions': ['median', 'percentile', 'correlation', 'covariance_matrix', 'histogram'],
            'tested': ['median', 'percentile', 'correlation', 'covariance_matrix', 'histogram']
        },
        
        'Distance & Similarity': {
            'functions': ['cosine_similarity', 'euclidean_distance', 'manhattan_distance', 'hamming_distance'],
            'tested': ['cosine_similarity', 'euclidean_distance', 'manhattan_distance', 'hamming_distance']
        },
        
        'Loss Functions': {
            'functions': ['mse', 'mae', 'rmse', 'msle', 'mape', 'smape', 'binary_cross_entropy', 
                         'cross_entropy', 'sparse_cross_entropy', 'focal_loss', 'hinge_loss', 
                         'squared_hinge_loss', 'huber_loss', 'log_cosh_loss', 'quantile_loss',
                         'f2_loss', 'kl_divergence', 'js_divergence', 'wasserstein_distance',
                         'contrastive_loss', 'triplet_loss', 'margin_ranking_loss', 'bernoulli_loss',
                         'poisson_loss', 'gamma_loss', 'lucas_loss', 'robust_l1_loss', 'robust_l2_loss',
                         'tukey_loss', 'multi_task_loss', 'weighted_loss', 'durbin_watson_loss'],
            'tested': ['mse', 'mae', 'rmse', 'msle', 'mape', 'smape', 'binary_cross_entropy', 
                      'cross_entropy', 'sparse_cross_entropy', 'focal_loss', 'hinge_loss', 
                      'squared_hinge_loss', 'huber_loss', 'log_cosh_loss', 'quantile_loss',
                      'f2_loss', 'kl_divergence', 'js_divergence', 'wasserstein_distance',
                      'contrastive_loss', 'triplet_loss', 'margin_ranking_loss', 'bernoulli_loss',
                      'poisson_loss', 'gamma_loss', 'lucas_loss', 'robust_l1_loss', 'robust_l2_loss',
                      'tukey_loss', 'multi_task_loss', 'weighted_loss', 'durbin_watson_loss']
        },
        
        'Machine Learning Core': {
            'functions': ['logreg_fit', 'logreg_predict', 'logreg_predict_proba', 'svm_fit', 'svm_predict',
                         'svm_hinge_loss', 'kmeans_fit', 'nb_fit', 'nb_predict', 'nb_predict_log_proba',
                         'pca_power_fit', 'pca_power_transform', 'pca_power_fit_transform', 'pca_whiten',
                         'mf_fit_sgd', 'mf_predict_full', 'mf_mse', 'softmax_fit', 'softmax_predict', 
                         'softmax_predict_proba', 'gmm_fit', 'gmm_predict', 'gmm_predict_proba',
                         'anomaly_fit', 'anomaly_predict', 'anomaly_score_logpdf', 'train_test_split'],
            'tested': ['logreg_fit', 'logreg_predict', 'logreg_predict_proba', 'svm_fit', 'svm_predict',
                      'svm_hinge_loss', 'kmeans_fit', 'nb_fit', 'nb_predict', 'pca_power_fit',
                      'pca_power_transform', 'pca_power_fit_transform', 'pca_whiten', 'mf_fit_sgd', 
                      'mf_predict_full', 'mf_mse', 'softmax_fit', 'softmax_predict', 'softmax_predict_proba',
                      'gmm_fit', 'gmm_predict', 'gmm_predict_proba', 'anomaly_fit', 'anomaly_predict',
                      'anomaly_score_logpdf', 'train_test_split']
        },
        
        'Activation Functions': {
            'functions': ['sigmoid', 'relu', 'leaky_relu', 'softmax'],
            'tested': ['sigmoid', 'relu', 'leaky_relu', 'softmax']
        },
        
        'Classification Metrics': {
            'functions': ['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix',
                         'confusion_matrix_binary', 'confusion_matrix_metrics', 'true_positives',
                         'false_positives', 'true_negatives', 'false_negatives', 'balanced_accuracy',
                         'matthews_correlation_coefficient', 'classification_report', 'roc_auc_binary',
                         'log_loss', 'brier_score', 'precision_multi', 'recall_multi', 'f1_score_multi',
                         'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1', 'micro_avg_precision',
                         'micro_avg_recall', 'micro_avg_f1', 'weighted_avg_precision', 'weighted_avg_recall',
                         'weighted_avg_f1', 'sensitivity', 'specificity', 'cohen_kappa',
                         'average_precision_score', 'precision_recall_curve', 'calculate_roc_metrics'],
            'tested': ['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix',
                      'confusion_matrix_binary', 'confusion_matrix_metrics', 'true_positives',
                      'false_positives', 'true_negatives', 'false_negatives', 'balanced_accuracy',
                      'matthews_correlation_coefficient', 'classification_report', 'roc_auc_binary',
                      'log_loss', 'brier_score', 'precision_multi', 'recall_multi', 'f1_score_multi',
                      'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1', 'micro_avg_precision',
                      'micro_avg_recall', 'micro_avg_f1', 'weighted_avg_precision', 'weighted_avg_recall',
                      'weighted_avg_f1', 'sensitivity', 'specificity', 'cohen_kappa',
                      'average_precision_score', 'precision_recall_curve', 'calculate_roc_metrics']
        },
        
        'Signal Processing': {
            'functions': ['fft_decompose', 'periodogram', 'autocorrelation_function', 'partial_autocorrelation_function',
                         'white_noise_test', 'additive_decompose', 'multiplicative_decompose', 'seasonal_decompose',
                         'stl_decompose', 'robust_stl_decompose', 'hierarchical_seasonal_decompose',
                         'multiple_seasonal_decompose', 'moving_average_trend', 'linear_trend', 'local_linear_trend',
                         'polynomial_trend', 'changepoint_detection', 'trend_slope', 'trend_acceleration',
                         'trend_changepoints', 'trend_strength', 'trend_volatility', 'detect_seasonality',
                         'seasonal_periods', 'seasonal_autocorrelation', 'seasonal_strength', 
                         'seasonal_strength_test', 'seasonal_decomposition_strength', 'remainder_strength',
                         'moving_average_filter', 'exponential_smoothing', 'median_filter', 
                         'savitzky_golay_filter', 'noise_variance_estimation', 'estimate_noise_level',
                         'signal_to_noise_ratio', 'continuous_wavelet_transform', 'discrete_wavelet_transform',
                         'wavelet_decompose', 'wavelet_reconstruct', 'wavelet_coherence', 
                         'wavelet_cross_correlation', 'spectral_centroid', 'spectral_bandwidth',
                         'power_spectral_density', 'dominant_frequencies', 'convolve',
                         'outlier_detection_seasonal', 'outlier_detection_spectral'],
            'tested': ['fft_decompose', 'periodogram', 'autocorrelation_function', 'white_noise_test',
                      'additive_decompose', 'seasonal_decompose', 'moving_average_trend', 'linear_trend',
                      'changepoint_detection', 'trend_slope', 'detect_seasonality', 'seasonal_periods',
                      'moving_average_filter', 'exponential_smoothing', 'median_filter', 
                      'noise_variance_estimation', 'signal_to_noise_ratio']
        },
        
        'Trading Indicators': {
            'functions': ['simple_moving_average', 'exponential_moving_average', 'weighted_moving_average',
                         'hull_moving_average', 'kaufman_adaptive_moving_average', 'rsi', 'bollinger_bands',
                         'keltner_channels', 'macd', 'stochastic_oscillator', 'awesome_oscillator',
                         'commodity_channel_index', 'williams_r', 'money_flow_index', 'average_true_range',
                         'rate_of_change', 'momentum', 'on_balance_volume', 'volume_weighted_average_price',
                         'head_shoulders_pattern', 'double_top_bottom', 'triangle_breakout'],
            'tested': ['simple_moving_average', 'exponential_moving_average', 'rsi', 'bollinger_bands',
                      'macd', 'stochastic_oscillator']
        },
        
        'Risk Management': {
            'functions': ['sharpe_ratio', 'sortino_ratio', 'maximum_drawdown', 'kelly_criterion', 'value_at_risk'],
            'tested': ['sharpe_ratio', 'maximum_drawdown', 'kelly_criterion', 'value_at_risk']
        },
        
        'Trading Strategies': {
            'functions': ['moving_average_crossover', 'momentum_strategy', 'pairs_trading', 'turtle_trading',
                         'breakout_strategy', 'gap_trading', 'bollinger_bands_reversion', 'rsi_mean_reversion',
                         'relative_strength_strategy', 'ichimoku_cloud_strategy', 'parabolic_sar_strategy',
                         'statistical_arbitrage', 'ornstein_uhlenbeck_reversion', 'woe_trading_strategy',
                         'reinforcement_learning_trader'],
            'tested': ['moving_average_crossover']
        },
        
        'Portfolio Management': {
            'functions': ['modern_portfolio_theory', 'black_litterman', 'risk_parity', 'fixed_fractional',
                         'volatility_position_sizing', 'bayesian_network_strategy'],
            'tested': []
        },
        
        'Fairness & Bias Metrics': {
            'functions': ['calculate_gini_metrics', 'wealth_gini_coefficient', 'calculate_fairness_metrics',
                         'calculate_diversity_metrics', 'analyze_missing_data'],
            'tested': ['calculate_gini_metrics', 'wealth_gini_coefficient']
        },
        
        'Network Analysis': {
            'functions': ['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 
                         'eigenvector_centrality', 'clustering_coefficient', 'global_clustering_coefficient',
                         'density', 'shortest_path', 'shortest_path_length', 'average_path_length',
                         'diameter', 'is_connected', 'connected_components', 'assortativity_coefficient',
                         'modularity', 'transitivity', 'pagerank', 'degree_distribution',
                         'greedy_modularity_communities', 'louvain_communities', 'Graph', 'WeightedGraph'],
            'tested': ['degree_centrality', 'betweenness_centrality', 'closeness_centrality',
                      'clustering_coefficient', 'density', 'shortest_path', 'is_connected']
        },
        
        'Attention Mechanisms': {
            'functions': ['scaled_dot_product_attention', 'multi_head_attention', 'positional_encoding'],
            'tested': ['scaled_dot_product_attention', 'multi_head_attention', 'positional_encoding']
        },
        
        'Model Selection': {
            'functions': ['kfold', 'stratified_kfold'],
            'tested': []
        },
        
        'Utility Functions': {
            'functions': ['class_distribution', 'support_per_class', 'z_score'],
            'tested': []
        }
    }
    
    print('TLM TEST COVERAGE BY CATEGORY')
    print('=' * 60)
    print()
    
    total_functions = 0
    total_tested = 0
    
    # Calculate coverage for each category
    category_stats = []
    for category, data in categories.items():
        available = [f for f in data['functions'] if f in all_functions]
        tested = [f for f in data['tested'] if f in all_functions]
        
        if available:  # Only include categories with actual functions
            coverage = len(tested) / len(available) * 100
            category_stats.append((category, len(tested), len(available), coverage, available, tested))
            total_functions += len(available)
            total_tested += len(tested)
    
    # Sort by coverage percentage (descending)
    category_stats.sort(key=lambda x: x[3], reverse=True)
    
    for category, tested_count, total_count, coverage, available, tested in category_stats:
        if coverage == 100:
            status = 'COMPLETE'
        elif coverage >= 80:
            status = 'HIGH    '
        elif coverage >= 50:
            status = 'MEDIUM  '
        else:
            status = 'LOW     '
            
        print(f'{status} {category:35s}: {tested_count:2d}/{total_count:2d} ({coverage:5.1f}%)')
    
    print()
    print(f'OVERALL COVERAGE: {total_tested}/{total_functions} ({100*total_tested/total_functions:.1f}%)')
    
    # Show detailed breakdown for incomplete categories
    print()
    print('DETAILED BREAKDOWN OF INCOMPLETE CATEGORIES:')
    print('=' * 60)
    
    for category, tested_count, total_count, coverage, available, tested in category_stats:
        if coverage < 100:
            untested = [f for f in available if f not in tested]
            print(f'\n{category} ({tested_count}/{total_count} - {coverage:.1f}%):')
            if tested:
                print(f'  TESTED ({len(tested)}): {", ".join(tested[:5])}{"..." if len(tested) > 5 else ""}')
            if untested:
                print(f'  UNTESTED ({len(untested)}): {", ".join(untested[:8])}{"..." if len(untested) > 8 else ""}')

if __name__ == "__main__":
    analyze_coverage()