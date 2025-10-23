#!/usr/bin/env python3
"""
Analyze and identify all remaining untested functions in TLM.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tlm

def get_all_functions():
    """Get all callable functions from TLM."""
    return [name for name in dir(tlm) if callable(getattr(tlm, name)) and not name.startswith('_')]

def get_tested_functions():
    """Return list of all successfully tested functions."""
    tested = [
        # Core Mathematical Operations (18/18)
        'add', 'sub', 'mul', 'div', 'asum', 'mean', 'var', 'std', 'amax', 'amin',
        'argmax', 'exp', 'log', 'sqrt', 'clip', 'abs', 'power', 'sign',
        
        # Array Operations (26/26)
        'array', 'shape', 'zeros', 'ones', 'eye', 'transpose', 'dot', 'matmul',
        'flatten', 'reshape', 'concatenate', 'norm', 'l2_normalize', 'tile', 'stack',
        'hstack', 'vstack', 'allclose', 'where', 'isnan', 'isinf', 'isfinite',
        'searchsorted', 'unique', 'diff', 'gradient',
        
        # Statistics & Probability (5/5)
        'median', 'percentile', 'correlation', 'covariance_matrix', 'histogram',
        
        # Distance & Similarity (4/4)
        'cosine_similarity', 'euclidean_distance', 'manhattan_distance', 'hamming_distance',
        
        # Loss Functions (32/32)
        'mse', 'mae', 'rmse', 'msle', 'mape', 'smape', 'binary_cross_entropy',
        'cross_entropy', 'sparse_cross_entropy', 'focal_loss', 'hinge_loss',
        'squared_hinge_loss', 'huber_loss', 'log_cosh_loss', 'quantile_loss',
        'f2_loss', 'kl_divergence', 'js_divergence', 'wasserstein_distance',
        'contrastive_loss', 'triplet_loss', 'margin_ranking_loss', 'bernoulli_loss',
        'poisson_loss', 'gamma_loss', 'lucas_loss', 'robust_l1_loss', 'robust_l2_loss',
        'tukey_loss', 'multi_task_loss', 'weighted_loss', 'durbin_watson_loss',
        
        # Machine Learning Core (26/27)
        'logreg_fit', 'logreg_predict', 'logreg_predict_proba', 'svm_fit', 'svm_predict',
        'svm_hinge_loss', 'kmeans_fit', 'nb_fit', 'nb_predict', 'nb_predict_log_proba',
        'pca_power_fit', 'pca_power_transform', 'pca_power_fit_transform', 'pca_whiten',
        'mf_fit_sgd', 'mf_predict_full', 'mf_mse', 'softmax_fit', 'softmax_predict',
        'softmax_predict_proba', 'gmm_fit', 'gmm_predict', 'gmm_predict_proba',
        'anomaly_fit', 'anomaly_predict', 'anomaly_score_logpdf', 'train_test_split',
        
        # Activation Functions (4/4)
        'sigmoid', 'relu', 'leaky_relu', 'softmax',
        
        # Classification Metrics (35/35)
        'accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix',
        'confusion_matrix_binary', 'confusion_matrix_metrics', 'true_positives',
        'false_positives', 'true_negatives', 'false_negatives', 'balanced_accuracy',
        'matthews_correlation_coefficient', 'classification_report', 'roc_auc_binary',
        'log_loss', 'brier_score', 'precision_multi', 'recall_multi', 'f1_score_multi',
        'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1', 'micro_avg_precision',
        'micro_avg_recall', 'micro_avg_f1', 'weighted_avg_precision', 'weighted_avg_recall',
        'weighted_avg_f1', 'sensitivity', 'specificity', 'cohen_kappa',
        'average_precision_score', 'precision_recall_curve', 'calculate_roc_metrics',
        
        # Risk Management (5/5)
        'sharpe_ratio', 'sortino_ratio', 'maximum_drawdown', 'kelly_criterion', 'value_at_risk',
        
        # Model Selection (2/2)
        'kfold', 'stratified_kfold',
        
        # Utility Functions (3/3)
        'class_distribution', 'support_per_class', 'z_score',
        
        # Attention Mechanisms (3/3)
        'scaled_dot_product_attention', 'multi_head_attention', 'positional_encoding',
        
        # Fairness Metrics (2/3 that worked)
        'calculate_fairness_metrics', 'analyze_missing_data',
        'calculate_gini_metrics', 'wealth_gini_coefficient',
        
        # Trading Indicators (working ones)
        'simple_moving_average', 'exponential_moving_average', 'rsi', 'bollinger_bands',
        'macd', 'stochastic_oscillator', 'hull_moving_average', 'rate_of_change',
        'momentum', 'on_balance_volume', 'head_shoulders_pattern', 'double_top_bottom',
        
        # Portfolio Management (working ones)
        'modern_portfolio_theory', 'risk_parity',
        
        # Trading Strategies (working ones)
        'moving_average_crossover', 'pairs_trading', 'bollinger_bands_reversion',
        'rsi_mean_reversion', 'woe_trading_strategy',
        
        # Signal Processing (working ones)
        'fft_decompose', 'periodogram', 'autocorrelation_function', 'white_noise_test',
        'additive_decompose', 'seasonal_decompose', 'moving_average_trend', 'linear_trend',
        'changepoint_detection', 'trend_slope', 'detect_seasonality', 'seasonal_periods',
        'moving_average_filter', 'exponential_smoothing', 'median_filter',
        'noise_variance_estimation', 'signal_to_noise_ratio',
        'partial_autocorrelation_function', 'hierarchical_seasonal_decompose',
        'multiple_seasonal_decompose', 'local_linear_trend', 'polynomial_trend',
        'trend_acceleration', 'trend_changepoints', 'trend_volatility',
        'seasonal_autocorrelation', 'seasonal_strength_test', 'savitzky_golay_filter',
        'estimate_noise_level', 'discrete_wavelet_transform', 'wavelet_decompose',
        'spectral_centroid', 'spectral_bandwidth', 'power_spectral_density',
        'dominant_frequencies', 'convolve', 'outlier_detection_seasonal',
        'outlier_detection_spectral', 'seasonal_decomposition_strength',
        
        # Network Analysis (working ones)
        'degree_centrality', 'betweenness_centrality', 'closeness_centrality',
        'clustering_coefficient', 'density', 'shortest_path', 'is_connected',
        'Graph', 'WeightedGraph',
    ]
    return tested

def main():
    """Identify all remaining untested functions."""
    all_funcs = get_all_functions()
    tested_funcs = get_tested_functions()
    
    # Find untested functions
    untested = [f for f in all_funcs if f not in tested_funcs]
    
    print("REMAINING UNTESTED FUNCTIONS ANALYSIS")
    print("=" * 60)
    print(f"Total Functions: {len(all_funcs)}")
    print(f"Tested Functions: {len(tested_funcs)}")
    print(f"Untested Functions: {len(untested)}")
    print(f"Coverage: {100 * len(tested_funcs) / len(all_funcs):.1f}%")
    
    print("\n" + "=" * 60)
    print("UNTESTED FUNCTIONS LIST")
    print("=" * 60)
    
    # Categorize untested functions
    categories = {
        'Trading Indicators': [],
        'Trading Strategies': [],
        'Portfolio Management': [],
        'Signal Processing': [],
        'Network Analysis': [],
        'Fairness Metrics': [],
        'Other': []
    }
    
    # Categorization keywords
    trading_indicators = ['weighted_moving_average', 'kaufman', 'keltner', 'awesome', 'commodity',
                         'williams', 'money_flow', 'average_true_range', 'volume_weighted', 'triangle']
    trading_strategies = ['momentum_strategy', 'turtle', 'breakout', 'gap', 'relative_strength',
                         'ichimoku', 'parabolic', 'statistical_arbitrage', 'ornstein', 'reinforcement']
    portfolio = ['black_litterman', 'fixed_fractional', 'volatility_position', 'bayesian_network']
    signal = ['multiplicative_decompose', 'stl_decompose', 'robust_stl', 'trend_strength',
             'continuous_wavelet', 'wavelet_coherence', 'wavelet_cross', 'wavelet_reconstruct',
             'remainder_strength']
    network = ['eigenvector_centrality', 'global_clustering', 'shortest_path_length',
              'average_path_length', 'diameter', 'connected_components', 'assortativity',
              'modularity', 'transitivity', 'pagerank', 'degree_distribution',
              'greedy_modularity', 'louvain']
    fairness = ['calculate_diversity_metrics']
    
    for func in untested:
        categorized = False
        
        # Check categories
        for keyword in trading_indicators:
            if keyword in func.lower():
                categories['Trading Indicators'].append(func)
                categorized = True
                break
        
        if not categorized:
            for keyword in trading_strategies:
                if keyword in func.lower():
                    categories['Trading Strategies'].append(func)
                    categorized = True
                    break
        
        if not categorized:
            for keyword in portfolio:
                if keyword in func.lower():
                    categories['Portfolio Management'].append(func)
                    categorized = True
                    break
        
        if not categorized:
            for keyword in signal:
                if keyword in func.lower():
                    categories['Signal Processing'].append(func)
                    categorized = True
                    break
        
        if not categorized:
            for keyword in network:
                if keyword in func.lower():
                    categories['Network Analysis'].append(func)
                    categorized = True
                    break
        
        if not categorized:
            for keyword in fairness:
                if keyword in func.lower():
                    categories['Fairness Metrics'].append(func)
                    categorized = True
                    break
        
        if not categorized:
            categories['Other'].append(func)
    
    # Print categorized results
    for category, funcs in categories.items():
        if funcs:
            print(f"\n{category} ({len(funcs)} functions):")
            for func in funcs:
                print(f"  - {func}")
    
    # Create list for systematic testing
    print("\n" + "=" * 60)
    print("FUNCTIONS TO TEST (for copy-paste):")
    print("=" * 60)
    print("untested_functions = [")
    for func in untested:
        print(f"    '{func}',")
    print("]")
    
    return untested

if __name__ == "__main__":
    untested = main()
    print(f"\nTotal remaining: {len(untested)} functions")