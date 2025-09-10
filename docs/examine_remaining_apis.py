#!/usr/bin/env python3
"""
Examine the API signatures of remaining untested functions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tlm
import inspect

def examine_function_apis():
    """Examine API signatures of untested functions."""
    
    untested_functions = [
        'assortativity_coefficient',
        'average_path_length',
        'average_true_range',
        'awesome_oscillator',
        'bayesian_network_strategy',
        'black_litterman',
        'breakout_strategy',
        'calculate_diversity_metrics',
        'commodity_channel_index',
        'connected_components',
        'continuous_wavelet_transform',
        'degree_distribution',
        'diameter',
        'eigenvector_centrality',
        'ensemble_strategy',
        'fixed_fractional',
        'gap_trading',
        'global_clustering_coefficient',
        'greedy_modularity_communities',
        'ichimoku_cloud_strategy',
        'kaufman_adaptive_moving_average',
        'keltner_channels',
        'louvain_communities',
        'modularity',
        'momentum_strategy',
        'money_flow_index',
        'multiplicative_decompose',
        'ornstein_uhlenbeck_reversion',
        'pagerank',
        'parabolic_sar_strategy',
        'reinforcement_learning_trader',
        'relative_strength_strategy',
        'remainder_strength',
        'robust_stl_decompose',
        'seasonal_strength',
        'shortest_path_length',
        'statistical_arbitrage',
        'stl_decompose',
        'transitivity',
        'trend_strength',
        'triangle_breakout',
        'turtle_trading',
        'volatility_position_sizing',
        'volume_weighted_average_price',
        'wavelet_coherence',
        'wavelet_cross_correlation',
        'wavelet_reconstruct',
        'weighted_moving_average',
        'williams_r',
    ]
    
    print("API SIGNATURES FOR UNTESTED FUNCTIONS")
    print("=" * 60)
    
    for func_name in untested_functions[:10]:  # Test first 10
        if hasattr(tlm, func_name):
            func = getattr(tlm, func_name)
            try:
                sig = inspect.signature(func)
                print(f"\n{func_name}:")
                print(f"  Signature: {sig}")
                
                # Get docstring if available
                if func.__doc__:
                    first_line = func.__doc__.split('\n')[0].strip()
                    print(f"  Doc: {first_line[:80]}")
            except Exception as e:
                print(f"\n{func_name}: Error getting signature - {e}")
        else:
            print(f"\n{func_name}: NOT FOUND")
    
    return untested_functions

if __name__ == "__main__":
    examine_function_apis()