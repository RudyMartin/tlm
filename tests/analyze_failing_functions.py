#!/usr/bin/env python3
"""
Detailed analysis of the 21 failing functions to understand exact errors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tlm
import random
import math

def test_failing_functions():
    """Test each failing function to get detailed error messages."""
    
    # Test data setup
    prices = [100.0 + random.gauss(0, 5) for _ in range(50)]
    ohlc_data = []
    for i, p in enumerate(prices):
        if i == 0:
            open_price = p
        else:
            open_price = prices[i-1]
        high = p + abs(random.gauss(0, 2))
        low = p - abs(random.gauss(0, 2))
        close = p
        ohlc_data.append((open_price, high, low, close))
    
    volumes = [1000000 + random.randint(-200000, 500000) for _ in range(50)]
    
    failing_functions = [
        ('calculate_diversity_metrics', [[30.0, 30.0, 40.0]]),
        ('continuous_wavelet_transform', [[math.sin(2*math.pi*i/20) for i in range(100)]]),
        ('ensemble_strategy', [prices]),
        ('fixed_fractional', [prices]),
        ('gap_trading', [ohlc_data]),
        ('ichimoku_cloud_strategy', [ohlc_data]),
        ('keltner_channels', [ohlc_data]),
        ('momentum_strategy', [prices]),
        ('multiplicative_decompose', [[10 + 5*math.sin(2*math.pi*i/20) for i in range(100)]]),
        ('ornstein_uhlenbeck_reversion', [prices]),
        ('parabolic_sar_strategy', [ohlc_data]),
        ('reinforcement_learning_trader', [prices]),
        ('relative_strength_strategy', [prices]),
        ('robust_stl_decompose', [[math.sin(2*math.pi*i/20) for i in range(100)]]),
        ('seasonal_strength', [[math.sin(2*math.pi*i/20) for i in range(100)]]),
        ('statistical_arbitrage', [prices, [p + random.gauss(0, 2) for p in prices]]),
        ('stl_decompose', [[math.sin(2*math.pi*i/20) for i in range(100)]]),
        ('triangle_breakout', [prices]),
        ('turtle_trading', [ohlc_data]),
        ('wavelet_coherence', [[math.sin(2*math.pi*i/20) for i in range(100)], [math.cos(2*math.pi*i/20) for i in range(100)]]),
        ('wavelet_cross_correlation', [[math.sin(2*math.pi*i/20) for i in range(100)], [math.cos(2*math.pi*i/20) for i in range(100)]]),
        ('wavelet_reconstruct', [None]),  # Special case
    ]
    
    print("DETAILED ERROR ANALYSIS FOR 21 FAILING FUNCTIONS")
    print("=" * 80)
    
    for func_name, test_args in failing_functions:
        print(f"\n{func_name}:")
        print("-" * 50)
        
        try:
            if not hasattr(tlm, func_name):
                print("X FUNCTION NOT FOUND")
                continue
            
            func = getattr(tlm, func_name)
            
            # Special handling for wavelet_reconstruct
            if func_name == 'wavelet_reconstruct':
                signal = [math.sin(2*math.pi*i/20) for i in range(100)]
                try:
                    coeffs = tlm.discrete_wavelet_transform(signal, 'db4')
                    result = func(coeffs, 'db4')
                    print("OK WORKS with proper coefficients")
                except Exception as e:
                    print(f"X ERROR: {e}")
                continue
            
            # Special handling for seasonal_strength
            if func_name == 'seasonal_strength':
                signal = [math.sin(2*math.pi*i/20) for i in range(100)]
                try:
                    result = func(signal, 20)
                    print("OK WORKS with proper parameters")
                except Exception as e:
                    print(f"X ERROR: {e}")
                continue
            
            # Test the function
            result = func(*test_args)
            print("OK ACTUALLY WORKS - May have been misclassified")
            
        except TypeError as e:
            print(f"X TYPE ERROR: {e}")
        except ValueError as e:
            print(f"X VALUE ERROR: {e}")
        except AttributeError as e:
            print(f"X ATTRIBUTE ERROR: {e}")
        except Exception as e:
            print(f"X OTHER ERROR ({type(e).__name__}): {e}")

def analyze_specific_errors():
    """Analyze specific patterns in the errors."""
    print("\n" + "=" * 80)
    print("SPECIFIC ERROR PATTERN ANALYSIS")
    print("=" * 80)
    
    # 1. Test calculate_diversity_metrics generator issue
    print("\n1. CALCULATE_DIVERSITY_METRICS - Generator Issue")
    print("-" * 50)
    try:
        composition = [30.0, 30.0, 40.0]
        result = tlm.calculate_diversity_metrics(composition)
        print(f"OK Works: {result}")
    except Exception as e:
        print(f"X Error: {e}")
        print("INFO This is the asum(generator) bug we identified")
    
    # 2. Test wavelet functions
    print("\n2. WAVELET FUNCTIONS - Parameter Issues")
    print("-" * 50)
    signal = [math.sin(2*math.pi*i/20) for i in range(100)]
    
    wavelet_funcs = ['continuous_wavelet_transform', 'wavelet_coherence', 'wavelet_cross_correlation']
    for func_name in wavelet_funcs:
        if hasattr(tlm, func_name):
            try:
                func = getattr(tlm, func_name)
                if func_name == 'continuous_wavelet_transform':
                    result = func(signal, 'morlet')
                elif func_name in ['wavelet_coherence', 'wavelet_cross_correlation']:
                    signal2 = [math.cos(2*math.pi*i/20) for i in range(100)]
                    result = func(signal, signal2, 'morlet')
                print(f"OK {func_name}: Works with proper parameters")
            except Exception as e:
                print(f"X {func_name}: {e}")
    
    # 3. Test trading strategy functions
    print("\n3. TRADING STRATEGIES - Float/Index Issues")
    print("-" * 50)
    prices = [100.0 + i * 0.5 for i in range(50)]
    ohlc_data = [(100+i, 102+i, 98+i, 101+i) for i in range(50)]
    
    trading_funcs = ['momentum_strategy', 'triangle_breakout', 'turtle_trading']
    for func_name in trading_funcs:
        if hasattr(tlm, func_name):
            try:
                func = getattr(tlm, func_name)
                if func_name in ['momentum_strategy', 'triangle_breakout']:
                    result = func(prices, 20)
                else:  # turtle_trading
                    result = func(ohlc_data, 20, 10)
                print(f"OK {func_name}: Works with proper parameters")
            except Exception as e:
                print(f"X {func_name}: {e}")
    
    # 4. Test signal processing decomposition
    print("\n4. SIGNAL PROCESSING - Decomposition Issues")  
    print("-" * 50)
    signal = [10 + 5*math.sin(2*math.pi*i/20) + random.gauss(0, 0.5) for i in range(100)]
    
    decomp_funcs = ['multiplicative_decompose', 'stl_decompose', 'robust_stl_decompose']
    for func_name in decomp_funcs:
        if hasattr(tlm, func_name):
            try:
                func = getattr(tlm, func_name)
                result = func(signal, 20)
                print(f"OK {func_name}: Works with proper parameters")
            except Exception as e:
                print(f"X {func_name}: {e}")

if __name__ == "__main__":
    test_failing_functions()
    analyze_specific_errors()