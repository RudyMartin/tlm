#!/usr/bin/env python3
"""
Phase 2: Medium Priority - Test 32 functions across business-critical domains
Target: Bring coverage from 69.6% to ~81.3%

Categories:
- Fairness & Bias Metrics (3 functions)
- Trading Indicators (16 functions)
- Portfolio Management (6 functions)
- Trading Strategies (7 functions)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tlm
import random

def test_fairness_metrics():
    """Test fairness and bias metrics (3 functions)."""
    print("=" * 60)
    print("TESTING FAIRNESS & BIAS METRICS (3 functions)")
    print("=" * 60)
    
    results = []
    
    # Test data for fairness metrics
    y_true = [0, 1, 1, 0, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 0, 1, 1, 1, 0]
    sensitive_attr = [0, 0, 1, 1, 0, 1, 1, 0]  # Protected attribute
    
    # 1. calculate_fairness_metrics
    try:
        if hasattr(tlm, 'calculate_fairness_metrics'):
            result = tlm.calculate_fairness_metrics(y_true, y_pred, sensitive_attr)
            print(f"PASS calculate_fairness_metrics: {result}")
            results.append(('calculate_fairness_metrics', True))
        else:
            print("SKIP calculate_fairness_metrics: Not found")
            results.append(('calculate_fairness_metrics', False))
    except Exception as e:
        print(f"FAIL calculate_fairness_metrics: {e}")
        results.append(('calculate_fairness_metrics', False))
    
    # 2. calculate_diversity_metrics
    try:
        if hasattr(tlm, 'calculate_diversity_metrics'):
            # Diversity in predictions or features
            data = [[1, 2, 3], [1, 2, 4], [2, 3, 4], [1, 1, 1]]
            result = tlm.calculate_diversity_metrics(data)
            print(f"PASS calculate_diversity_metrics: {result}")
            results.append(('calculate_diversity_metrics', True))
        else:
            print("SKIP calculate_diversity_metrics: Not found")
            results.append(('calculate_diversity_metrics', False))
    except Exception as e:
        print(f"FAIL calculate_diversity_metrics: {e}")
        results.append(('calculate_diversity_metrics', False))
    
    # 3. analyze_missing_data
    try:
        if hasattr(tlm, 'analyze_missing_data'):
            # Data with None values
            data = [[1, None, 3], [4, 5, None], [7, 8, 9], [None, 11, 12]]
            result = tlm.analyze_missing_data(data)
            print(f"PASS analyze_missing_data: {result}")
            results.append(('analyze_missing_data', True))
        else:
            print("SKIP analyze_missing_data: Not found")
            results.append(('analyze_missing_data', False))
    except Exception as e:
        print(f"FAIL analyze_missing_data: {e}")
        results.append(('analyze_missing_data', False))
    
    passed = sum(1 for _, success in results if success)
    print(f"\nFairness Metrics: {passed}/{len(results)} functions working")
    return results

def test_trading_indicators():
    """Test trading indicators (16 functions)."""
    print("\n" + "=" * 60)
    print("TESTING TRADING INDICATORS (16 functions)")
    print("=" * 60)
    
    results = []
    
    # Generate sample price data
    prices = [100.0 + random.gauss(0, 5) for _ in range(50)]
    high = [p + abs(random.gauss(0, 2)) for p in prices]
    low = [p - abs(random.gauss(0, 2)) for p in prices]
    volume = [1000000 + random.randint(-200000, 500000) for _ in range(50)]
    
    indicators = [
        ('weighted_moving_average', [prices, [1, 2, 3, 4, 5, 4, 3, 2, 1][:len(prices)]]),
        ('hull_moving_average', [prices, 10]),
        ('kaufman_adaptive_moving_average', [prices, 10, 2, 30]),
        ('keltner_channels', [high, low, prices, 20, 2]),
        ('awesome_oscillator', [high, low]),
        ('commodity_channel_index', [high, low, prices, 20]),
        ('williams_r', [high, low, prices, 14]),
        ('money_flow_index', [high, low, prices, volume, 14]),
        ('average_true_range', [high, low, prices, 14]),
        ('rate_of_change', [prices, 10]),
        ('momentum', [prices, 10]),
        ('on_balance_volume', [prices, volume]),
        ('volume_weighted_average_price', [high, low, prices, volume]),
        ('head_shoulders_pattern', [prices]),
        ('double_top_bottom', [prices]),
        ('triangle_breakout', [prices])
    ]
    
    for func_name, test_args in indicators:
        try:
            if hasattr(tlm, func_name):
                func = getattr(tlm, func_name)
                result = func(*test_args)
                print(f"PASS {func_name}: Works")
                results.append((func_name, True))
            else:
                print(f"SKIP {func_name}: Not found")
                results.append((func_name, False))
        except Exception as e:
            print(f"FAIL {func_name}: {str(e)[:50]}...")
            results.append((func_name, False))
    
    passed = sum(1 for _, success in results if success)
    print(f"\nTrading Indicators: {passed}/{len(results)} functions working")
    return results

def test_portfolio_management():
    """Test portfolio management functions (6 functions)."""
    print("\n" + "=" * 60)
    print("TESTING PORTFOLIO MANAGEMENT (6 functions)")
    print("=" * 60)
    
    results = []
    
    # Sample portfolio data
    returns = [[0.01, -0.02, 0.03], [0.02, 0.01, -0.01], 
               [0.03, -0.01, 0.02], [-0.01, 0.02, 0.01]]
    cov_matrix = [[0.01, 0.005, 0.002], [0.005, 0.02, 0.001], [0.002, 0.001, 0.015]]
    expected_returns = [0.08, 0.10, 0.12]
    
    portfolio_funcs = [
        ('modern_portfolio_theory', [expected_returns, cov_matrix]),
        ('black_litterman', [expected_returns, cov_matrix, [0.09, 0.11, 0.13]]),
        ('risk_parity', [cov_matrix]),
        ('fixed_fractional', [1000000, 0.02, 50000]),
        ('volatility_position_sizing', [1000000, 0.02, 0.15]),
        ('bayesian_network_strategy', [returns])
    ]
    
    for func_name, test_args in portfolio_funcs:
        try:
            if hasattr(tlm, func_name):
                func = getattr(tlm, func_name)
                result = func(*test_args)
                print(f"PASS {func_name}: {result}")
                results.append((func_name, True))
            else:
                print(f"SKIP {func_name}: Not found")
                results.append((func_name, False))
        except Exception as e:
            print(f"FAIL {func_name}: {str(e)[:50]}...")
            results.append((func_name, False))
    
    passed = sum(1 for _, success in results if success)
    print(f"\nPortfolio Management: {passed}/{len(results)} functions working")
    return results

def test_trading_strategies():
    """Test trading strategies (7 selected functions)."""
    print("\n" + "=" * 60)
    print("TESTING TRADING STRATEGIES (7 functions)")
    print("=" * 60)
    
    results = []
    
    # Generate sample data
    prices = [100.0 + i * 0.5 + random.gauss(0, 2) for i in range(100)]
    prices2 = [100.0 + i * 0.3 + random.gauss(0, 2) for i in range(100)]  # Correlated asset
    volume = [1000000 + random.randint(-200000, 500000) for _ in range(100)]
    
    strategies = [
        ('momentum_strategy', [prices, 20, 0.02]),
        ('pairs_trading', [prices, prices2, 30, 2.0]),
        ('turtle_trading', [prices, 20, 10]),
        ('breakout_strategy', [prices, 20, 1.5]),
        ('gap_trading', [prices, 0.02]),
        ('bollinger_bands_reversion', [prices, 20, 2]),
        ('rsi_mean_reversion', [prices, 14, 30, 70])
    ]
    
    for func_name, test_args in strategies:
        try:
            if hasattr(tlm, func_name):
                func = getattr(tlm, func_name)
                result = func(*test_args)
                print(f"PASS {func_name}: Generated signals")
                results.append((func_name, True))
            else:
                print(f"SKIP {func_name}: Not found")
                results.append((func_name, False))
        except Exception as e:
            print(f"FAIL {func_name}: {str(e)[:50]}...")
            results.append((func_name, False))
    
    passed = sum(1 for _, success in results if success)
    print(f"\nTrading Strategies: {passed}/{len(results)} functions working")
    return results

def main():
    """Run Phase 2 Medium Priority tests."""
    print("=" * 60)
    print("PHASE 2: MEDIUM PRIORITY TEST SUITE")
    print("Testing 32 functions to reach ~81% coverage")
    print("=" * 60)
    
    all_results = []
    
    # Run all category tests
    all_results.extend(test_fairness_metrics())
    all_results.extend(test_trading_indicators())
    all_results.extend(test_portfolio_management())
    all_results.extend(test_trading_strategies())
    
    # Calculate summary
    passed = sum(1 for _, success in all_results if success)
    total = len(all_results)
    
    print("\n" + "=" * 60)
    print("PHASE 2 RESULTS SUMMARY")
    print("=" * 60)
    
    # Group by category
    categories = {
        'Fairness & Bias': 3,
        'Trading Indicators': 16,
        'Portfolio Management': 6,
        'Trading Strategies': 7
    }
    
    start_idx = 0
    for category, count in categories.items():
        cat_results = all_results[start_idx:start_idx + count]
        cat_passed = sum(1 for _, success in cat_results if success)
        print(f"{category:20s}: {cat_passed:2d}/{count:2d} ({100*cat_passed/count:5.1f}%)")
        start_idx += count
    
    print("-" * 60)
    print(f"Total Functions Tested: {total}")
    print(f"Passed: {passed} ({100*passed/total:.1f}%)")
    print(f"Failed: {total - passed} ({100*(total - passed)/total:.1f}%)")
    
    print("\n" + "=" * 60)
    print("COVERAGE IMPACT")
    print("=" * 60)
    print(f"Previous Coverage: 197/283 (69.6%)")
    print(f"New Functions Added: +{passed}")
    new_coverage = 197 + passed
    print(f"New Coverage: {new_coverage}/283 ({100*new_coverage/283:.1f}%)")
    print(f"Coverage Increase: +{100*passed/283:.1f} percentage points")
    
    if passed >= 25:
        print("\nPHASE 2 SUCCESS - Excellent progress!")
    elif passed >= 16:
        print("\nPHASE 2 GOOD - Significant functions added!")
    else:
        print(f"\nPHASE 2 PARTIAL - {passed} functions added")
    
    print("\nNext: Run Phase 3 for advanced specialized functions")
    
    return passed >= 16  # Success if at least 50% working

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)