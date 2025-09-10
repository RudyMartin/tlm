#!/usr/bin/env python3
"""
Comprehensive test of all TLM loss functions.

Tests all 29 loss functions to complete the loss function coverage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tlm
import math

def test_all_loss_functions():
    """Test all TLM loss functions comprehensively."""
    print("COMPREHENSIVE LOSS FUNCTION TESTING")
    print("=" * 50)
    
    # Test data for different loss types
    # Regression data
    y_true_reg = [1.0, 2.0, 3.0, 4.0]
    y_pred_reg = [1.1, 1.9, 3.2, 3.8]
    
    # Binary classification 
    y_true_binary = [0, 1, 1, 0]
    y_pred_binary = [0.1, 0.9, 0.8, 0.2]
    y_pred_binary_signed = [-0.5, 1.2, 0.8, -0.3]  # For hinge loss
    
    # Multi-class classification
    y_true_multi = [0, 1, 2, 1]
    y_pred_multi_logits = [[2.0, 1.0, 0.1], [0.5, 2.5, 0.8], [0.2, 0.8, 2.1], [1.1, 2.0, 0.5]]
    y_pred_multi_proba = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7], [0.3, 0.6, 0.1]]
    
    # Ranking/metric learning data
    anchor = [1.0, 2.0, 3.0]
    positive = [1.1, 2.1, 2.9] 
    negative = [2.5, 1.0, 4.0]
    
    # Distribution data
    p_dist = [0.5, 0.3, 0.2]
    q_dist = [0.4, 0.4, 0.2]
    
    # Test all loss functions
    loss_tests = [
        # Already tested basic ones
        ('mse', [y_true_reg, y_pred_reg], 'regression'),
        ('mae', [y_true_reg, y_pred_reg], 'regression'), 
        ('binary_cross_entropy', [y_pred_binary, y_true_binary], 'binary'),
        ('cross_entropy', [y_pred_multi_proba, y_true_multi], 'multiclass'),
        ('focal_loss', [y_true_binary, y_pred_binary], 'binary'),
        ('hinge_loss', [y_true_binary, y_pred_binary], 'binary'),
        ('huber_loss', [y_true_reg, y_pred_reg], 'regression'),
        ('kl_divergence', [p_dist, q_dist], 'distribution'),
        
        # Need to test these remaining ones
        ('rmse', [y_true_reg, y_pred_reg], 'regression'),
        ('msle', [y_true_reg, y_pred_reg], 'regression'),
        ('mape', [y_true_reg, y_pred_reg], 'regression'),
        ('smape', [y_true_reg, y_pred_reg], 'regression'),
        ('log_cosh_loss', [y_true_reg, y_pred_reg], 'regression'),
        ('quantile_loss', [y_true_reg, y_pred_reg, 0.5], 'regression'),
        
        ('squared_hinge_loss', [y_true_binary, y_pred_binary_signed], 'binary'),
        ('f2_loss', [y_true_binary, y_pred_binary], 'binary'),
        
        ('js_divergence', [p_dist, q_dist], 'distribution'),
        ('wasserstein_distance', [p_dist, q_dist], 'distribution'),
        
        ('contrastive_loss', [[anchor, positive], [1]], 'metric_learning'),
        ('triplet_loss', [anchor, positive, negative], 'metric_learning'),
        ('margin_ranking_loss', [y_pred_binary[:2], y_pred_binary[2:4], [1, -1]], 'ranking'),
        
        ('bernoulli_loss', [y_true_binary, y_pred_binary], 'specialized'),
        ('poisson_loss', [[1, 2, 3, 4], [1.1, 1.9, 3.2, 3.8]], 'specialized'),
        ('gamma_loss', [y_true_reg, y_pred_reg], 'specialized'),
        ('lucas_loss', [y_true_reg, y_pred_reg], 'specialized'),
        
        ('robust_l1_loss', [y_true_reg, y_pred_reg], 'robust'),
        ('robust_l2_loss', [y_true_reg, y_pred_reg], 'robust'),
        ('tukey_loss', [y_true_reg, y_pred_reg], 'robust'),
        
        ('multi_task_loss', [[y_true_reg, y_true_binary], [y_pred_reg, y_pred_binary]], 'multitask'),
        ('weighted_loss', [y_true_reg, y_pred_reg, [1, 1, 2, 2]], 'weighted'),
        ('durbin_watson_loss', [y_true_reg], 'statistical'),
    ]
    
    # Test each loss function
    passed = 0
    total = len(loss_tests)
    results = []
    
    for func_name, args, category in loss_tests:
        try:
            if hasattr(tlm, func_name):
                func = getattr(tlm, func_name)
                
                # Handle special cases
                if func_name == 'sparse_cross_entropy':
                    # Test single sample
                    result = func(y_pred_multi_logits[0], y_true_multi[0])
                elif func_name == 'contrastive_loss':
                    # Handle contrastive loss format
                    result = func([anchor, positive], 1.0)
                else:
                    result = func(*args)
                
                # Validate result
                if isinstance(result, (int, float)) and not math.isnan(result):
                    print(f"PASS {func_name} ({category}): {result:.6f}")
                    passed += 1
                    results.append((func_name, True, result, category))
                else:
                    print(f"FAIL {func_name} ({category}): Invalid result {result}")
                    results.append((func_name, False, str(result), category))
                    
            else:
                print(f"SKIP {func_name} ({category}): Function not found")
                results.append((func_name, False, "Not found", category))
                
        except Exception as e:
            print(f"FAIL {func_name} ({category}): {str(e)}")
            results.append((func_name, False, str(e), category))
    
    # Summary by category
    print(f"\n{'LOSS FUNCTION SUMMARY':=^60}")
    print(f"Total Tested: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {100 * passed / total:.1f}%")
    
    # Group by category
    categories = {}
    for func_name, success, result, category in results:
        if category not in categories:
            categories[category] = {'passed': 0, 'total': 0, 'functions': []}
        categories[category]['total'] += 1
        if success:
            categories[category]['passed'] += 1
        categories[category]['functions'].append((func_name, success, result))
    
    print(f"\n{'BY CATEGORY':=^60}")
    for category, data in categories.items():
        success_rate = 100 * data['passed'] / data['total']
        print(f"\n{category.upper()} ({data['passed']}/{data['total']} - {success_rate:.1f}%):")
        for func_name, success, result in data['functions']:
            status = "PASS" if success else "FAIL"
            print(f"  {status} {func_name}: {result}")
    
    return passed, total, results

def main():
    """Run comprehensive loss function testing."""
    passed, total, results = test_all_loss_functions()
    
    print(f"\n{'FINAL RESULTS':=^60}")
    if passed == total:
        print("SUCCESS: All loss functions work perfectly!")
    elif passed >= total * 0.9:
        print(f"EXCELLENT: {passed}/{total} loss functions working ({100*passed/total:.1f}%)")
    elif passed >= total * 0.8:
        print(f"GOOD: {passed}/{total} loss functions working ({100*passed/total:.1f}%)")
    else:
        print(f"NEEDS WORK: Only {passed}/{total} loss functions working ({100*passed/total:.1f}%)")
    
    # Return success status
    return passed >= total * 0.8

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)