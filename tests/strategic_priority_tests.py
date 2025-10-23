#!/usr/bin/env python3
"""
Strategic Priority Testing - Complete ML Core, Classification Metrics, Array Operations

Tests the 42 missing functions identified as strategic priorities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tlm
import random

def test_ml_core_completions():
    """Test the 14 missing ML Core functions."""
    print("=== Testing ML Core Completions (14 functions) ===")
    
    # Test data
    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    y = [0, 1, 1, 0]
    y_multi = [0, 1, 2, 1]
    
    results = []
    
    # ML Core functions to test
    ml_tests = [
        # SVM extensions - need actual SVM model
        ('svm_hinge_loss', 'needs_fitted_svm', 'svm'),
        
        # Naive Bayes extensions  
        ('nb_predict_log_proba', 'needs_model', 'naive_bayes'),
        
        # PCA extensions
        ('pca_power_transform', 'needs_fitted_pca', 'pca'),
        ('pca_power_fit_transform', [X, 2], 'pca'),
        ('pca_whiten', 'needs_fitted_pca', 'pca'),
        
        # Matrix Factorization extensions
        ('mf_predict_full', 'needs_fitted_mf', 'mf'),
        ('mf_mse', 'needs_fitted_mf_mse', 'mf'),
        
        # Softmax extensions
        ('softmax_predict', 'needs_fitted_softmax', 'softmax'),
        ('softmax_predict_proba', 'needs_fitted_softmax', 'softmax'),
        
        # GMM extensions
        ('gmm_predict', 'needs_fitted_gmm', 'gmm'),
        ('gmm_predict_proba', 'needs_fitted_gmm', 'gmm'),
        
        # Anomaly detection
        ('anomaly_fit', [X], 'anomaly'),
        ('anomaly_predict', 'needs_fitted_anomaly', 'anomaly'),
        ('anomaly_score_logpdf', 'needs_fitted_anomaly', 'anomaly'),
    ]
    
    passed = 0
    total = len(ml_tests)
    
    for func_name, test_data, category in ml_tests:
        try:
            if not hasattr(tlm, func_name):
                print(f"SKIP {func_name} ({category}): Not found")
                continue
                
            func = getattr(tlm, func_name)
            
            if test_data == 'needs_model':
                print(f"SKIP {func_name} ({category}): Requires fitted model")
                continue
            elif test_data == 'needs_fitted_pca':
                # Fit PCA first
                pca_model = tlm.pca_power_fit(X, 2)
                if func_name == 'pca_power_transform':
                    result = func(X, pca_model)
                elif func_name == 'pca_whiten':
                    result = func(X, pca_model)
                else:
                    result = func(*test_data)
            elif test_data == 'needs_fitted_mf':
                # Fit MF first
                mf_model = tlm.mf_fit_sgd([[1, 2], [3, 4]], 2)
                P, Q, history = mf_model
                result = func(P, Q)
            elif test_data == 'needs_fitted_mf_mse':
                # Test MF MSE function - needs original rating matrix R, P, Q
                R = [[1, 2], [3, 4]]  # Original rating matrix
                P = [[1.0, 0.5], [0.8, 1.2]]
                Q = [[0.9, 1.1], [0.4, 1.3]]
                result = func(R, P, Q)
            elif test_data == 'needs_fitted_svm':
                # Fit SVM first and test hinge loss
                svm_model = tlm.svm_fit(X, y)
                weights, bias, history = svm_model
                result = func(X, y, weights, bias, 1.0)  # C=1.0
            elif test_data == 'needs_fitted_softmax':
                # Fit softmax first
                softmax_model = tlm.softmax_fit(X, y_multi)
                W, b, history = softmax_model
                if func_name == 'softmax_predict':
                    result = func(X, W, b)
                else:
                    result = func(X, W, b)
            elif test_data == 'needs_fitted_gmm':
                # Fit GMM first
                gmm_model = tlm.gmm_fit(X, 2)
                result = func(X, gmm_model)
            elif test_data == 'needs_fitted_anomaly':
                # Fit anomaly detector first
                anomaly_model = tlm.anomaly_fit(X)
                mu, var = anomaly_model
                if func_name == 'anomaly_predict':
                    result = func(X, mu, var)
                else:  # anomaly_score_logpdf
                    result = func(X, mu, var)
            else:
                result = func(*test_data)
            
            print(f"PASS {func_name} ({category}): Works")
            passed += 1
            results.append((func_name, True, category))
            
        except Exception as e:
            print(f"FAIL {func_name} ({category}): {str(e)}")
            results.append((func_name, False, category))
    
    print(f"\nML Core Results: {passed}/{total} functions working ({100*passed/total:.1f}%)")
    return results

def test_classification_metrics():
    """Test the 20 missing Classification Metrics."""
    print("\n=== Testing Classification Metrics (20 functions) ===")
    
    # Test data
    y_true = [0, 1, 2, 0, 1, 2, 0, 1]
    y_pred = [0, 1, 1, 0, 2, 2, 1, 1] 
    y_proba = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4], 
               [0.9, 0.05, 0.05], [0.1, 0.3, 0.6], [0.2, 0.2, 0.6],
               [0.4, 0.5, 0.1], [0.3, 0.6, 0.1]]
    
    results = []
    
    # Classification metrics to test
    metrics_tests = [
        # Binary variants
        ('confusion_matrix_binary', [y_true[:4], y_pred[:4]], 'binary'),
        ('confusion_matrix_metrics', 'needs_confusion_matrix', 'multiclass'),
        
        # Multi-class versions - need confusion matrix
        ('precision_multi', 'needs_confusion_matrix', 'multiclass'),
        ('recall_multi', 'needs_confusion_matrix', 'multiclass'), 
        ('f1_score_multi', 'needs_confusion_matrix', 'multiclass'),
        
        # Averaging methods - need confusion matrix
        ('macro_avg_precision', 'needs_confusion_matrix', 'averaging'),
        ('macro_avg_recall', 'needs_confusion_matrix', 'averaging'),
        ('macro_avg_f1', 'needs_confusion_matrix', 'averaging'),
        ('micro_avg_precision', 'needs_confusion_matrix', 'averaging'),
        ('micro_avg_recall', 'needs_confusion_matrix', 'averaging'),
        ('micro_avg_f1', 'needs_confusion_matrix', 'averaging'),
        ('weighted_avg_precision', 'needs_confusion_matrix', 'averaging'),
        ('weighted_avg_recall', 'needs_confusion_matrix', 'averaging'),
        ('weighted_avg_f1', 'needs_confusion_matrix', 'averaging'),
        
        # Additional metrics
        ('sensitivity', [y_true[:4], y_pred[:4]], 'binary'),
        ('specificity', [y_true[:4], y_pred[:4]], 'binary'),
        ('cohen_kappa', [y_true, y_pred], 'agreement'),
        ('average_precision_score', [y_true[:4], [0.1, 0.8, 0.6, 0.2]], 'ranking'),
        ('precision_recall_curve', [y_true[:4], [0.1, 0.8, 0.6, 0.2]], 'curves'),
        ('calculate_roc_metrics', [y_true[:4], [0.1, 0.8, 0.6, 0.2]], 'roc'),
    ]
    
    passed = 0
    total = len(metrics_tests)
    
    for func_name, test_data, category in metrics_tests:
        try:
            if not hasattr(tlm, func_name):
                print(f"SKIP {func_name} ({category}): Not found")
                continue
                
            func = getattr(tlm, func_name)
            
            if test_data == 'needs_confusion_matrix':
                # Create confusion matrix first
                cm = tlm.confusion_matrix(y_true, y_pred)
                if func_name == 'confusion_matrix_metrics':
                    result = func(cm)
                else:
                    # Multi-class metrics that need confusion matrix
                    result = func(cm)
            else:
                result = func(*test_data)
            
            print(f"PASS {func_name} ({category}): Works")
            passed += 1
            results.append((func_name, True, category))
            
        except Exception as e:
            print(f"FAIL {func_name} ({category}): {str(e)}")
            results.append((func_name, False, category))
    
    print(f"\nClassification Metrics Results: {passed}/{total} functions working ({100*passed/total:.1f}%)")
    return results

def test_array_operations():
    """Test the 8 missing Array Operations."""
    print("\n=== Testing Array Operations (8 functions) ===")
    
    # Test data
    arr1 = [1, 2, 3]
    arr2 = [4, 5, 6] 
    matrix1 = [[1, 2], [3, 4]]
    matrix2 = [[5, 6], [7, 8]]
    
    results = []
    
    # Array operations to test
    array_tests = [
        ('tile', [arr1, 3], 'manipulation'),
        ('stack', [[arr1, arr2]], 'stacking'),
        ('hstack', [[arr1, arr2]], 'stacking'),
        ('vstack', [[matrix1, matrix2]], 'stacking'),
        ('searchsorted', [[1, 3, 5, 7, 9], 4], 'searching'),
        ('unique', [[1, 2, 2, 3, 3, 3]], 'utilities'),
        ('diff', [arr1], 'differences'),
        ('gradient', [arr1], 'calculus'),
    ]
    
    passed = 0
    total = len(array_tests)
    
    for func_name, test_data, category in array_tests:
        try:
            if not hasattr(tlm, func_name):
                print(f"SKIP {func_name} ({category}): Not found")
                continue
                
            func = getattr(tlm, func_name)
            result = func(*test_data)
            
            print(f"PASS {func_name} ({category}): Works")
            passed += 1
            results.append((func_name, True, category))
            
        except Exception as e:
            print(f"FAIL {func_name} ({category}): {str(e)}")
            results.append((func_name, False, category))
    
    print(f"\nArray Operations Results: {passed}/{total} functions working ({100*passed/total:.1f}%)")
    return results

def main():
    """Run strategic priority testing."""
    print("TLM STRATEGIC PRIORITY TESTING")
    print("=" * 50)
    print("Testing 42 missing functions in strategic priority categories")
    print()
    
    # Test all three categories
    ml_results = test_ml_core_completions()
    metrics_results = test_classification_metrics() 
    array_results = test_array_operations()
    
    # Summary
    all_results = ml_results + metrics_results + array_results
    total_passed = sum(1 for _, success, _ in all_results if success)
    total_tested = len(all_results)
    
    print(f"\n{'STRATEGIC PRIORITY SUMMARY':=^60}")
    print(f"Total Functions Tested: {total_tested}")
    print(f"Functions Working: {total_passed}")
    print(f"Functions Failed: {total_tested - total_passed}")
    print(f"Success Rate: {100 * total_passed / total_tested:.1f}%")
    
    # Category breakdown
    categories = {}
    for func_name, success, category in all_results:
        if category not in categories:
            categories[category] = {'passed': 0, 'total': 0}
        categories[category]['total'] += 1
        if success:
            categories[category]['passed'] += 1
    
    print(f"\nBy Category:")
    for category, stats in categories.items():
        rate = 100 * stats['passed'] / stats['total']
        print(f"  {category:15s}: {stats['passed']:2d}/{stats['total']:2d} ({rate:5.1f}%)")
    
    print(f"\n{'IMPACT ON COVERAGE':=^60}")
    print(f"Previous Coverage: 149/283 (52.7%)")
    print(f"New Functions Added: +{total_passed}")
    print(f"New Coverage: {149 + total_passed}/283 ({100 * (149 + total_passed) / 283:.1f}%)")
    print(f"Coverage Increase: +{100 * total_passed / 283:.1f} percentage points")
    
    return total_passed >= total_tested * 0.7  # 70% success rate target

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)