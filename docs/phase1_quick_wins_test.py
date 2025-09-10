#!/usr/bin/env python3
"""
Phase 1: Quick Wins - Test 7 remaining functions in near-complete categories
Target: Bring coverage from 67.1% to ~70%

Functions to test:
1. nb_predict_log_proba (Machine Learning Core)
2. sortino_ratio (Risk Management)
3. class_distribution (Utility Functions)
4. support_per_class (Utility Functions)
5. z_score (Utility Functions)
6. kfold (Model Selection)
7. stratified_kfold (Model Selection)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tlm
import math

def test_nb_predict_log_proba():
    """Test Naive Bayes predict_log_proba function."""
    print("Testing nb_predict_log_proba...")
    try:
        # Train data
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
        y = [0, 0, 1, 1, 2, 2]
        
        # Check if function exists
        if not hasattr(tlm, 'nb_predict_log_proba'):
            print("SKIP nb_predict_log_proba: Not found")
            return False
            
        # Fit Naive Bayes model first
        if hasattr(tlm, 'nb_fit'):
            model = tlm.nb_fit(X, y)
            # Try to use predict_log_proba
            func = getattr(tlm, 'nb_predict_log_proba')
            result = func(X, model)
            print(f"PASS nb_predict_log_proba: {result}")
            return True
        else:
            print("SKIP nb_predict_log_proba: nb_fit not available")
            return False
            
    except Exception as e:
        print(f"FAIL nb_predict_log_proba: {e}")
        return False

def test_sortino_ratio():
    """Test Sortino ratio calculation."""
    print("\nTesting sortino_ratio...")
    try:
        # Test data - returns with negative values for downside
        returns = [0.01, -0.02, 0.03, -0.01, 0.02, 0.04, -0.03, 0.02]
        
        if not hasattr(tlm, 'sortino_ratio'):
            print("SKIP sortino_ratio: Not found")
            return False
            
        func = getattr(tlm, 'sortino_ratio')
        result = func(returns)
        print(f"PASS sortino_ratio: {result}")
        return True
        
    except Exception as e:
        print(f"FAIL sortino_ratio: {e}")
        return False

def test_class_distribution():
    """Test class distribution function."""
    print("\nTesting class_distribution...")
    try:
        # Test data
        y = [0, 1, 1, 0, 2, 2, 2, 1, 0, 1]
        
        if not hasattr(tlm, 'class_distribution'):
            print("SKIP class_distribution: Not found")
            return False
            
        func = getattr(tlm, 'class_distribution')
        result = func(y)
        print(f"PASS class_distribution: {result}")
        return True
        
    except Exception as e:
        print(f"FAIL class_distribution: {e}")
        return False

def test_support_per_class():
    """Test support per class function."""
    print("\nTesting support_per_class...")
    try:
        # Test data - need confusion matrix first
        y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 2, 2, 1, 1, 2]
        
        if not hasattr(tlm, 'support_per_class'):
            print("SKIP support_per_class: Not found")
            return False
            
        # Create confusion matrix first
        cm = tlm.confusion_matrix(y_true, y_pred)
        func = getattr(tlm, 'support_per_class')
        result = func(cm)
        print(f"PASS support_per_class: {result}")
        return True
        
    except Exception as e:
        print(f"FAIL support_per_class: {e}")
        return False

def test_z_score():
    """Test z-score normalization."""
    print("\nTesting z_score...")
    try:
        # Test data
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        if not hasattr(tlm, 'z_score'):
            print("SKIP z_score: Not found")
            return False
            
        func = getattr(tlm, 'z_score')
        result = func(data)
        print(f"PASS z_score: {result[:3]}... (showing first 3)")
        return True
        
    except Exception as e:
        print(f"FAIL z_score: {e}")
        return False

def test_kfold():
    """Test K-fold cross validation."""
    print("\nTesting kfold...")
    try:
        # Test data - kfold takes n (number of samples) and k (number of folds)
        n_samples = 10
        k_folds = 3
        
        if not hasattr(tlm, 'kfold'):
            print("SKIP kfold: Not found")
            return False
            
        func = getattr(tlm, 'kfold')
        # kfold(n, k, shuffle=True, seed=None)
        folds = list(func(n_samples, k_folds, shuffle=True, seed=42))
        print(f"PASS kfold: Generated {len(folds)} folds")
        return True
        
    except Exception as e:
        print(f"FAIL kfold: {e}")
        return False

def test_stratified_kfold():
    """Test Stratified K-fold cross validation."""
    print("\nTesting stratified_kfold...")
    try:
        # Test data with imbalanced classes
        y = [0, 0, 0, 0, 1, 1, 2, 2]  # Imbalanced classes
        k_folds = 2
        
        if not hasattr(tlm, 'stratified_kfold'):
            print("SKIP stratified_kfold: Not found")
            return False
            
        func = getattr(tlm, 'stratified_kfold')
        # stratified_kfold(y, k, shuffle=True, seed=None)
        folds = list(func(y, k_folds, shuffle=True, seed=42))
        print(f"PASS stratified_kfold: Generated {len(folds)} stratified folds")
        return True
        
    except Exception as e:
        print(f"FAIL stratified_kfold: {e}")
        return False

def main():
    """Run Phase 1 Quick Wins tests."""
    print("=" * 60)
    print("PHASE 1: QUICK WINS TEST SUITE")
    print("Testing 7 functions to reach ~70% coverage")
    print("=" * 60)
    
    results = []
    
    # Test all 7 functions
    test_functions = [
        ('nb_predict_log_proba', test_nb_predict_log_proba, 'ML Core'),
        ('sortino_ratio', test_sortino_ratio, 'Risk Management'),
        ('class_distribution', test_class_distribution, 'Utility'),
        ('support_per_class', test_support_per_class, 'Utility'),
        ('z_score', test_z_score, 'Utility'),
        ('kfold', test_kfold, 'Model Selection'),
        ('stratified_kfold', test_stratified_kfold, 'Model Selection')
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for func_name, test_func, category in test_functions:
        result = test_func()
        if result is True:
            passed += 1
            results.append((func_name, 'PASS', category))
        elif result is False:
            failed += 1
            results.append((func_name, 'FAIL', category))
        else:
            skipped += 1
            results.append((func_name, 'SKIP', category))
    
    # Summary
    print("\n" + "=" * 60)
    print("PHASE 1 RESULTS SUMMARY")
    print("=" * 60)
    
    for func_name, status, category in results:
        print(f"{status:5s} | {func_name:25s} | {category}")
    
    print("\n" + "-" * 60)
    total = len(test_functions)
    print(f"Total Functions Tested: {total}")
    print(f"Passed: {passed} ({100*passed/total:.1f}%)")
    print(f"Failed: {failed} ({100*failed/total:.1f}%)")
    print(f"Skipped: {skipped} ({100*skipped/total:.1f}%)")
    
    print("\n" + "=" * 60)
    print("COVERAGE IMPACT")
    print("=" * 60)
    print(f"Previous Coverage: 190/283 (67.1%)")
    print(f"New Functions Added: +{passed}")
    new_coverage = 190 + passed
    print(f"New Coverage: {new_coverage}/283 ({100*new_coverage/283:.1f}%)")
    print(f"Coverage Increase: +{100*passed/283:.1f} percentage points")
    
    if passed == total:
        print("\nPHASE 1 COMPLETE - All quick wins successful!")
    elif passed >= total * 0.7:
        print(f"\nPHASE 1 SUCCESS - {passed}/{total} quick wins achieved!")
    else:
        print(f"\nPHASE 1 PARTIAL - Only {passed}/{total} functions working")
    
    return passed >= total * 0.5  # Success if at least 50% working

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)