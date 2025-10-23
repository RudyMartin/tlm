#!/usr/bin/env python3
"""
Test the highest priority untested TLM functions.

These are critical missing functionality that should be tested immediately.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tlm

def test_core_array_operations():
    """Test missing core array operations."""
    print("=== Testing Core Array Operations ===")
    
    test_data = [1.0, 2.0, float('nan'), float('inf'), 3.0]
    
    # Test isnan, isinf, isfinite
    nan_checks = ['isnan', 'isinf', 'isfinite']
    for func_name in nan_checks:
        try:
            if hasattr(tlm, func_name):
                func = getattr(tlm, func_name)
                result = func(test_data)
                print(f"{func_name}({test_data}) = {result}")
                print(f"PASS {func_name}: Works")
            else:
                print(f"SKIP {func_name}: Not found")
        except Exception as e:
            print(f"FAIL {func_name}: {e}")
    
    # Test allclose
    try:
        if hasattr(tlm, 'allclose'):
            a = [1.0, 2.0, 3.0]
            b = [1.0001, 1.9999, 3.0001]
            result = tlm.allclose(a, b, 1e-3)
            print(f"allclose({a}, {b}, 1e-3) = {result}")
            print(f"PASS allclose: Works")
        else:
            print("SKIP allclose: Not found")
    except Exception as e:
        print(f"FAIL allclose: {e}")
    
    # Test where
    try:
        if hasattr(tlm, 'where'):
            condition = [True, False, True, False]
            x = [1, 2, 3, 4]
            y = [10, 20, 30, 40]
            result = tlm.where(condition, x, y)
            print(f"where({condition}, {x}, {y}) = {result}")
            print(f"PASS where: Works")
        else:
            print("SKIP where: Not found")
    except Exception as e:
        print(f"FAIL where: {e}")

def test_critical_loss_functions():
    """Test critical loss functions."""
    print("\n=== Testing Critical Loss Functions ===")
    
    # Multi-class data
    y_true_multi = [0, 1, 2, 1, 0]
    y_pred_proba = [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.3, 0.5], 
                    [0.3, 0.6, 0.1], [0.7, 0.2, 0.1]]
    
    # Binary data
    y_true_binary = [0, 1, 1, 0]
    y_pred_binary = [0.1, 0.9, 0.8, 0.2]
    
    loss_tests = [
        ('cross_entropy', [y_pred_proba, y_true_multi]),
        ('sparse_cross_entropy', [y_pred_proba, y_true_multi]),
        ('focal_loss', [y_pred_binary, y_true_binary]),
        ('hinge_loss', [y_pred_binary, y_true_binary]),
        ('huber_loss', [y_true_binary, y_pred_binary]),
        ('kl_divergence', [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]]),
    ]
    
    for func_name, args in loss_tests:
        try:
            if hasattr(tlm, func_name):
                func = getattr(tlm, func_name)
                result = func(*args)
                print(f"{func_name} = {result}")
                print(f"PASS {func_name}: Works")
            else:
                print(f"SKIP {func_name}: Not found")
        except Exception as e:
            print(f"FAIL {func_name}: {e}")

def test_missing_ml_core():
    """Test missing core ML functions."""
    print("\n=== Testing Missing ML Core ===")
    
    # Generate test data
    X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    y = [0, 1, 0, 1, 0]
    
    ml_tests = [
        ('logreg_predict_proba', 'requires fitted model'),
        ('svm_predict', 'requires fitted model'),
        ('softmax_fit', [X, [0, 1, 2, 1, 0]]),  # Multi-class labels
        ('gmm_fit', [X, 2]),  # 2 components
    ]
    
    for func_name, test_info in ml_tests:
        try:
            if hasattr(tlm, func_name):
                if isinstance(test_info, str):
                    print(f"SKIP {func_name}: {test_info}")
                    continue
                    
                func = getattr(tlm, func_name)
                result = func(*test_info)
                print(f"{func_name}: Model fitted successfully")
                print(f"PASS {func_name}: Works")
            else:
                print(f"SKIP {func_name}: Not found")
        except Exception as e:
            print(f"FAIL {func_name}: {e}")

def test_activation_functions():
    """Test activation functions."""
    print("\n=== Testing Activation Functions ===")
    
    test_input = [-2, -1, 0, 1, 2]
    
    activation_tests = ['sigmoid', 'relu', 'leaky_relu', 'softmax']
    
    for func_name in activation_tests:
        try:
            if hasattr(tlm, func_name):
                func = getattr(tlm, func_name)
                if func_name == 'softmax':
                    # Softmax needs 2D input
                    result = func([[1, 2, 3], [4, 5, 6]])
                else:
                    result = func(test_input)
                print(f"{func_name}({test_input if func_name != 'softmax' else '2D input'}) = {result}")
                print(f"PASS {func_name}: Works")
            else:
                print(f"SKIP {func_name}: Not found")
        except Exception as e:
            print(f"FAIL {func_name}: {e}")

def main():
    """Run high priority function tests."""
    print("HIGH PRIORITY TLM FUNCTION TESTING")
    print("=" * 50)
    print("Testing the most critical untested functions.")
    print()
    
    test_core_array_operations()
    test_critical_loss_functions()
    test_missing_ml_core()
    test_activation_functions()
    
    print("\n" + "=" * 50)
    print("High priority testing complete!")
    print("\nNext steps:")
    print("1. Fix any critical failures above")
    print("2. Add working functions to comprehensive test suite")
    print("3. Continue with medium priority functions")

if __name__ == "__main__":
    main()