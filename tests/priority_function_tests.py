#!/usr/bin/env python3
"""
Quick validation tests for high-priority untested TLM functions.

These are the most important functions to test next to complete core functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tlm

def test_core_math_completions():
    """Test the 3 missing core math functions."""
    print("=== Testing Core Math Completions ===")
    
    # Test abs
    try:
        result = tlm.abs([-2, -1, 0, 1, 2])
        expected = [2, 1, 0, 1, 2]
        print(f"abs([-2,-1,0,1,2]) = {result}")
        print(f"Expected: {expected}")
        print(f"PASS abs: {'PASS' if result == expected else 'FAIL'}")
    except Exception as e:
        print(f"FAIL abs: ERROR - {e}")
    
    # Test power
    try:
        result = tlm.power([1, 2, 3, 4], 2)
        expected = [1, 4, 9, 16]
        print(f"\npower([1,2,3,4], 2) = {result}")
        print(f"Expected: {expected}")
        print(f"PASS power: {'PASS' if result == expected else 'FAIL'}")
    except Exception as e:
        print(f"FAIL power: ERROR - {e}")
    
    # Test sign
    try:
        result = tlm.sign([-2, -0.5, 0, 0.5, 2])
        expected = [-1, -1, 0, 1, 1]
        print(f"\nsign([-2,-0.5,0,0.5,2]) = {result}")
        print(f"Expected: {expected}")
        print(f"PASS sign: {'PASS' if result == expected else 'FAIL'}")
    except Exception as e:
        print(f"FAIL sign: ERROR - {e}")

def test_ml_core_completions():
    """Test missing ML core functions."""
    print("\n\n=== Testing ML Core Completions ===")
    
    # Generate test data
    X = [[1, 2], [2, 3], [3, 4], [4, 5]]
    y = [0, 0, 1, 1]
    
    # Test logreg_predict_proba
    try:
        # First fit a model
        model = tlm.logreg_fit(X, y)
        proba = tlm.logreg_predict_proba([[2.5, 3.5]], model['weights'], model['bias'])
        print(f"logreg_predict_proba([[2.5, 3.5]]) = {proba}")
        print(f"PASS logreg_predict_proba: {'PASS' if 0 <= proba[0] <= 1 else 'FAIL'}")
    except Exception as e:
        print(f"FAIL logreg_predict_proba: ERROR - {e}")
    
    # Test svm_predict
    try:
        svm_model = tlm.svm_fit(X, y)
        prediction = tlm.svm_predict([[2.5, 3.5]], svm_model['weights'], svm_model['bias'])
        print(f"\nsvm_predict([[2.5, 3.5]]) = {prediction}")
        print(f"PASS svm_predict: {'PASS' if prediction[0] in [0, 1] else 'FAIL'}")
    except Exception as e:
        print(f"FAIL svm_predict: ERROR - {e}")
    
    # Test nb_fit and nb_predict
    try:
        nb_model = tlm.nb_fit(X, y)
        nb_prediction = tlm.nb_predict([[2.5, 3.5]], nb_model)
        print(f"\nnb_fit and nb_predict([[2.5, 3.5]]) = {nb_prediction}")
        print(f"PASS nb_fit/predict: {'PASS' if nb_prediction[0] in [0, 1] else 'FAIL'}")
    except Exception as e:
        print(f"FAIL nb_fit/predict: ERROR - {e}")

def test_statistics_completions():
    """Test missing statistics functions."""
    print("\n\n=== Testing Statistics Completions ===")
    
    # Test covariance_matrix
    try:
        X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        cov = tlm.covariance_matrix(X)
        print(f"covariance_matrix([[1,2,3],[4,5,6],[7,8,9]]) shape = {tlm.shape(cov)}")
        print(f"PASS covariance_matrix: {'PASS' if tlm.shape(cov) == (3, 3) else 'FAIL'}")
    except Exception as e:
        print(f"FAIL covariance_matrix: ERROR - {e}")
    
    # Test histogram
    try:
        data = [1, 1, 2, 2, 2, 3, 3, 4, 5]
        hist = tlm.histogram(data, bins=5)
        print(f"\nhistogram([1,1,2,2,2,3,3,4,5], bins=5) = {hist}")
        print(f"PASS histogram: {'PASS' if len(hist) == 2 else 'FAIL'}")  # Should return (counts, bins)
    except Exception as e:
        print(f"FAIL histogram: ERROR - {e}")

def test_loss_functions_sample():
    """Test a few key loss functions."""
    print("\n\n=== Testing Sample Loss Functions ===")
    
    y_true = [0, 1, 1, 0]
    y_pred = [0.1, 0.9, 0.8, 0.2]
    
    # Test mse
    try:
        loss = tlm.mse(y_true, y_pred)
        print(f"mse({y_true}, {y_pred}) = {loss}")
        print(f"PASS mse: {'PASS' if 0 <= loss <= 1 else 'FAIL'}")
    except Exception as e:
        print(f"FAIL mse: ERROR - {e}")
    
    # Test mae
    try:
        loss = tlm.mae(y_true, y_pred)
        print(f"\nmae({y_true}, {y_pred}) = {loss}")
        print(f"PASS mae: {'PASS' if 0 <= loss <= 1 else 'FAIL'}")
    except Exception as e:
        print(f"FAIL mae: ERROR - {e}")
    
    # Test binary_cross_entropy
    try:
        loss = tlm.binary_cross_entropy(y_pred, y_true)
        print(f"\nbinary_cross_entropy({y_pred}, {y_true}) = {loss}")
        print(f"PASS binary_cross_entropy: {'PASS' if loss > 0 else 'FAIL'}")
    except Exception as e:
        print(f"FAIL binary_cross_entropy: ERROR - {e}")

def main():
    """Run priority function tests."""
    print("TLM Priority Function Testing")
    print("=" * 50)
    print("Testing high-priority untested functions to complete core functionality.\n")
    
    test_core_math_completions()
    test_ml_core_completions()
    test_statistics_completions()
    test_loss_functions_sample()
    
    print("\n" + "=" * 50)
    print("Priority testing complete!")
    print("\nNext steps:")
    print("1. Fix any failed functions above")
    print("2. Add these tests to comprehensive validation suite")
    print("3. Expand testing to specialized domains (signal processing, trading)")

if __name__ == "__main__":
    main()