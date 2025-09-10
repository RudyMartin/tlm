#!/usr/bin/env python3
"""
TLM Comprehensive Mathematical Validation Suite

This module provides extensive validation of TLM's mathematical functions against
multiple reference implementations including NumPy, SciPy, and mpmath.

The suite tests:
- Numerical accuracy and precision
- Edge case handling
- Special value behavior (inf, nan, zero)
- Broadcasting compatibility
- Performance characteristics
- Statistical properties

Usage:
    python comprehensive_math_validation.py
    
Dependencies:
    - numpy (for primary reference)
    - scipy (for statistical functions)
    - mpmath (for arbitrary precision validation)
    
Note: All dependencies are optional - tests will skip if not available
"""

import math
import time
import sys
import warnings
from typing import List, Dict, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass
from pathlib import Path

# Add TLM to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import tlm

# Optional dependencies
HAS_NUMPY = False
HAS_SCIPY = False  
HAS_MPMATH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("WARNING: NumPy not available - skipping NumPy comparisons")

try:
    import scipy.stats as stats
    import scipy.special as special
    HAS_SCIPY = True
except ImportError:
    print("WARNING: SciPy not available - skipping SciPy comparisons")

try:
    import mpmath
    HAS_MPMATH = True
except ImportError:
    print("WARNING: mpmath not available - skipping high-precision comparisons")


@dataclass
class ValidationResult:
    """Result of a validation test."""
    function_name: str
    test_name: str
    passed: bool
    tlm_result: Any
    reference_result: Any
    error_message: str
    max_diff: Optional[float]
    relative_error: Optional[float]
    execution_time_tlm: float
    execution_time_ref: float


class ComprehensiveMathValidator:
    """Comprehensive validation suite for TLM mathematical functions."""
    
    def __init__(self, tolerance: float = 1e-10, relative_tolerance: float = 1e-8):
        self.tolerance = tolerance
        self.relative_tolerance = relative_tolerance
        self.results: List[ValidationResult] = []
        
    def compare_values(self, tlm_result: Any, ref_result: Any, 
                      func_name: str, test_name: str) -> ValidationResult:
        """Compare TLM result against reference implementation."""
        
        try:
            # Convert to numpy arrays for comparison if numpy available
            if HAS_NUMPY:
                tlm_array = np.asarray(tlm_result, dtype=float)
                ref_array = np.asarray(ref_result, dtype=float)
                
                # Check shapes
                if tlm_array.shape != ref_array.shape:
                    return ValidationResult(
                        function_name=func_name,
                        test_name=test_name,
                        passed=False,
                        tlm_result=tlm_result,
                        reference_result=ref_result,
                        error_message=f"Shape mismatch: TLM {tlm_array.shape} vs Ref {ref_array.shape}",
                        max_diff=None,
                        relative_error=None,
                        execution_time_tlm=0,
                        execution_time_ref=0
                    )
                
                # Calculate differences
                abs_diff = np.abs(tlm_array - ref_array)
                max_diff = np.max(abs_diff)
                
                # Calculate relative error
                ref_magnitude = np.abs(ref_array)
                relative_error = np.max(abs_diff / np.maximum(ref_magnitude, 1e-15))
                
                # Check tolerance - handle NaN cases
                if np.isnan(tlm_array).any() and np.isnan(ref_array).any():
                    # If both have NaN in same places, consider it a pass
                    passed = np.allclose(tlm_array, ref_array, 
                                       rtol=self.relative_tolerance, 
                                       atol=self.tolerance, 
                                       equal_nan=True)
                else:
                    passed = np.allclose(tlm_array, ref_array, 
                                       rtol=self.relative_tolerance, 
                                       atol=self.tolerance)
                
                error_msg = "OK" if passed else f"Max diff: {max_diff:.2e}, Rel err: {relative_error:.2e}"
                
                return ValidationResult(
                    function_name=func_name,
                    test_name=test_name,
                    passed=passed,
                    tlm_result=tlm_result,
                    reference_result=ref_result,
                    error_message=error_msg,
                    max_diff=float(max_diff),
                    relative_error=float(relative_error),
                    execution_time_tlm=0,
                    execution_time_ref=0
                )
            else:
                # Fallback comparison without numpy
                if isinstance(tlm_result, (list, tuple)) and isinstance(ref_result, (list, tuple)):
                    if len(tlm_result) != len(ref_result):
                        passed = False
                        error_msg = f"Length mismatch: {len(tlm_result)} vs {len(ref_result)}"
                    else:
                        max_diff = max(abs(a - b) for a, b in zip(tlm_result, ref_result))
                        passed = max_diff < self.tolerance
                        error_msg = "OK" if passed else f"Max diff: {max_diff:.2e}"
                else:
                    max_diff = abs(float(tlm_result) - float(ref_result))
                    passed = max_diff < self.tolerance
                    error_msg = "OK" if passed else f"Diff: {max_diff:.2e}"
                
                return ValidationResult(
                    function_name=func_name,
                    test_name=test_name,
                    passed=passed,
                    tlm_result=tlm_result,
                    reference_result=ref_result,
                    error_message=error_msg,
                    max_diff=max_diff if 'max_diff' in locals() else None,
                    relative_error=None,
                    execution_time_tlm=0,
                    execution_time_ref=0
                )
                
        except Exception as e:
            return ValidationResult(
                function_name=func_name,
                test_name=test_name,
                passed=False,
                tlm_result=tlm_result,
                reference_result=ref_result,
                error_message=f"Comparison error: {str(e)}",
                max_diff=None,
                relative_error=None,
                execution_time_tlm=0,
                execution_time_ref=0
            )

    def benchmark_function(self, func: Callable, args: tuple, runs: int = 100) -> Tuple[Any, float]:
        """Benchmark function execution time."""
        # Warmup
        result = func(*args)
        
        # Time multiple runs
        start_time = time.perf_counter()
        for _ in range(runs):
            result = func(*args)
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / runs * 1000  # Convert to ms
        return result, avg_time

    def test_basic_operations(self):
        """Test basic mathematical operations."""
        print("\n=== Testing Basic Operations ===")
        
        test_cases = [
            # Scalars, vectors, matrices
            (5, [1, 2, 3, 4, 5], [[1, 2], [3, 4], [5, 6]]),
        ]
        
        if not HAS_NUMPY:
            print("Skipping basic operations - NumPy required")
            return
            
        for scalar, vector, matrix in test_cases:
            # Addition tests
            if hasattr(tlm, 'add'):
                tlm_result, tlm_time = self.benchmark_function(tlm.add, (vector, scalar))
                ref_result, ref_time = self.benchmark_function(np.add, (vector, scalar))
                
                result = self.compare_values(tlm_result, ref_result, 'add', 'vector_scalar_add')
                result.execution_time_tlm = tlm_time
                result.execution_time_ref = ref_time
                self.results.append(result)
                
            # Matrix operations
            if hasattr(tlm, 'matmul'):
                A = [[1, 2], [3, 4]]
                B = [[5, 6], [7, 8]]
                
                tlm_result, tlm_time = self.benchmark_function(tlm.matmul, (A, B))
                ref_result, ref_time = self.benchmark_function(np.matmul, (A, B))
                
                result = self.compare_values(tlm_result, ref_result, 'matmul', 'matrix_multiplication')
                result.execution_time_tlm = tlm_time
                result.execution_time_ref = ref_time
                self.results.append(result)

    def test_statistical_functions(self):
        """Test statistical functions against NumPy and SciPy."""
        print("\n=== Testing Statistical Functions ===")
        
        test_data = [
            [1, 2, 3, 4, 5],
            [1, 1, 2, 3, 5, 8, 13],
            [-2, -1, 0, 1, 2],
            [1e-10, 1e-5, 1, 1e5, 1e10],  # Wide range
        ]
        
        if not HAS_NUMPY:
            print("Skipping statistical functions - NumPy required") 
            return
            
        for i, data in enumerate(test_data):
            # Mean
            if hasattr(tlm, 'mean'):
                tlm_result, tlm_time = self.benchmark_function(tlm.mean, (data,))
                ref_result, ref_time = self.benchmark_function(np.mean, (data,))
                
                result = self.compare_values(tlm_result, ref_result, 'mean', f'dataset_{i}')
                result.execution_time_tlm = tlm_time
                result.execution_time_ref = ref_time
                self.results.append(result)
                
            # Variance  
            if hasattr(tlm, 'var'):
                tlm_result, tlm_time = self.benchmark_function(tlm.var, (data,))  # default ddof=0
                ref_result, ref_time = self.benchmark_function(np.var, (data,))
                
                result = self.compare_values(tlm_result, ref_result, 'var', f'dataset_{i}')
                result.execution_time_tlm = tlm_time
                result.execution_time_ref = ref_time  
                self.results.append(result)
                
            # Standard deviation
            if hasattr(tlm, 'std'):
                tlm_result, tlm_time = self.benchmark_function(tlm.std, (data,))
                ref_result, ref_time = self.benchmark_function(np.std, (data,))
                
                result = self.compare_values(tlm_result, ref_result, 'std', f'dataset_{i}')
                result.execution_time_tlm = tlm_time
                result.execution_time_ref = ref_time
                self.results.append(result)

    def test_edge_cases(self):
        """Test edge cases and special values."""
        print("\n=== Testing Edge Cases ===")
        
        if not HAS_NUMPY:
            print("Skipping edge cases - NumPy required")
            return
        
        # Test with special values
        special_values = [
            [0, 0, 0],
            [1, 1, 1],
            [-1, -1, -1], 
            [float('inf'), 1, 2],
            [1, float('nan'), 2],
            [1e-100, 1e-50, 1e50, 1e100],  # Extreme ranges
        ]
        
        for i, data in enumerate(special_values):
            try:
                if hasattr(tlm, 'mean'):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tlm_result = tlm.mean(data)
                        ref_result = np.mean(data)
                        
                    result = self.compare_values(tlm_result, ref_result, 'mean', f'special_case_{i}')
                    self.results.append(result)
                    
            except Exception as e:
                result = ValidationResult(
                    function_name='mean',
                    test_name=f'special_case_{i}',
                    passed=False,
                    tlm_result=None,
                    reference_result=None,
                    error_message=f"Exception: {str(e)}",
                    max_diff=None,
                    relative_error=None,
                    execution_time_tlm=0,
                    execution_time_ref=0
                )
                self.results.append(result)

    def test_precision_limits(self):
        """Test numerical precision limits."""
        print("\n=== Testing Precision Limits ===")
        
        if not HAS_MPMATH:
            print("Skipping precision tests - mpmath required")
            return
            
        # Test cases that challenge floating point precision
        precision_cases = [
            ([1, 1e-15, 1e-16], "tiny_differences"),
            ([1e20, 1, 1e-20], "large_scale_differences"), 
            ([1/3, 1/3, 1/3], "rational_numbers"),
        ]
        
        for data, case_name in precision_cases:
            if hasattr(tlm, 'mean'):
                try:
                    # TLM result
                    tlm_result = tlm.mean(data)
                    
                    # High precision reference
                    mpmath.mp.dps = 50  # 50 decimal places
                    mp_data = [mpmath.mpf(x) for x in data]
                    mp_result = sum(mp_data) / len(mp_data)
                    ref_result = float(mp_result)
                    
                    result = self.compare_values(tlm_result, ref_result, 'mean', case_name)
                    self.results.append(result)
                    
                except Exception as e:
                    result = ValidationResult(
                        function_name='mean',
                        test_name=case_name,
                        passed=False,
                        tlm_result=None,
                        reference_result=None,
                        error_message=f"Precision test error: {str(e)}",
                        max_diff=None,
                        relative_error=None,
                        execution_time_tlm=0,
                        execution_time_ref=0
                    )
                    self.results.append(result)

    def test_scipy_functions(self):
        """Test against SciPy statistical functions."""
        print("\n=== Testing Against SciPy ===")
        
        if not HAS_SCIPY:
            print("Skipping SciPy tests - SciPy not available")
            return
            
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # Test correlation if available
        if hasattr(tlm, 'correlation'):
            x = [1, 2, 3, 4, 5]
            y = [2, 4, 1, 3, 5]
            
            try:
                tlm_result = tlm.correlation(x, y)
                ref_result = stats.pearsonr(x, y)[0]  # Get correlation coefficient
                
                result = self.compare_values(tlm_result, ref_result, 'correlation', 'pearson_correlation')
                self.results.append(result)
                
            except Exception as e:
                result = ValidationResult(
                    function_name='correlation',
                    test_name='pearson_correlation',
                    passed=False,
                    tlm_result=None,
                    reference_result=None,
                    error_message=f"SciPy correlation error: {str(e)}",
                    max_diff=None,
                    relative_error=None,
                    execution_time_tlm=0,
                    execution_time_ref=0
                )
                self.results.append(result)

    def run_comprehensive_validation(self):
        """Run the complete validation suite."""
        print("TLM Comprehensive Mathematical Validation Suite")
        print("=" * 60)
        print(f"Tolerance: {self.tolerance}")
        print(f"Relative tolerance: {self.relative_tolerance}")
        print(f"NumPy available: {HAS_NUMPY}")
        print(f"SciPy available: {HAS_SCIPY}")
        print(f"mpmath available: {HAS_MPMATH}")
        
        # Run all test suites
        self.test_basic_operations()
        self.test_statistical_functions()
        self.test_edge_cases()
        self.test_precision_limits()
        self.test_scipy_functions()
        
        return self.generate_report()

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        success_rate = (passed / total * 100) if total > 0 else 0
        
        # Group results by function
        by_function = {}
        for result in self.results:
            func = result.function_name
            if func not in by_function:
                by_function[func] = []
            by_function[func].append(result)
        
        # Calculate performance statistics
        tlm_times = [r.execution_time_tlm for r in self.results if r.execution_time_tlm > 0]
        ref_times = [r.execution_time_ref for r in self.results if r.execution_time_ref > 0]
        
        performance_ratio = (sum(tlm_times) / sum(ref_times)) if ref_times else None
        
        report = {
            'summary': {
                'total_tests': total,
                'passed': passed,
                'failed': total - passed,
                'success_rate': success_rate,
                'performance_ratio': performance_ratio
            },
            'by_function': {},
            'failed_tests': [],
            'precision_analysis': {}
        }
        
        # Detailed function analysis
        for func_name, func_results in by_function.items():
            func_passed = sum(1 for r in func_results if r.passed)
            func_total = len(func_results)
            
            max_errors = [r.max_diff for r in func_results if r.max_diff is not None]
            rel_errors = [r.relative_error for r in func_results if r.relative_error is not None]
            
            report['by_function'][func_name] = {
                'tests': func_total,
                'passed': func_passed,
                'success_rate': func_passed / func_total * 100,
                'max_absolute_error': max(max_errors) if max_errors else None,
                'max_relative_error': max(rel_errors) if rel_errors else None,
                'avg_absolute_error': sum(max_errors) / len(max_errors) if max_errors else None
            }
        
        # Failed tests
        report['failed_tests'] = [
            {
                'function': r.function_name,
                'test': r.test_name,
                'error': r.error_message,
                'tlm_result': str(r.tlm_result)[:100],
                'ref_result': str(r.reference_result)[:100]
            }
            for r in self.results if not r.passed
        ]
        
        return report

    def print_report(self, report: Dict[str, Any]):
        """Print formatted validation report."""
        summary = report['summary']
        
        print(f"\n{'='*60}")
        print("COMPREHENSIVE VALIDATION REPORT")
        print(f"{'='*60}")
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        
        if summary['performance_ratio']:
            print(f"Performance ratio (TLM/Reference): {summary['performance_ratio']:.2f}x")
        
        print(f"\n{'Function Analysis':=^60}")
        for func_name, func_data in report['by_function'].items():
            print(f"\n{func_name}:")
            print(f"  Tests: {func_data['tests']}")
            print(f"  Success rate: {func_data['success_rate']:.1f}%")
            if func_data['max_absolute_error']:
                print(f"  Max absolute error: {func_data['max_absolute_error']:.2e}")
            if func_data['max_relative_error']:
                print(f"  Max relative error: {func_data['max_relative_error']:.2e}")
        
        if report['failed_tests']:
            print(f"\n{'Failed Tests':=^60}")
            for failure in report['failed_tests']:
                print(f"\n{failure['function']}.{failure['test']}:")
                print(f"  Error: {failure['error']}")
                if failure['tlm_result'] != 'None':
                    print(f"  TLM result: {failure['tlm_result']}")
                if failure['ref_result'] != 'None':
                    print(f"  Ref result: {failure['ref_result']}")
        
        # Final assessment
        print(f"\n{'Assessment':=^60}")
        if summary['success_rate'] >= 95:
            print("EXCELLENT: TLM demonstrates high mathematical accuracy")
        elif summary['success_rate'] >= 90:
            print("GOOD: TLM shows solid mathematical reliability")
        elif summary['success_rate'] >= 80:
            print("ACCEPTABLE: Some numerical issues detected")
        else:
            print("NEEDS IMPROVEMENT: Significant mathematical issues found")


def main():
    """Run comprehensive validation and generate documentation."""
    validator = ComprehensiveMathValidator(
        tolerance=1e-10,
        relative_tolerance=1e-8
    )
    
    report = validator.run_comprehensive_validation()
    validator.print_report(report)
    
    # Save detailed report
    import json
    report_path = Path(__file__).parent / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: {report_path}")
    
    return report['summary']['success_rate'] >= 90


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)