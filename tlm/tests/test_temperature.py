"""
Tests for temperature scaling functions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import tlm


def test_temperature_softmax_standard():
    """Test standard softmax (T=1.0)."""
    logits = [2.0, 1.0, 0.5]
    probs = tlm.temperature_scaled_softmax(logits, temperature=1.0)

    # Check sums to 1
    assert abs(sum(probs) - 1.0) < 1e-6
    print("✓ Softmax probabilities sum to 1.0")

    # Check ordering preserved
    assert probs[0] > probs[1] > probs[2]
    print("✓ Softmax preserves ordering")


def test_temperature_softmax_cold():
    """Test cold temperature (T→0) produces near one-hot."""
    logits = [2.0, 1.0, 0.5]
    probs = tlm.temperature_scaled_softmax(logits, temperature=0.01)

    # First element should be close to 1.0
    assert probs[0] > 0.99
    print("✓ Cold temperature (T=0.01) produces near one-hot")

    # Other elements should be close to 0
    assert probs[1] < 0.01
    assert probs[2] < 0.01
    print("✓ Non-maximum values close to 0")


def test_temperature_softmax_hot():
    """Test hot temperature (T>1) produces more uniform distribution."""
    logits = [2.0, 1.0, 0.5]

    probs_cold = tlm.temperature_scaled_softmax(logits, temperature=0.5)
    probs_hot = tlm.temperature_scaled_softmax(logits, temperature=2.0)

    # Hot temperature should reduce gap between max and min
    gap_cold = probs_cold[0] - probs_cold[2]
    gap_hot = probs_hot[0] - probs_hot[2]

    assert gap_hot < gap_cold
    print("✓ Hot temperature reduces probability gaps")


def test_temperature_softmax_zero():
    """Test deterministic behavior at T=0."""
    logits = [1.0, 3.0, 2.0]
    probs = tlm.temperature_scaled_softmax(logits, temperature=1e-10)

    # Should be one-hot at index 1 (max value)
    assert abs(probs[1] - 1.0) < 1e-6
    assert abs(probs[0]) < 1e-6
    assert abs(probs[2]) < 1e-6
    print("✓ T→0 produces perfect one-hot distribution")


def test_temperature_argmax_deterministic():
    """Test deterministic argmax at T→0."""
    logits = [1.0, 3.0, 2.0]
    idx = tlm.temperature_argmax(logits, temperature=0.0001)
    assert idx == 1  # Index of maximum (3.0)
    print("✓ Temperature argmax returns correct index")


def test_apply_temperature():
    """Test general temperature application."""
    scores = [0.9, 0.5, 0.1]

    # Higher temperature should reduce differences
    scaled = tlm.apply_temperature(scores, temperature=2.0)

    gap_original = scores[0] - scores[2]
    gap_scaled = scaled[0] - scaled[2]

    assert gap_scaled < gap_original
    print("✓ Apply temperature reduces score gaps")


def test_apply_temperature_zero():
    """Test apply_temperature at T=0 produces one-hot."""
    scores = [0.9, 0.5, 0.1]
    scaled = tlm.apply_temperature(scores, temperature=1e-10)

    # Should be one-hot at index 0 (max score)
    assert abs(scaled[0] - 1.0) < 1e-6
    assert abs(scaled[1]) < 1e-6
    assert abs(scaled[2]) < 1e-6
    print("✓ Apply temperature at T→0 produces one-hot")


def test_temperature_schedule_linear():
    """Test linear temperature schedule."""
    # Start at 1.0, end at 0.0, 100 steps
    t_start = tlm.temperature_schedule(1.0, 0.0, 0, 100, 'linear')
    t_mid = tlm.temperature_schedule(1.0, 0.0, 50, 100, 'linear')
    t_end = tlm.temperature_schedule(1.0, 0.0, 100, 100, 'linear')

    assert abs(t_start - 1.0) < 0.01
    assert abs(t_mid - 0.5) < 0.01
    assert abs(t_end - 0.0) < 0.01
    print("✓ Linear temperature schedule works correctly")


def test_temperature_schedule_exponential():
    """Test exponential temperature schedule."""
    t_start = tlm.temperature_schedule(1.0, 0.1, 0, 100, 'exponential')
    t_end = tlm.temperature_schedule(1.0, 0.1, 100, 100, 'exponential')

    assert abs(t_start - 1.0) < 0.01
    assert abs(t_end - 0.1) < 0.01
    print("✓ Exponential temperature schedule works correctly")


def test_temperature_schedule_cosine():
    """Test cosine temperature schedule."""
    t_start = tlm.temperature_schedule(1.0, 0.0, 0, 100, 'cosine')
    t_end = tlm.temperature_schedule(1.0, 0.0, 100, 100, 'cosine')

    assert abs(t_start - 1.0) < 0.01
    assert abs(t_end - 0.0) < 0.01
    print("✓ Cosine temperature schedule works correctly")


if __name__ == '__main__':
    print("\n=== Testing Temperature Functions ===\n")
    test_temperature_softmax_standard()
    test_temperature_softmax_cold()
    test_temperature_softmax_hot()
    test_temperature_softmax_zero()
    test_temperature_argmax_deterministic()
    test_apply_temperature()
    test_apply_temperature_zero()
    test_temperature_schedule_linear()
    test_temperature_schedule_exponential()
    test_temperature_schedule_cosine()
    print("\n✅ All temperature tests passed!\n")
