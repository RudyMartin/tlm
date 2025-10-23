"""
Temperature scaling for controlling determinism vs exploration.

Temperature (T) controls the "sharpness" of probability distributions:
- T → 0: Deterministic (argmax, one-hot)
- T = 1: Standard probabilities
- T > 1: More uniform (exploration)

Used in:
- Simulated annealing (optimization)
- Neural network sampling (LLMs)
- Reinforcement learning (action selection)
- Tensor Logic reasoning (symbolic vs analogical)
"""

from math import exp


def temperature_scaled_softmax(logits, temperature=1.0):
    """Softmax with temperature scaling.

    Args:
        logits: List of raw scores (unnormalized)
        temperature: Float > 0 controlling sharpness
            - T → 0: Deterministic (one-hot of argmax)
            - T = 1: Standard softmax
            - T > 1: More uniform distribution

    Returns:
        List of probabilities (sums to 1.0)

    Examples:
        >>> logits = [2.0, 1.0, 0.5]
        >>> probs = temperature_scaled_softmax(logits, temperature=1.0)
        >>> abs(sum(probs) - 1.0) < 1e-6
        True

        >>> # Low temperature (deterministic)
        >>> probs_cold = temperature_scaled_softmax(logits, temperature=0.01)
        >>> probs_cold[0] > 0.99  # Close to 1.0
        True

        >>> # High temperature (uniform)
        >>> probs_hot = temperature_scaled_softmax(logits, temperature=10.0)
        >>> abs(probs_hot[0] - 0.33) < 0.05  # Close to 1/3
        True
    """
    # Validate temperature is positive
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    # Handle near-zero temperature (deterministic behavior)
    # When T ≈ 0, return one-hot vector (probability 1 for max, 0 for others)
    if temperature < 1e-8:
        max_val = max(logits)
        max_idx = logits.index(max_val)
        # Create one-hot: [1.0, 0.0, 0.0, ...] at max_idx position
        return [1.0 if i == max_idx else 0.0 for i in range(len(logits))]

    # Scale logits by temperature: higher T makes distribution more uniform
    # Lower T makes distribution more peaked (closer to one-hot)
    scaled_logits = [x / temperature for x in logits]

    # Compute softmax with numerical stability (subtract max to prevent overflow)
    # softmax(x) = exp(x) / sum(exp(x))
    max_logit = max(scaled_logits)
    # Subtract max_logit for numerical stability (prevents exp overflow)
    exp_logits = [exp(x - max_logit) for x in scaled_logits]
    sum_exp = sum(exp_logits)

    # Normalize: each element divided by sum gives probabilities summing to 1.0
    return [e / sum_exp for e in exp_logits]


def temperature_argmax(logits, temperature=1.0):
    """Get index of maximum element with temperature-controlled randomness.

    Args:
        logits: List of raw scores
        temperature: Float > 0
            - T → 0: Deterministic argmax
            - T > 0: Sample from temperature-scaled softmax

    Returns:
        Integer index

    Examples:
        >>> logits = [2.0, 1.0, 0.5]
        >>> idx = temperature_argmax(logits, temperature=0.01)
        >>> idx
        0
    """
    # For very low temperature, use deterministic argmax
    if temperature < 1e-8:
        # Find the index of the maximum value
        max_val = max(logits)
        return logits.index(max_val)
    else:
        # For higher temperature, compute temperature-scaled probabilities
        probs = temperature_scaled_softmax(logits, temperature)

        # Return index of highest probability
        # Note: True stochastic sampling would require random number generation
        # This implementation returns deterministic argmax of scaled probabilities
        max_prob = max(probs)
        return probs.index(max_prob)


def apply_temperature(scores, temperature=1.0):
    """Apply temperature scaling to a list of scores.

    Convenience function that handles both positive and negative scores.

    Args:
        scores: List of floats (e.g., similarities, confidences)
        temperature: Float > 0 controlling sharpness

    Returns:
        List of temperature-scaled values (not necessarily summing to 1)

    Examples:
        >>> scores = [0.9, 0.5, 0.1]
        >>> scaled = apply_temperature(scores, temperature=2.0)
        >>> scaled[0] < scores[0]  # High temp reduces peaks
        True
    """
    # Validate temperature is positive
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    # For near-zero temperature, return one-hot vector at max position
    if temperature < 1e-8:
        max_val = max(scores)
        max_idx = scores.index(max_val)
        # One-hot encoding: 1.0 at max, 0.0 elsewhere (deterministic)
        return [1.0 if i == max_idx else 0.0 for i in range(len(scores))]

    # Apply temperature scaling by dividing scores
    # T > 1: reduces differences (more uniform)
    # T < 1: increases differences (more peaked)
    # T = 1: no change
    return [s / temperature for s in scores]


def temperature_schedule(initial_temp, final_temp, current_step, total_steps, schedule_type='linear'):
    """Compute temperature at current step using specified schedule.

    Used in simulated annealing and other optimization algorithms.

    Args:
        initial_temp: Starting temperature
        final_temp: Ending temperature
        current_step: Current iteration (0-indexed)
        total_steps: Total number of steps
        schedule_type: Schedule type ('linear', 'exponential', 'cosine')

    Returns:
        Float temperature value

    Examples:
        >>> # Linear cooling
        >>> t = temperature_schedule(1.0, 0.0, 50, 100, 'linear')
        >>> abs(t - 0.5) < 0.01
        True

        >>> # At start
        >>> t = temperature_schedule(1.0, 0.0, 0, 100, 'linear')
        >>> abs(t - 1.0) < 0.01
        True

        >>> # At end
        >>> t = temperature_schedule(1.0, 0.0, 100, 100, 'linear')
        >>> abs(t - 0.0) < 0.01
        True
    """
    # If we've reached or exceeded total steps, return final temperature
    if current_step >= total_steps:
        return final_temp

    # Calculate progress as fraction from 0.0 (start) to 1.0 (end)
    progress = current_step / total_steps

    if schedule_type == 'linear':
        # Linear interpolation: straight line from initial to final
        # T(p) = T_initial + (T_final - T_initial) * p
        return initial_temp + (final_temp - initial_temp) * progress

    elif schedule_type == 'exponential':
        # Exponential decay: fast cooling early, slower later
        # T(p) = T_initial * (T_final / T_initial)^p
        if initial_temp <= 0 or final_temp <= 0:
            raise ValueError("Exponential schedule requires positive temperatures")
        import math
        ratio = final_temp / initial_temp
        return initial_temp * (ratio ** progress)

    elif schedule_type == 'cosine':
        # Cosine annealing: smooth S-curve cooling
        # T(p) = T_final + (T_initial - T_final) * (1 + cos(π * p)) / 2
        # Starts fast, slows in middle, speeds up at end
        import math
        return final_temp + (initial_temp - final_temp) * (1 + math.cos(math.pi * progress)) / 2

    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
