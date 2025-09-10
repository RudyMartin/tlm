import math
from typing import List, Optional, Union
from ..pure.ops import asum

# Type definitions for consistency
Scalar = float
Vector = List[Scalar]
Matrix = List[Vector]

__all__ = [
    # Regression losses
    'mse', 'mae', 'rmse', 'msle', 'mape', 'smape',
    'huber_loss', 'log_cosh_loss', 'quantile_loss',
    # Classification losses
    'binary_cross_entropy', 'cross_entropy', 'sparse_cross_entropy',
    'focal_loss', 'hinge_loss', 'squared_hinge_loss', 'f2_loss',
    # Distribution losses
    'bernoulli_loss', 'poisson_loss', 'gamma_loss', 'lucas_loss',
    # Probability losses
    'kl_divergence', 'js_divergence', 'wasserstein_distance',
    # Ranking losses
    'contrastive_loss', 'triplet_loss', 'margin_ranking_loss',
    # Robust losses
    'robust_l1_loss', 'robust_l2_loss', 'tukey_loss',
    # Multi-task losses
    'multi_task_loss', 'weighted_loss',
    # Statistical tests as losses
    'durbin_watson_loss'
]

def mse(y_true: List[float], y_pred: List[float]) -> float:
    """Calculate mean squared error."""
    n = len(y_true)
    return asum([(y_true[i]-y_pred[i])**2 for i in range(n)]) / n

def mae(y_true: List[float], y_pred: List[float]) -> float:
    """Calculate mean absolute error."""
    n = len(y_true)
    return asum([abs(y_true[i]-y_pred[i]) for i in range(n)]) / n

def binary_cross_entropy(p: List[float], y: List[int]) -> float:
    """Calculate binary cross entropy loss."""
    eps = 1e-12
    n = len(y)
    total = 0.0
    for i in range(n):
        pi = min(max(p[i], eps), 1.0-eps)
        yi = y[i]
        total += -(yi*math.log(pi) + (1-yi)*math.log(1-pi))
    return total / n

def cross_entropy(logits: Matrix, y_idx: List[int]) -> float:
    """Calculate categorical cross entropy loss."""
    # logits: (n,K) list; y_idx: (n) ints
    n = len(logits)
    total = 0.0
    for i in range(n):
        row = logits[i]
        m = max(row)
        ex = [math.exp(z - m) for z in row]
        s = sum(ex)
        pi = ex[y_idx[i]] / s
        total += -math.log(max(pi, 1e-12))
    return total / n


# REGRESSION LOSS FUNCTIONS

def rmse(y_true: List[float], y_pred: List[float]) -> float:
    """Calculate root mean squared error."""
    return math.sqrt(mse(y_true, y_pred))


def msle(y_true: List[float], y_pred: List[float]) -> float:
    """Calculate mean squared logarithmic error."""
    n = len(y_true)
    total = 0.0
    for i in range(n):
        y_t = max(0.0, y_true[i])  # Ensure non-negative
        y_p = max(0.0, y_pred[i])
        total += (math.log(1 + y_t) - math.log(1 + y_p)) ** 2
    return total / n


def mape(y_true: List[float], y_pred: List[float], epsilon: float = 1e-8) -> float:
    """Calculate mean absolute percentage error."""
    n = len(y_true)
    total = 0.0
    for i in range(n):
        if abs(y_true[i]) < epsilon:
            continue  # Skip near-zero values
        total += abs((y_true[i] - y_pred[i]) / y_true[i])
    return total / n * 100.0


def smape(y_true: List[float], y_pred: List[float], epsilon: float = 1e-8) -> float:
    """Calculate symmetric mean absolute percentage error."""
    n = len(y_true)
    total = 0.0
    for i in range(n):
        denominator = (abs(y_true[i]) + abs(y_pred[i])) / 2.0
        if denominator < epsilon:
            continue  # Skip near-zero values
        total += abs(y_true[i] - y_pred[i]) / denominator
    return total / n * 100.0


def huber_loss(y_true: List[float], y_pred: List[float], delta: float = 1.0) -> float:
    """Calculate Huber loss (robust to outliers)."""
    n = len(y_true)
    total = 0.0
    for i in range(n):
        error = abs(y_true[i] - y_pred[i])
        if error <= delta:
            total += 0.5 * error ** 2
        else:
            total += delta * error - 0.5 * delta ** 2
    return total / n


def log_cosh_loss(y_true: List[float], y_pred: List[float]) -> float:
    """Calculate log-cosh loss (smooth approximation to MAE)."""
    n = len(y_true)
    total = 0.0
    for i in range(n):
        error = y_pred[i] - y_true[i]
        # Use log(cosh(x)) ≈ |x| - log(2) for large |x| to avoid overflow
        if abs(error) > 10:
            total += abs(error) - math.log(2.0)
        else:
            total += math.log(math.cosh(error))
    return total / n


def quantile_loss(y_true: List[float], y_pred: List[float], quantile: float = 0.5) -> float:
    """Calculate quantile loss (asymmetric loss for quantile regression)."""
    if not 0 < quantile < 1:
        raise ValueError("Quantile must be between 0 and 1")
    
    n = len(y_true)
    total = 0.0
    for i in range(n):
        error = y_true[i] - y_pred[i]
        if error >= 0:
            total += quantile * error
        else:
            total += (quantile - 1) * error
    return total / n


# CLASSIFICATION LOSS FUNCTIONS

def sparse_cross_entropy(logits: List[float], y_true: int) -> float:
    """Calculate sparse categorical cross entropy for single sample."""
    m = max(logits)
    ex = [math.exp(z - m) for z in logits]
    s = sum(ex)
    pi = ex[y_true] / s
    return -math.log(max(pi, 1e-12))


def focal_loss(y_true: List[int], y_pred: List[float], alpha: float = 1.0, 
               gamma: float = 2.0) -> float:
    """Calculate focal loss for addressing class imbalance."""
    eps = 1e-12
    n = len(y_true)
    total = 0.0
    
    for i in range(n):
        p = min(max(y_pred[i], eps), 1.0 - eps)
        y = y_true[i]
        
        if y == 1:
            # Positive class
            total += -alpha * ((1 - p) ** gamma) * math.log(p)
        else:
            # Negative class  
            total += -(1 - alpha) * (p ** gamma) * math.log(1 - p)
    
    return total / n


def hinge_loss(y_true: List[int], y_pred: List[float]) -> float:
    """Calculate hinge loss for SVM."""
    n = len(y_true)
    total = 0.0
    for i in range(n):
        # Convert labels to {-1, 1}
        y = 2 * y_true[i] - 1 if y_true[i] in [0, 1] else y_true[i]
        total += max(0, 1 - y * y_pred[i])
    return total / n


def squared_hinge_loss(y_true: List[int], y_pred: List[float]) -> float:
    """Calculate squared hinge loss."""
    n = len(y_true)
    total = 0.0
    for i in range(n):
        y = 2 * y_true[i] - 1 if y_true[i] in [0, 1] else y_true[i]
        hinge = max(0, 1 - y * y_pred[i])
        total += hinge ** 2
    return total / n


# PROBABILITY LOSS FUNCTIONS

def kl_divergence(p: List[float], q: List[float]) -> float:
    """Calculate Kullback-Leibler divergence."""
    eps = 1e-12
    total = 0.0
    for i in range(len(p)):
        pi = max(p[i], eps)
        qi = max(q[i], eps)
        total += pi * math.log(pi / qi)
    return total


def js_divergence(p: List[float], q: List[float]) -> float:
    """Calculate Jensen-Shannon divergence."""
    # JS divergence is symmetric version of KL divergence
    m = [(p[i] + q[i]) / 2.0 for i in range(len(p))]
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def wasserstein_distance(p: List[float], q: List[float]) -> float:
    """Calculate 1D Wasserstein distance (Earth Mover's Distance)."""
    # For discrete distributions, this is the cumulative difference
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")
    
    # Normalize to ensure they are probability distributions
    p_sum = sum(p)
    q_sum = sum(q)
    
    if p_sum == 0 or q_sum == 0:
        return 0.0
    
    p_norm = [x / p_sum for x in p]
    q_norm = [x / q_sum for x in q]
    
    # Calculate cumulative distributions and their difference
    total = 0.0
    p_cum = 0.0
    q_cum = 0.0
    
    for i in range(len(p)):
        p_cum += p_norm[i]
        q_cum += q_norm[i]
        total += abs(p_cum - q_cum)
    
    return total


# RANKING LOSS FUNCTIONS

def contrastive_loss(y_true: List[int], distances: List[float], margin: float = 1.0) -> float:
    """Calculate contrastive loss for siamese networks."""
    n = len(y_true)
    total = 0.0
    
    for i in range(n):
        d = distances[i]
        if y_true[i] == 1:  # Similar pair
            total += d ** 2
        else:  # Dissimilar pair
            total += max(0, margin - d) ** 2
    
    return total / (2.0 * n)


def triplet_loss(anchor_pred: List[float], positive_pred: List[float], 
                 negative_pred: List[float], margin: float = 1.0) -> float:
    """Calculate triplet loss for learning embeddings."""
    n = len(anchor_pred)
    total = 0.0
    
    for i in range(n):
        # Calculate distances (simplified as absolute difference)
        pos_dist = abs(anchor_pred[i] - positive_pred[i])
        neg_dist = abs(anchor_pred[i] - negative_pred[i])
        total += max(0, pos_dist - neg_dist + margin)
    
    return total / n


def margin_ranking_loss(input1: List[float], input2: List[float], 
                       target: List[int], margin: float = 0.0) -> float:
    """Calculate margin ranking loss."""
    n = len(input1)
    total = 0.0
    
    for i in range(n):
        y = target[i]  # Should be 1 or -1
        total += max(0, -y * (input1[i] - input2[i]) + margin)
    
    return total / n


# ROBUST LOSS FUNCTIONS

def robust_l1_loss(y_true: List[float], y_pred: List[float], threshold: float = 1.0) -> float:
    """Calculate robust L1 loss with threshold."""
    n = len(y_true)
    total = 0.0
    
    for i in range(n):
        error = abs(y_true[i] - y_pred[i])
        if error <= threshold:
            total += error
        else:
            total += threshold  # Cap the loss
    
    return total / n


def robust_l2_loss(y_true: List[float], y_pred: List[float], threshold: float = 1.0) -> float:
    """Calculate robust L2 loss with threshold."""
    n = len(y_true)
    total = 0.0
    
    for i in range(n):
        error = abs(y_true[i] - y_pred[i])
        if error <= threshold:
            total += error ** 2
        else:
            total += threshold ** 2  # Cap the loss
    
    return total / n


def tukey_loss(y_true: List[float], y_pred: List[float], c: float = 4.685) -> float:
    """Calculate Tukey's biweight loss (robust to outliers)."""
    n = len(y_true)
    total = 0.0
    
    for i in range(n):
        error = abs(y_true[i] - y_pred[i])
        if error <= c:
            u = error / c
            total += (c ** 2 / 6.0) * (1 - (1 - u ** 2) ** 3)
        else:
            total += c ** 2 / 6.0  # Maximum loss
    
    return total / n


# MULTI-TASK AND UTILITY FUNCTIONS

def multi_task_loss(losses: List[float], weights: Optional[List[float]] = None) -> float:
    """Calculate weighted multi-task loss."""
    if weights is None:
        weights = [1.0] * len(losses)
    
    if len(losses) != len(weights):
        raise ValueError("Number of losses must match number of weights")
    
    total = 0.0
    weight_sum = 0.0
    
    for i in range(len(losses)):
        total += weights[i] * losses[i]
        weight_sum += weights[i]
    
    return total / weight_sum if weight_sum > 0 else 0.0


def weighted_loss(y_true: List[float], y_pred: List[float], 
                  sample_weights: List[float], loss_fn) -> float:
    """Calculate weighted version of any loss function."""
    if len(y_true) != len(sample_weights):
        raise ValueError("Sample weights must match data length")
    
    # Calculate loss for each sample
    individual_losses = []
    for i in range(len(y_true)):
        loss = loss_fn([y_true[i]], [y_pred[i]])
        individual_losses.append(loss * sample_weights[i])
    
    return sum(individual_losses) / sum(sample_weights)


# F2 LOSS AND STATISTICAL TEST LOSSES

def f2_loss(y_true: List[int], y_pred: List[float], threshold: float = 0.5) -> float:
    """Calculate F2 loss (emphasizes recall over precision)."""
    # Convert predictions to binary
    y_pred_binary = [1 if p >= threshold else 0 for p in y_pred]
    
    # Calculate true positives, false positives, false negatives
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred_binary[i] == 1)
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred_binary[i] == 1)
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred_binary[i] == 0)
    
    # F2 score (beta = 2, emphasizes recall)
    beta = 2.0
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    
    if precision + recall == 0:
        f2_score = 0.0
    else:
        f2_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    
    # Return as loss (1 - F2 score)
    return 1.0 - f2_score


def durbin_watson_loss(residuals: List[float]) -> float:
    """
    Calculate Durbin-Watson statistic as a loss function.
    
    DW statistic tests for autocorrelation in residuals.
    Values near 2.0 indicate no autocorrelation (good).
    Values near 0 or 4 indicate positive/negative autocorrelation (bad).
    
    Args:
        residuals: Residuals from regression model
        
    Returns:
        Loss value: abs(DW - 2.0) so that 0 is optimal
    """
    n = len(residuals)
    if n < 2:
        return 0.0
    
    # Calculate Durbin-Watson statistic
    numerator = 0.0
    denominator = 0.0
    
    for i in range(1, n):
        numerator += (residuals[i] - residuals[i-1]) ** 2
    
    for i in range(n):
        denominator += residuals[i] ** 2
    
    if denominator == 0:
        return 0.0
    
    dw_statistic = numerator / denominator
    
    # Convert to loss: distance from ideal value of 2.0
    return abs(dw_statistic - 2.0)


# DISTRIBUTION-BASED LOSSES

def bernoulli_loss(y_true: List[int], y_pred: List[float]) -> float:
    """
    Calculate Bernoulli negative log-likelihood loss.
    
    Models binary outcomes as Bernoulli distribution.
    Equivalent to binary cross-entropy but framed as distribution loss.
    
    Args:
        y_true: Binary targets (0 or 1)
        y_pred: Predicted probabilities [0, 1]
        
    Returns:
        Negative log-likelihood
    """
    eps = 1e-12
    n = len(y_true)
    total = 0.0
    
    for i in range(n):
        p = min(max(y_pred[i], eps), 1.0 - eps)
        y = y_true[i]
        
        # Bernoulli log-likelihood: y*log(p) + (1-y)*log(1-p)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    
    return total / n


def poisson_loss(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate Poisson negative log-likelihood loss.
    
    Models count data as Poisson distribution.
    Useful for regression on non-negative integer counts.
    
    Args:
        y_true: True counts (non-negative)
        y_pred: Predicted rates (positive)
        
    Returns:
        Negative log-likelihood
    """
    eps = 1e-12
    n = len(y_true)
    total = 0.0
    
    for i in range(n):
        y = max(0.0, y_true[i])  # Ensure non-negative
        rate = max(eps, y_pred[i])  # Ensure positive rate
        
        # Poisson log-likelihood: y*log(rate) - rate - log(y!)
        # We omit log(y!) as it doesn't depend on predictions
        total += -(y * math.log(rate) - rate)
    
    return total / n


def gamma_loss(y_true: List[float], y_pred: List[float], alpha: float = 1.0) -> float:
    """
    Calculate Gamma negative log-likelihood loss.
    
    Models positive continuous data as Gamma distribution.
    Useful for modeling positive skewed data like waiting times.
    
    Args:
        y_true: True positive values
        y_pred: Predicted scale parameters (positive)
        alpha: Shape parameter (fixed)
        
    Returns:
        Negative log-likelihood
    """
    eps = 1e-12
    n = len(y_true)
    total = 0.0
    
    for i in range(n):
        y = max(eps, y_true[i])  # Ensure positive
        scale = max(eps, y_pred[i])  # Ensure positive
        
        # Gamma log-likelihood (simplified, omitting constant terms)
        # -log(Gamma(α)) - α*log(β) + (α-1)*log(y) - y/β
        # Here β = scale, we omit constants
        total += -(alpha - 1) * math.log(y) + y / scale + alpha * math.log(scale)
    
    return total / n


def lucas_loss(y_true: List[float], y_pred: List[float], p: float = 2.0) -> float:
    """
    Calculate Lucas loss (L_p quasi-norm loss).
    
    Generalizes MSE and MAE through parameter p.
    Named after Robert Lucas Jr. for econometric applications.
    
    Args:
        y_true: True values
        y_pred: Predicted values  
        p: Norm parameter (p=1: MAE, p=2: MSE, p>2: more robust)
        
    Returns:
        L_p loss value
    """
    if p <= 0:
        raise ValueError("p must be positive")
    
    n = len(y_true)
    total = 0.0
    
    for i in range(n):
        error = abs(y_true[i] - y_pred[i])
        if p == 1.0:
            total += error
        elif p == 2.0:
            total += error ** 2
        else:
            total += error ** p
    
    return (total / n) ** (1.0 / p) if p != 1.0 and p != 2.0 else total / n
