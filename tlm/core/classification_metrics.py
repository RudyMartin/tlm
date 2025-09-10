"""
Comprehensive classification metrics derived from confusion matrix.

This module provides all essential metrics for evaluating classification models,
particularly important for the TidyLLM ecosystem where we need to evaluate
document classification, sentiment analysis, and other ML tasks.

All metrics are implemented in pure Python with zero dependencies.
"""

from typing import List, Optional, Dict, Tuple, Union
import math

# Type definitions
Scalar = float
Vector = List[Scalar]
Matrix = List[Vector]

__all__ = [
    # Binary classification metrics
    'true_positives', 'true_negatives', 'false_positives', 'false_negatives',
    'precision', 'recall', 'f1_score', 'specificity', 'sensitivity',
    'balanced_accuracy', 'matthews_correlation_coefficient', 'cohen_kappa',
    
    # Multi-class metrics
    'precision_multi', 'recall_multi', 'f1_score_multi',
    'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1',
    'weighted_avg_precision', 'weighted_avg_recall', 'weighted_avg_f1',
    'micro_avg_precision', 'micro_avg_recall', 'micro_avg_f1',
    
    # Advanced metrics
    'classification_report', 'roc_auc_binary', 'precision_recall_curve',
    'average_precision_score', 'log_loss', 'brier_score',
    
    # Utility functions
    'confusion_matrix_binary', 'confusion_matrix_metrics',
    'support_per_class', 'class_distribution'
]

# ===========================
# Binary Classification Metrics
# ===========================

def confusion_matrix_binary(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    """
    Compute TP, TN, FP, FN for binary classification.
    Assumes labels are 0 and 1.
    
    Returns:
        (TP, TN, FP, FN)
    """
    tp = tn = fp = fn = 0
    
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
    
    return tp, tn, fp, fn

def true_positives(y_true: List[int], y_pred: List[int]) -> int:
    """Count true positives."""
    tp, _, _, _ = confusion_matrix_binary(y_true, y_pred)
    return tp

def true_negatives(y_true: List[int], y_pred: List[int]) -> int:
    """Count true negatives."""
    _, tn, _, _ = confusion_matrix_binary(y_true, y_pred)
    return tn

def false_positives(y_true: List[int], y_pred: List[int]) -> int:
    """Count false positives."""
    _, _, fp, _ = confusion_matrix_binary(y_true, y_pred)
    return fp

def false_negatives(y_true: List[int], y_pred: List[int]) -> int:
    """Count false negatives."""
    _, _, _, fn = confusion_matrix_binary(y_true, y_pred)
    return fn

def precision(y_true: List[int], y_pred: List[int], zero_division: float = 0.0) -> float:
    """
    Precision: TP / (TP + FP)
    How many predicted positives are actually positive?
    """
    tp, _, fp, _ = confusion_matrix_binary(y_true, y_pred)
    
    if tp + fp == 0:
        return zero_division
    
    return tp / (tp + fp)

def recall(y_true: List[int], y_pred: List[int], zero_division: float = 0.0) -> float:
    """
    Recall (Sensitivity, True Positive Rate): TP / (TP + FN)
    How many actual positives were correctly identified?
    """
    tp, _, _, fn = confusion_matrix_binary(y_true, y_pred)
    
    if tp + fn == 0:
        return zero_division
    
    return tp / (tp + fn)

def sensitivity(y_true: List[int], y_pred: List[int], zero_division: float = 0.0) -> float:
    """Sensitivity is another name for recall."""
    return recall(y_true, y_pred, zero_division)

def specificity(y_true: List[int], y_pred: List[int], zero_division: float = 0.0) -> float:
    """
    Specificity (True Negative Rate): TN / (TN + FP)
    How many actual negatives were correctly identified?
    """
    _, tn, fp, _ = confusion_matrix_binary(y_true, y_pred)
    
    if tn + fp == 0:
        return zero_division
    
    return tn / (tn + fp)

def f1_score(y_true: List[int], y_pred: List[int], zero_division: float = 0.0) -> float:
    """
    F1 Score: Harmonic mean of precision and recall.
    2 * (precision * recall) / (precision + recall)
    """
    prec = precision(y_true, y_pred, zero_division)
    rec = recall(y_true, y_pred, zero_division)
    
    if prec + rec == 0:
        return zero_division
    
    return 2 * (prec * rec) / (prec + rec)

def balanced_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """
    Balanced Accuracy: Average of sensitivity and specificity.
    Useful for imbalanced datasets.
    """
    sens = sensitivity(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    
    return (sens + spec) / 2

def matthews_correlation_coefficient(y_true: List[int], y_pred: List[int]) -> float:
    """
    Matthews Correlation Coefficient (MCC).
    Ranges from -1 to 1. 0 means no better than random.
    
    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    """
    tp, tn, fp, fn = confusion_matrix_binary(y_true, y_pred)
    
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

def cohen_kappa(y_true: List[int], y_pred: List[int]) -> float:
    """
    Cohen's Kappa: Measure of agreement between two raters.
    Accounts for agreement by chance.
    
    κ = (p_o - p_e) / (1 - p_e)
    where p_o is observed agreement and p_e is expected agreement.
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    
    tp, tn, fp, fn = confusion_matrix_binary(y_true, y_pred)
    
    # Observed agreement
    p_o = (tp + tn) / n
    
    # Expected agreement
    n_true_pos = tp + fn  # Actual positives
    n_pred_pos = tp + fp  # Predicted positives
    n_true_neg = tn + fp  # Actual negatives
    n_pred_neg = tn + fn  # Predicted negatives
    
    p_e = ((n_true_pos * n_pred_pos) + (n_true_neg * n_pred_neg)) / (n * n)
    
    if p_e == 1:
        return 1.0 if p_o == 1 else 0.0
    
    return (p_o - p_e) / (1 - p_e)

# ===========================
# Multi-class Metrics
# ===========================

def confusion_matrix_metrics(cm: Matrix) -> Dict[str, List[float]]:
    """
    Extract per-class TP, TN, FP, FN from confusion matrix.
    
    Returns dict with keys: 'tp', 'tn', 'fp', 'fn'
    Each value is a list of metrics per class.
    """
    n_classes = len(cm)
    tp = [0] * n_classes
    tn = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes
    
    for k in range(n_classes):
        # True Positives: diagonal element
        tp[k] = cm[k][k]
        
        # False Positives: sum of column k except diagonal
        fp[k] = sum(cm[i][k] for i in range(n_classes) if i != k)
        
        # False Negatives: sum of row k except diagonal
        fn[k] = sum(cm[k][j] for j in range(n_classes) if j != k)
        
        # True Negatives: all other elements
        tn[k] = sum(cm[i][j] for i in range(n_classes) for j in range(n_classes) 
                   if i != k and j != k)
    
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

def precision_multi(cm: Matrix, average: str = 'macro', zero_division: float = 0.0) -> Union[float, List[float]]:
    """
    Precision for multi-class classification.
    
    average: 'macro', 'micro', 'weighted', or None for per-class
    """
    metrics = confusion_matrix_metrics(cm)
    n_classes = len(cm)
    
    # Per-class precision
    precisions = []
    for k in range(n_classes):
        tp = metrics['tp'][k]
        fp = metrics['fp'][k]
        if tp + fp == 0:
            precisions.append(zero_division)
        else:
            precisions.append(tp / (tp + fp))
    
    if average is None:
        return precisions
    elif average == 'macro':
        return sum(precisions) / n_classes
    elif average == 'micro':
        total_tp = sum(metrics['tp'])
        total_fp = sum(metrics['fp'])
        if total_tp + total_fp == 0:
            return zero_division
        return total_tp / (total_tp + total_fp)
    elif average == 'weighted':
        support = support_per_class(cm)
        total = sum(support)
        if total == 0:
            return zero_division
        return sum(p * s for p, s in zip(precisions, support)) / total
    else:
        raise ValueError(f"Unknown average type: {average}")

def recall_multi(cm: Matrix, average: str = 'macro', zero_division: float = 0.0) -> Union[float, List[float]]:
    """
    Recall for multi-class classification.
    
    average: 'macro', 'micro', 'weighted', or None for per-class
    """
    metrics = confusion_matrix_metrics(cm)
    n_classes = len(cm)
    
    # Per-class recall
    recalls = []
    for k in range(n_classes):
        tp = metrics['tp'][k]
        fn = metrics['fn'][k]
        if tp + fn == 0:
            recalls.append(zero_division)
        else:
            recalls.append(tp / (tp + fn))
    
    if average is None:
        return recalls
    elif average == 'macro':
        return sum(recalls) / n_classes
    elif average == 'micro':
        total_tp = sum(metrics['tp'])
        total_fn = sum(metrics['fn'])
        if total_tp + total_fn == 0:
            return zero_division
        return total_tp / (total_tp + total_fn)
    elif average == 'weighted':
        support = support_per_class(cm)
        total = sum(support)
        if total == 0:
            return zero_division
        return sum(r * s for r, s in zip(recalls, support)) / total
    else:
        raise ValueError(f"Unknown average type: {average}")

def f1_score_multi(cm: Matrix, average: str = 'macro', zero_division: float = 0.0) -> Union[float, List[float]]:
    """
    F1 score for multi-class classification.
    
    average: 'macro', 'micro', 'weighted', or None for per-class
    """
    if average == 'micro':
        # For micro-average, F1 = precision = recall
        prec = precision_multi(cm, average='micro', zero_division=zero_division)
        return prec
    
    # Get per-class precision and recall
    precisions = precision_multi(cm, average=None, zero_division=zero_division)
    recalls = recall_multi(cm, average=None, zero_division=zero_division)
    
    # Calculate per-class F1
    f1_scores = []
    for p, r in zip(precisions, recalls):
        if p + r == 0:
            f1_scores.append(zero_division)
        else:
            f1_scores.append(2 * p * r / (p + r))
    
    if average is None:
        return f1_scores
    elif average == 'macro':
        return sum(f1_scores) / len(f1_scores)
    elif average == 'weighted':
        support = support_per_class(cm)
        total = sum(support)
        if total == 0:
            return zero_division
        return sum(f * s for f, s in zip(f1_scores, support)) / total
    else:
        raise ValueError(f"Unknown average type: {average}")

# Convenience functions for common averaging strategies
def macro_avg_precision(cm: Matrix, zero_division: float = 0.0) -> float:
    """Macro-averaged precision."""
    return precision_multi(cm, average='macro', zero_division=zero_division)

def macro_avg_recall(cm: Matrix, zero_division: float = 0.0) -> float:
    """Macro-averaged recall."""
    return recall_multi(cm, average='macro', zero_division=zero_division)

def macro_avg_f1(cm: Matrix, zero_division: float = 0.0) -> float:
    """Macro-averaged F1 score."""
    return f1_score_multi(cm, average='macro', zero_division=zero_division)

def weighted_avg_precision(cm: Matrix, zero_division: float = 0.0) -> float:
    """Weighted-averaged precision."""
    return precision_multi(cm, average='weighted', zero_division=zero_division)

def weighted_avg_recall(cm: Matrix, zero_division: float = 0.0) -> float:
    """Weighted-averaged recall."""
    return recall_multi(cm, average='weighted', zero_division=zero_division)

def weighted_avg_f1(cm: Matrix, zero_division: float = 0.0) -> float:
    """Weighted-averaged F1 score."""
    return f1_score_multi(cm, average='weighted', zero_division=zero_division)

def micro_avg_precision(cm: Matrix, zero_division: float = 0.0) -> float:
    """Micro-averaged precision."""
    return precision_multi(cm, average='micro', zero_division=zero_division)

def micro_avg_recall(cm: Matrix, zero_division: float = 0.0) -> float:
    """Micro-averaged recall."""
    return recall_multi(cm, average='micro', zero_division=zero_division)

def micro_avg_f1(cm: Matrix, zero_division: float = 0.0) -> float:
    """Micro-averaged F1 score."""
    return f1_score_multi(cm, average='micro', zero_division=zero_division)

# ===========================
# Utility Functions
# ===========================

def support_per_class(cm: Matrix) -> List[int]:
    """
    Get support (number of true instances) per class from confusion matrix.
    """
    return [sum(row) for row in cm]

def class_distribution(y: List[int]) -> Dict[int, int]:
    """
    Get distribution of classes in labels.
    """
    dist = {}
    for label in y:
        dist[label] = dist.get(label, 0) + 1
    return dist

def classification_report(y_true: List[int], y_pred: List[int], 
                         target_names: Optional[List[str]] = None,
                         digits: int = 2) -> str:
    """
    Generate a text report showing main classification metrics.
    
    Similar to sklearn's classification_report but in pure Python.
    """
    from ..core.metrics import confusion_matrix
    
    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(cm)
    
    # Get per-class metrics
    precisions = precision_multi(cm, average=None)
    recalls = recall_multi(cm, average=None)
    f1_scores = f1_score_multi(cm, average=None)
    support = support_per_class(cm)
    
    # Get averaged metrics
    macro_prec = precision_multi(cm, average='macro')
    macro_rec = recall_multi(cm, average='macro')
    macro_f1 = f1_score_multi(cm, average='macro')
    
    weighted_prec = precision_multi(cm, average='weighted')
    weighted_rec = recall_multi(cm, average='weighted')
    weighted_f1 = f1_score_multi(cm, average='weighted')
    
    # Overall accuracy
    accuracy = sum(cm[i][i] for i in range(n_classes)) / sum(sum(row) for row in cm)
    
    # Build report
    if target_names is None:
        target_names = [str(i) for i in range(n_classes)]
    
    # Find the longest target name for formatting
    name_width = max(len(name) for name in target_names + ['accuracy', 'macro avg', 'weighted avg'])
    
    headers = ['precision', 'recall', 'f1-score', 'support']
    header_fmt = f"{{:>{name_width}s}} " + " {:>9s}" * len(headers)
    row_fmt = f"{{:>{name_width}s}} " + f" {{:>9.{digits}f}}" * 3 + " {:>9d}"
    
    report = []
    report.append(header_fmt.format('', *headers))
    report.append('')
    
    # Per-class metrics
    for i, name in enumerate(target_names):
        report.append(row_fmt.format(name, precisions[i], recalls[i], f1_scores[i], support[i]))
    
    report.append('')
    
    # Accuracy
    accuracy_fmt = f"{{:>{name_width}s}} " + " " * 9 + " " * 9 + f" {{:>9.{digits}f}}" + " {:>9d}"
    report.append(accuracy_fmt.format('accuracy', accuracy, sum(support)))
    
    # Macro average
    macro_fmt = f"{{:>{name_width}s}} " + f" {{:>9.{digits}f}}" * 3 + " {:>9d}"
    report.append(macro_fmt.format('macro avg', macro_prec, macro_rec, macro_f1, sum(support)))
    
    # Weighted average
    report.append(macro_fmt.format('weighted avg', weighted_prec, weighted_rec, weighted_f1, sum(support)))
    
    return '\n'.join(report)

# ===========================
# Advanced Metrics
# ===========================

def roc_auc_binary(y_true: List[int], y_scores: List[float]) -> float:
    """
    Calculate ROC AUC for binary classification.
    
    y_scores: Probability scores for the positive class.
    
    Uses the Wilcoxon-Mann-Whitney statistic method.
    """
    # Count positives and negatives
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # No discrimination possible
    
    # Use Wilcoxon-Mann-Whitney statistic
    # AUC = P(score_positive > score_negative)
    concordant = 0
    total = 0
    
    for i, (true_i, score_i) in enumerate(zip(y_true, y_scores)):
        for j, (true_j, score_j) in enumerate(zip(y_true, y_scores)):
            if true_i == 1 and true_j == 0:  # positive vs negative
                total += 1
                if score_i > score_j:
                    concordant += 1
                elif score_i == score_j:
                    concordant += 0.5  # Tie handling
    
    if total == 0:
        return 0.5
    
    return concordant / total

def precision_recall_curve(y_true: List[int], y_scores: List[float], 
                          thresholds: Optional[List[float]] = None) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute precision-recall curve.
    
    Returns:
        (precisions, recalls, thresholds)
    """
    if thresholds is None:
        # Use unique scores as thresholds
        thresholds = sorted(set(y_scores), reverse=True)
    
    precisions = []
    recalls = []
    used_thresholds = []
    
    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        
        tp, _, fp, fn = confusion_matrix_binary(y_true, y_pred)
        
        if tp + fp == 0:
            prec = 1.0  # No positive predictions
        else:
            prec = tp / (tp + fp)
        
        if tp + fn == 0:
            rec = 0.0  # No actual positives
        else:
            rec = tp / (tp + fn)
        
        precisions.append(prec)
        recalls.append(rec)
        used_thresholds.append(threshold)
    
    return precisions, recalls, used_thresholds

def average_precision_score(y_true: List[int], y_scores: List[float]) -> float:
    """
    Compute average precision (area under precision-recall curve).
    
    Uses trapezoidal rule.
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    
    # Sort by recall for proper integration
    sorted_points = sorted(zip(recalls, precisions))
    recalls = [r for r, _ in sorted_points]
    precisions = [p for _, p in sorted_points]
    
    # Compute area using trapezoidal rule
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i-1]) * (precisions[i] + precisions[i-1]) / 2
    
    return ap

def log_loss(y_true: List[int], y_pred_proba: List[float], eps: float = 1e-15) -> float:
    """
    Binary cross-entropy loss / log loss.
    
    y_pred_proba: Predicted probabilities for positive class.
    eps: Small value to avoid log(0).
    """
    n = len(y_true)
    loss = 0.0
    
    for true, pred_prob in zip(y_true, y_pred_proba):
        # Clip probabilities to avoid log(0)
        pred_prob = max(min(pred_prob, 1 - eps), eps)
        
        if true == 1:
            loss -= math.log(pred_prob)
        else:
            loss -= math.log(1 - pred_prob)
    
    return loss / n

def brier_score(y_true: List[int], y_pred_proba: List[float]) -> float:
    """
    Brier score for binary classification.
    
    Measures the mean squared difference between predicted probabilities 
    and actual outcomes. Lower is better.
    """
    n = len(y_true)
    score = 0.0
    
    for true, pred_prob in zip(y_true, y_pred_proba):
        score += (pred_prob - true) ** 2
    
    return score / n