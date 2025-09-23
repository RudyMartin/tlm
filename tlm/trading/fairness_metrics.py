"""
Diversity, bias, and fairness metrics for trading algorithms and ML models.

Implements comprehensive fairness assessment including Gini variants,
demographic parity, equalized odds, and algorithmic bias detection.
Essential for ethical AI and regulatory compliance in finance.
"""

from typing import List, Dict, Tuple, Optional, NamedTuple, Union
import math
import statistics
from ..pure.ops import sum as asum

# Type definitions
Predictions = List[float]
Labels = List[int]
Groups = List[str]  # Protected attribute groups
Portfolios = List[List[float]]  # Multiple portfolio compositions

class FairnessMetrics(NamedTuple):
    """Container for fairness assessment results."""
    demographic_parity: float
    equalized_odds: float
    calibration: float
    individual_fairness: float
    overall_fairness_score: float

class DiversityMetrics(NamedTuple):
    """Container for diversity assessment results."""
    shannon_diversity: float
    simpson_diversity: float
    berger_parker_dominance: float
    pielou_evenness: float
    effective_number_species: float

class GiniMetrics(NamedTuple):
    """Container for various Gini coefficient calculations."""
    wealth_gini: float          # Standard wealth inequality
    portfolio_gini: float       # Portfolio concentration
    prediction_gini: float      # Model prediction inequality
    demographic_gini: Dict[str, float]  # Group-specific inequalities
    intersectional_gini: float  # Multi-dimensional inequality


# GINI COEFFICIENT VARIANTS

def wealth_gini_coefficient(wealth_distribution: List[float]) -> float:
    """
    Calculate standard Gini coefficient for wealth/income distribution.
    
    Gini = 0: Perfect equality (everyone has same wealth)
    Gini = 1: Perfect inequality (one person has all wealth)
    
    Args:
        wealth_distribution: List of individual wealth amounts
        
    Returns:
        Gini coefficient (0-1)
    """
    if not wealth_distribution or len(wealth_distribution) < 2:
        return 0.0
    
    # Remove negative values and sort
    positive_wealth = [max(0, w) for w in wealth_distribution]
    n = len(positive_wealth)
    
    if asum(positive_wealth) == 0:
        return 0.0  # Everyone has zero wealth
    
    # Sort in ascending order
    sorted_wealth = sorted(positive_wealth)
    
    # Calculate Gini using the formula: G = (2∑(i*x_i))/(n*∑x_i) - (n+1)/n
    total_wealth = asum(sorted_wealth)
    weighted_sum = asum((i + 1) * sorted_wealth[i] for i in range(n))
    
    gini = (2 * weighted_sum) / (n * total_wealth) - (n + 1) / n
    return max(0.0, min(1.0, gini))


def portfolio_gini_coefficient(portfolio_weights: List[float]) -> float:
    """
    Calculate Gini coefficient for portfolio concentration.
    
    Measures how concentrated a portfolio is across assets.
    Used for diversity assessment in asset allocation.
    
    Args:
        portfolio_weights: Asset weights in portfolio
        
    Returns:
        Portfolio concentration Gini (0 = perfectly diversified, 1 = concentrated)
    """
    if not portfolio_weights:
        return 0.0
    
    # Normalize weights to sum to 1
    total_weight = asum(abs(w) for w in portfolio_weights)
    if total_weight == 0:
        return 0.0
    
    normalized_weights = [abs(w) / total_weight for w in portfolio_weights]
    return wealth_gini_coefficient(normalized_weights)


def prediction_gini_coefficient(predictions: Predictions, actual: Labels) -> float:
    """
    Calculate Gini coefficient for model predictions.
    
    Measures inequality in prediction confidence/scores.
    High Gini indicates model is very confident about some predictions
    but uncertain about others.
    
    Args:
        predictions: Model prediction scores
        actual: Actual labels (used for validation)
        
    Returns:
        Prediction inequality Gini coefficient
    """
    if not predictions:
        return 0.0
    
    # Use absolute prediction values to measure confidence spread
    abs_predictions = [abs(p) for p in predictions]
    return wealth_gini_coefficient(abs_predictions)


def demographic_gini_coefficients(outcomes: List[float], groups: Groups) -> Dict[str, float]:
    """
    Calculate group-specific Gini coefficients.
    
    Measures inequality within each demographic group.
    
    Args:
        outcomes: Economic outcomes (returns, wealth, etc.)
        groups: Group membership for each individual
        
    Returns:
        Dictionary mapping group names to their Gini coefficients
    """
    if len(outcomes) != len(groups):
        return {}
    
    group_outcomes = {}
    for outcome, group in zip(outcomes, groups):
        if group not in group_outcomes:
            group_outcomes[group] = []
        group_outcomes[group].append(outcome)
    
    group_ginis = {}
    for group, group_vals in group_outcomes.items():
        group_ginis[group] = wealth_gini_coefficient(group_vals)
    
    return group_ginis


def intersectional_gini_coefficient(outcomes: List[float], 
                                  primary_groups: Groups,
                                  secondary_groups: Groups) -> float:
    """
    Calculate intersectional Gini coefficient.
    
    Measures inequality across combinations of protected attributes
    (e.g., race × gender interactions).
    
    Args:
        outcomes: Economic outcomes
        primary_groups: First protected attribute (e.g., race)
        secondary_groups: Second protected attribute (e.g., gender)
        
    Returns:
        Intersectional Gini coefficient
    """
    if len(outcomes) != len(primary_groups) or len(outcomes) != len(secondary_groups):
        return 0.0
    
    # Create intersectional groups
    intersectional_outcomes = {}
    for outcome, group1, group2 in zip(outcomes, primary_groups, secondary_groups):
        intersectional_group = f"{group1}×{group2}"
        if intersectional_group not in intersectional_outcomes:
            intersectional_outcomes[intersectional_group] = []
        intersectional_outcomes[intersectional_group].append(outcome)
    
    # Calculate group means
    group_means = []
    for group_vals in intersectional_outcomes.values():
        if group_vals:
            group_means.append(asum(group_vals) / len(group_vals))
    
    return wealth_gini_coefficient(group_means)


def calculate_gini_metrics(outcomes: List[float], 
                          portfolio_weights: Optional[List[float]] = None,
                          predictions: Optional[Predictions] = None,
                          groups: Optional[Groups] = None,
                          secondary_groups: Optional[Groups] = None) -> GiniMetrics:
    """
    Calculate comprehensive Gini coefficient metrics.
    
    Args:
        outcomes: Primary outcomes (wealth, returns, etc.)
        portfolio_weights: Optional portfolio weights
        predictions: Optional model predictions
        groups: Optional primary demographic groups
        secondary_groups: Optional secondary demographic groups
        
    Returns:
        GiniMetrics with all calculated Gini variants
    """
    wealth_gini = wealth_gini_coefficient(outcomes)
    
    portfolio_gini = 0.0
    if portfolio_weights:
        portfolio_gini = portfolio_gini_coefficient(portfolio_weights)
    
    prediction_gini = 0.0
    if predictions:
        # Create dummy labels if not provided
        dummy_labels = [1 if p > statistics.median(predictions) else 0 for p in predictions]
        prediction_gini = prediction_gini_coefficient(predictions, dummy_labels)
    
    demographic_ginis = {}
    if groups:
        demographic_ginis = demographic_gini_coefficients(outcomes, groups)
    
    intersectional_gini = 0.0
    if groups and secondary_groups:
        intersectional_gini = intersectional_gini_coefficient(outcomes, groups, secondary_groups)
    
    return GiniMetrics(
        wealth_gini=wealth_gini,
        portfolio_gini=portfolio_gini,
        prediction_gini=prediction_gini,
        demographic_gini=demographic_ginis,
        intersectional_gini=intersectional_gini
    )


# FAIRNESS METRICS

def demographic_parity(predictions: Predictions, groups: Groups, 
                      threshold: float = 0.5) -> float:
    """
    Calculate demographic parity metric.
    
    Measures whether positive prediction rates are equal across groups.
    Perfect fairness = 0, higher values indicate more bias.
    
    Args:
        predictions: Model prediction scores
        groups: Protected attribute groups
        threshold: Classification threshold
        
    Returns:
        Demographic parity violation score (0 = fair, higher = more biased)
    """
    if len(predictions) != len(groups):
        return 1.0  # Maximum unfairness for invalid input
    
    # Convert predictions to binary decisions
    decisions = [1 if p >= threshold else 0 for p in predictions]
    
    # Calculate positive rates by group
    group_rates = {}
    for decision, group in zip(decisions, groups):
        if group not in group_rates:
            group_rates[group] = {'positive': 0, 'total': 0}
        group_rates[group]['total'] += 1
        if decision == 1:
            group_rates[group]['positive'] += 1
    
    # Calculate positive rates
    positive_rates = []
    for group_data in group_rates.values():
        if group_data['total'] > 0:
            rate = group_data['positive'] / group_data['total']
            positive_rates.append(rate)
    
    if len(positive_rates) < 2:
        return 0.0  # Only one group, can't measure disparity
    
    # Calculate max difference in positive rates
    max_rate = max(positive_rates)
    min_rate = min(positive_rates)
    
    return max_rate - min_rate


def equalized_odds(predictions: Predictions, labels: Labels, groups: Groups,
                  threshold: float = 0.5) -> float:
    """
    Calculate equalized odds fairness metric.
    
    Measures whether true positive and false positive rates are equal across groups.
    
    Args:
        predictions: Model prediction scores
        labels: True labels
        groups: Protected attribute groups
        threshold: Classification threshold
        
    Returns:
        Equalized odds violation score (0 = fair, higher = more biased)
    """
    if len(predictions) != len(labels) or len(predictions) != len(groups):
        return 1.0
    
    decisions = [1 if p >= threshold else 0 for p in predictions]
    
    # Calculate TPR and FPR by group
    group_metrics = {}
    for decision, label, group in zip(decisions, labels, groups):
        if group not in group_metrics:
            group_metrics[group] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        
        if label == 1 and decision == 1:
            group_metrics[group]['tp'] += 1
        elif label == 0 and decision == 1:
            group_metrics[group]['fp'] += 1
        elif label == 0 and decision == 0:
            group_metrics[group]['tn'] += 1
        else:  # label == 1 and decision == 0
            group_metrics[group]['fn'] += 1
    
    tprs = []
    fprs = []
    
    for metrics in group_metrics.values():
        # True Positive Rate
        tpr = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
        tprs.append(tpr)
        
        # False Positive Rate
        fpr = metrics['fp'] / (metrics['fp'] + metrics['tn']) if (metrics['fp'] + metrics['tn']) > 0 else 0
        fprs.append(fpr)
    
    if len(tprs) < 2:
        return 0.0
    
    # Calculate disparities
    tpr_disparity = max(tprs) - min(tprs)
    fpr_disparity = max(fprs) - min(fprs)
    
    return max(tpr_disparity, fpr_disparity)


def calibration_fairness(predictions: Predictions, labels: Labels, groups: Groups,
                        n_bins: int = 10) -> float:
    """
    Calculate calibration fairness metric.
    
    Measures whether predicted probabilities are well-calibrated across groups.
    
    Args:
        predictions: Model prediction probabilities
        labels: True binary labels
        groups: Protected attribute groups
        n_bins: Number of calibration bins
        
    Returns:
        Calibration unfairness score (0 = well-calibrated across groups)
    """
    if len(predictions) != len(labels) or len(predictions) != len(groups):
        return 1.0
    
    unique_groups = list(set(groups))
    if len(unique_groups) < 2:
        return 0.0
    
    group_calibrations = {}
    
    for group in unique_groups:
        group_preds = [predictions[i] for i in range(len(groups)) if groups[i] == group]
        group_labels = [labels[i] for i in range(len(groups)) if groups[i] == group]
        
        if len(group_preds) < n_bins:
            continue
        
        # Calculate calibration error for this group
        calibration_error = 0.0
        
        for bin_idx in range(n_bins):
            bin_start = bin_idx / n_bins
            bin_end = (bin_idx + 1) / n_bins
            
            bin_preds = []
            bin_labels = []
            
            for pred, label in zip(group_preds, group_labels):
                if bin_start <= pred < bin_end or (bin_idx == n_bins - 1 and pred == 1.0):
                    bin_preds.append(pred)
                    bin_labels.append(label)
            
            if len(bin_preds) > 0:
                avg_pred = asum(bin_preds) / len(bin_preds)
                avg_actual = asum(bin_labels) / len(bin_labels)
                bin_error = abs(avg_pred - avg_actual)
                calibration_error += bin_error * len(bin_preds) / len(group_preds)
        
        group_calibrations[group] = calibration_error
    
    if len(group_calibrations) < 2:
        return 0.0
    
    calibration_values = list(group_calibrations.values())
    return max(calibration_values) - min(calibration_values)


def individual_fairness(features: List[List[float]], predictions: Predictions,
                       distance_threshold: float = 0.1) -> float:
    """
    Calculate individual fairness metric.
    
    Measures whether similar individuals receive similar predictions.
    
    Args:
        features: Feature vectors for individuals
        predictions: Model predictions
        distance_threshold: Threshold for considering individuals "similar"
        
    Returns:
        Individual unfairness score (0 = fair, higher = more unfair)
    """
    if len(features) != len(predictions) or len(features) < 2:
        return 0.0
    
    unfairness_scores = []
    
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            # Calculate feature distance (Euclidean)
            if len(features[i]) != len(features[j]):
                continue
            
            feature_distance = math.sqrt(asum((features[i][k] - features[j][k])**2 
                                            for k in range(len(features[i]))))
            
            # If individuals are similar (low feature distance)
            if feature_distance < distance_threshold:
                # Calculate prediction difference
                pred_difference = abs(predictions[i] - predictions[j])
                unfairness_scores.append(pred_difference)
    
    if not unfairness_scores:
        return 0.0
    
    return asum(unfairness_scores) / len(unfairness_scores)


def calculate_fairness_metrics(predictions: Predictions, labels: Labels, groups: Groups,
                              features: Optional[List[List[float]]] = None,
                              threshold: float = 0.5) -> FairnessMetrics:
    """
    Calculate comprehensive fairness metrics.
    
    Args:
        predictions: Model predictions
        labels: True labels
        groups: Protected attribute groups
        features: Optional feature vectors for individual fairness
        threshold: Classification threshold
        
    Returns:
        FairnessMetrics with all calculated fairness measures
    """
    demo_parity = demographic_parity(predictions, groups, threshold)
    eq_odds = equalized_odds(predictions, labels, groups, threshold)
    calibration = calibration_fairness(predictions, labels, groups)
    
    individual_fair = 0.0
    if features:
        individual_fair = individual_fairness(features, predictions)
    
    # Calculate overall fairness score (lower is better)
    overall_score = (demo_parity + eq_odds + calibration + individual_fair) / 4
    
    return FairnessMetrics(
        demographic_parity=demo_parity,
        equalized_odds=eq_odds,
        calibration=calibration,
        individual_fairness=individual_fair,
        overall_fairness_score=overall_score
    )


# DIVERSITY METRICS

def shannon_diversity_index(composition: List[float]) -> float:
    """
    Calculate Shannon diversity index.
    
    H = -Σ(p_i * ln(p_i))
    Higher values indicate more diversity.
    
    Args:
        composition: Proportional composition (e.g., portfolio weights, species abundance)
        
    Returns:
        Shannon diversity index
    """
    if not composition:
        return 0.0
    
    # Normalize to proportions
    total = asum(composition)
    if total == 0:
        return 0.0
    
    proportions = [x / total for x in composition if x > 0]
    
    shannon = -asum(p * math.log(p) for p in proportions if p > 0)
    return shannon


def simpson_diversity_index(composition: List[float]) -> float:
    """
    Calculate Simpson diversity index.
    
    D = 1 - Σ(p_i^2)
    Values range from 0-1, higher values indicate more diversity.
    
    Args:
        composition: Proportional composition
        
    Returns:
        Simpson diversity index
    """
    if not composition:
        return 0.0
    
    total = asum(composition)
    if total == 0:
        return 0.0
    
    proportions = [x / total for x in composition]
    simpson = 1 - asum(p**2 for p in proportions)
    
    return max(0.0, simpson)


def berger_parker_dominance(composition: List[float]) -> float:
    """
    Calculate Berger-Parker dominance index.
    
    Dominance = max(p_i) (proportion of most abundant species/asset)
    Higher values indicate less diversity (more dominance).
    
    Args:
        composition: Proportional composition
        
    Returns:
        Berger-Parker dominance index
    """
    if not composition:
        return 1.0
    
    total = asum(composition)
    if total == 0:
        return 1.0
    
    proportions = [x / total for x in composition]
    return max(proportions) if proportions else 1.0


def pielou_evenness(composition: List[float]) -> float:
    """
    Calculate Pielou's evenness index.
    
    J = H / ln(S)
    where H is Shannon diversity and S is number of species/assets
    Values range 0-1, with 1 indicating perfect evenness.
    
    Args:
        composition: Proportional composition
        
    Returns:
        Pielou's evenness index
    """
    if not composition:
        return 0.0
    
    non_zero_count = sum(1 for x in composition if x > 0)
    if non_zero_count <= 1:
        return 1.0 if non_zero_count == 1 else 0.0
    
    shannon = shannon_diversity_index(composition)
    max_diversity = math.log(non_zero_count)
    
    return shannon / max_diversity if max_diversity > 0 else 0.0


def effective_number_of_species(composition: List[float], q: float = 1.0) -> float:
    """
    Calculate effective number of species/assets.
    
    For q=1: Effective number = exp(Shannon diversity)
    For q=2: Effective number = 1/Simpson index
    
    Args:
        composition: Proportional composition
        q: Diversity order parameter
        
    Returns:
        Effective number of species/assets
    """
    if not composition:
        return 0.0
    
    if q == 1.0:
        shannon = shannon_diversity_index(composition)
        return math.exp(shannon)
    elif q == 2.0:
        simpson = simpson_diversity_index(composition)
        return 1.0 / (1.0 - simpson) if simpson < 1.0 else float('inf')
    else:
        # Generalized formula for other q values
        total = asum(composition)
        if total == 0:
            return 0.0
        
        proportions = [x / total for x in composition if x > 0]
        
        if q == 0:
            return len(proportions)  # Species richness
        else:
            hill_number = (asum(p**q for p in proportions))**(1/(1-q))
            return hill_number


def calculate_diversity_metrics(composition: List[float]) -> DiversityMetrics:
    """
    Calculate comprehensive diversity metrics.
    
    Args:
        composition: Proportional composition (portfolio weights, species abundance, etc.)
        
    Returns:
        DiversityMetrics with all calculated diversity measures
    """
    shannon = shannon_diversity_index(composition)
    simpson = simpson_diversity_index(composition)
    dominance = berger_parker_dominance(composition)
    evenness = pielou_evenness(composition)
    effective_species = effective_number_of_species(composition)
    
    return DiversityMetrics(
        shannon_diversity=shannon,
        simpson_diversity=simpson,
        berger_parker_dominance=dominance,
        pielou_evenness=evenness,
        effective_number_species=effective_species
    )


# BIAS DETECTION METHODS

def algorithmic_bias_score(predictions: Predictions, groups: Groups,
                          sensitive_features: List[List[float]]) -> Dict[str, float]:
    """
    Detect algorithmic bias in model predictions.
    
    Args:
        predictions: Model predictions
        groups: Protected attribute groups
        sensitive_features: Sensitive feature values that shouldn't influence predictions
        
    Returns:
        Dictionary of bias scores for different bias types
    """
    bias_scores = {}
    
    # Statistical parity bias
    bias_scores['statistical_parity'] = demographic_parity(predictions, groups)
    
    # Feature correlation bias
    if sensitive_features and len(sensitive_features) == len(predictions):
        correlations = []
        for feature_idx in range(len(sensitive_features[0])):
            feature_values = [row[feature_idx] for row in sensitive_features]
            correlation = _pearson_correlation(feature_values, predictions)
            correlations.append(abs(correlation))
        
        bias_scores['feature_correlation'] = max(correlations) if correlations else 0.0
    
    # Representation bias (group size imbalance)
    group_counts = {}
    for group in groups:
        group_counts[group] = group_counts.get(group, 0) + 1
    
    if len(group_counts) > 1:
        group_sizes = list(group_counts.values())
        representation_gini = wealth_gini_coefficient(group_sizes)
        bias_scores['representation'] = representation_gini
    
    return bias_scores


def intersectional_bias_analysis(outcomes: List[float], 
                                primary_groups: Groups,
                                secondary_groups: Groups) -> Dict[str, Dict[str, float]]:
    """
    Analyze intersectional bias across multiple protected attributes.
    
    Args:
        outcomes: Model outcomes or economic results
        primary_groups: First protected attribute
        secondary_groups: Second protected attribute
        
    Returns:
        Nested dictionary of intersectional bias metrics
    """
    if len(outcomes) != len(primary_groups) or len(outcomes) != len(secondary_groups):
        return {}
    
    # Group outcomes by intersectional categories
    intersectional_groups = {}
    for outcome, group1, group2 in zip(outcomes, primary_groups, secondary_groups):
        key = f"{group1}×{group2}"
        if key not in intersectional_groups:
            intersectional_groups[key] = []
        intersectional_groups[key].append(outcome)
    
    results = {}
    
    # Calculate metrics for each intersectional group
    for group, group_outcomes in intersectional_groups.items():
        group_results = {
            'mean_outcome': asum(group_outcomes) / len(group_outcomes),
            'outcome_std': statistics.stdev(group_outcomes) if len(group_outcomes) > 1 else 0.0,
            'sample_size': len(group_outcomes)
        }
        results[group] = group_results
    
    # Calculate overall intersectional inequality
    group_means = [results[group]['mean_outcome'] for group in results]
    results['_overall'] = {
        'intersectional_gini': wealth_gini_coefficient(group_means),
        'mean_range': max(group_means) - min(group_means) if group_means else 0.0,
        'coefficient_of_variation': statistics.stdev(group_means) / statistics.mean(group_means) if group_means and statistics.mean(group_means) != 0 else 0.0
    }
    
    return results


# HELPER FUNCTIONS

def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    sum_x = asum(x)
    sum_y = asum(y)
    sum_xy = asum(x[i] * y[i] for i in range(n))
    sum_x2 = asum(xi**2 for xi in x)
    sum_y2 = asum(yi**2 for yi in y)
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator