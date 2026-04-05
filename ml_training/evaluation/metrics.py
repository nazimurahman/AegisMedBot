"""
Evaluation Metrics for Medical Models.

This module provides comprehensive metrics for evaluating medical AI models,
including classification metrics, regression metrics, and specialized medical
metrics like AUROC, calibration curves, and clinical utility measures.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    average_precision_score, brier_score_loss, log_loss
)
from scipy.stats import pearsonr, spearmanr
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Comprehensive metrics calculator for medical model evaluation.
    
    This class provides methods to calculate various evaluation metrics
    for classification and regression tasks commonly encountered in
    medical AI applications.
    """
    
    def __init__(self, average_method: str = 'binary'):
        """
        Initialize metrics calculator.
        
        Args:
            average_method: Averaging method for multi-class metrics
                ('binary', 'micro', 'macro', 'weighted')
        """
        self.average_method = average_method
    
    def calculate_classification_metrics(
        self,
        y_true: Union[List, np.ndarray, torch.Tensor],
        y_pred: Union[List, np.ndarray, torch.Tensor],
        y_proba: Optional[Union[List, np.ndarray, torch.Tensor]] = None,
        labels: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for AUC and calibration)
            labels: List of label names
            
        Returns:
            Dictionary containing all calculated metrics
        """
        # Convert to numpy arrays
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(
            y_true, y_pred, average=self.average_method, zero_division=0
        )
        metrics['recall'] = recall_score(
            y_true, y_pred, average=self.average_method, zero_division=0
        )
        metrics['f1_score'] = f1_score(
            y_true, y_pred, average=self.average_method, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Calculate TP, TN, FP, FN for binary classification
        if len(np.unique(y_true)) == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            # Derived metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # AUC metrics (if probabilities available)
        if y_proba is not None:
            y_proba = self._to_numpy(y_proba)
            
            # Handle different probability shapes
            if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
                y_proba_binary = y_proba[:, 1]  # Positive class probability
            else:
                y_proba_binary = y_proba
            
            # ROC AUC
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba_binary)
            
            # Precision-Recall AUC
            metrics['pr_auc'] = average_precision_score(y_true, y_proba_binary)
            
            # Brier score (calibration)
            metrics['brier_score'] = brier_score_loss(y_true, y_proba_binary)
            
            # Log loss
            metrics['log_loss'] = log_loss(y_true, y_proba_binary)
        
        # Classification report
        if labels:
            report = classification_report(
                y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
            )
            metrics['classification_report'] = report
        
        return metrics
    
    def calculate_regression_metrics(
        self,
        y_true: Union[List, np.ndarray, torch.Tensor],
        y_pred: Union[List, np.ndarray, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing regression metrics
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        metrics = {}
        
        # Error metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # Variance explained
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Correlation metrics
        if len(y_true) > 1 and len(y_pred) > 1:
            pearson_corr, pearson_p = pearsonr(y_true, y_pred)
            metrics['pearson_correlation'] = pearson_corr
            metrics['pearson_p_value'] = pearson_p
            
            spearman_corr, spearman_p = spearmanr(y_true, y_pred)
            metrics['spearman_correlation'] = spearman_corr
            metrics['spearman_p_value'] = spearman_p
        
        # Relative errors
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            relative_errors = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / 
                                      y_true[non_zero_mask])
            metrics['mean_relative_error'] = np.mean(relative_errors)
            metrics['median_relative_error'] = np.median(relative_errors)
        
        return metrics
    
    def calculate_medical_specific_metrics(
        self,
        y_true: Union[List, np.ndarray, torch.Tensor],
        y_pred: Union[List, np.ndarray, torch.Tensor],
        y_proba: Optional[Union[List, np.ndarray, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Calculate medical-specific metrics like:
        - Sensitivity and specificity at different thresholds
        - Net reclassification improvement (NRI)
        - Integrated discrimination improvement (IDI)
        - Calibration curves
        - Decision curve analysis
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of medical metrics
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        metrics = {}
        
        # Metrics at different thresholds
        if y_proba is not None:
            y_proba = self._to_numpy(y_proba)
            
            # Calculate metrics at multiple thresholds
            thresholds = np.linspace(0.1, 0.9, 9)
            for threshold in thresholds:
                y_pred_threshold = (y_proba >= threshold).astype(int)
                
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_threshold).ravel()
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                metrics[f'sensitivity_thresh_{threshold:.1f}'] = sensitivity
                metrics[f'specificity_thresh_{threshold:.1f}'] = specificity
                metrics[f'ppv_thresh_{threshold:.1f}'] = ppv
                metrics[f'npv_thresh_{threshold:.1f}'] = npv
            
            # Calculate Youden's index and optimal threshold
            fpr, tpr, thresholds_roc = self._roc_curve(y_true, y_proba)
            youden_index = tpr - fpr
            optimal_idx = np.argmax(youden_index)
            
            metrics['optimal_threshold'] = thresholds_roc[optimal_idx]
            metrics['optimal_sensitivity'] = tpr[optimal_idx]
            metrics['optimal_specificity'] = 1 - fpr[optimal_idx]
        
        return metrics
    
    def _roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_thresholds: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ROC curve points.
        
        Args:
            y_true: Ground truth labels
            y_proba: Predicted probabilities
            n_thresholds: Number of thresholds to use
            
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        from sklearn.metrics import roc_curve
        return roc_curve(y_true, y_proba)
    
    def _to_numpy(self, data: Any) -> np.ndarray:
        """
        Convert input data to numpy array.
        
        Args:
            data: Input data (list, numpy array, or torch tensor)
            
        Returns:
            Numpy array
        """
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, list):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array([data])


class CalibrationMetrics:
    """
    Calibration metrics for probabilistic predictions.
    
    Calculates calibration-related metrics including:
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Reliability diagrams
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize calibration metrics calculator.
        
        Args:
            n_bins: Number of bins for calibration estimation
        """
        self.n_bins = n_bins
    
    def calculate_ece(
        self,
        y_true: Union[List, np.ndarray, torch.Tensor],
        y_proba: Union[List, np.ndarray, torch.Tensor]
    ) -> float:
        """
        Calculate Expected Calibration Error.
        
        ECE = sum(bin_size * |accuracy(bin) - confidence(bin)|)
        
        Args:
            y_true: Ground truth labels
            y_proba: Predicted probabilities
            
        Returns:
            Expected Calibration Error value
        """
        y_true = self._to_numpy(y_true)
        y_proba = self._to_numpy(y_proba)
        
        # Handle multi-class probabilities
        if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]  # Use positive class probability
        
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        
        for i in range(self.n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find predictions in this bin
            in_bin = (y_proba >= bin_lower) & (y_proba < bin_upper)
            if not np.any(in_bin):
                continue
            
            # Calculate bin statistics
            bin_size = np.sum(in_bin) / len(y_proba)
            bin_accuracy = np.mean(y_true[in_bin])
            bin_confidence = np.mean(y_proba[in_bin])
            
            ece += bin_size * np.abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def calculate_mce(
        self,
        y_true: Union[List, np.ndarray, torch.Tensor],
        y_proba: Union[List, np.ndarray, torch.Tensor]
    ) -> float:
        """
        Calculate Maximum Calibration Error.
        
        MCE = max(|accuracy(bin) - confidence(bin)|)
        
        Args:
            y_true: Ground truth labels
            y_proba: Predicted probabilities
            
        Returns:
            Maximum Calibration Error value
        """
        y_true = self._to_numpy(y_true)
        y_proba = self._to_numpy(y_proba)
        
        if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]
        
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        max_calibration_error = 0.0
        
        for i in range(self.n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (y_proba >= bin_lower) & (y_proba < bin_upper)
            if not np.any(in_bin):
                continue
            
            bin_accuracy = np.mean(y_true[in_bin])
            bin_confidence = np.mean(y_proba[in_bin])
            
            calibration_error = np.abs(bin_accuracy - bin_confidence)
            max_calibration_error = max(max_calibration_error, calibration_error)
        
        return max_calibration_error
    
    def get_reliability_diagram_data(
        self,
        y_true: Union[List, np.ndarray, torch.Tensor],
        y_proba: Union[List, np.ndarray, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """
        Get data for reliability diagram.
        
        Args:
            y_true: Ground truth labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary with bin centers, accuracies, and confidences
        """
        y_true = self._to_numpy(y_true)
        y_proba = self._to_numpy(y_proba)
        
        if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]
        
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(self.n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (y_proba >= bin_lower) & (y_proba < bin_upper)
            if np.any(in_bin):
                bin_accuracies.append(np.mean(y_true[in_bin]))
                bin_confidences.append(np.mean(y_proba[in_bin]))
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(0.0)
        
        return {
            'bin_centers': bin_centers,
            'accuracies': np.array(bin_accuracies),
            'confidences': np.array(bin_confidences)
        }
    
    def _to_numpy(self, data: Any) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        return np.array(data)


class ClinicalUtilityMetrics:
    """
    Clinical utility metrics for medical decision-making.
    
    Calculates metrics that assess the practical clinical value
    of predictive models.
    """
    
    def calculate_net_benefit(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate net benefit for decision curve analysis.
        
        Net Benefit = (TP / N) - (FP / N) * (p_t / (1 - p_t))
        
        Args:
            y_true: Ground truth labels
            y_proba: Predicted probabilities
            thresholds: Decision thresholds
            
        Returns:
            Dictionary with thresholds and net benefit values
        """
        y_true = self._to_numpy(y_true)
        y_proba = self._to_numpy(y_proba)
        
        n = len(y_true)
        net_benefits = []
        
        for threshold in thresholds:
            # Make predictions at threshold
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate TP and FP
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            
            # Calculate net benefit
            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
            net_benefits.append(net_benefit)
        
        return {
            'thresholds': thresholds,
            'net_benefit': np.array(net_benefits)
        }
    
    def calculate_number_needed_to_treat(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate Number Needed to Treat (NNT) for binary classification.
        
        NNT = 1 / (event_rate_treated - event_rate_control)
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Number needed to treat value
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        # Event rates in treated and control groups
        treated = y_pred == 1
        control = y_pred == 0
        
        if np.any(treated) and np.any(control):
            event_rate_treated = np.mean(y_true[treated])
            event_rate_control = np.mean(y_true[control])
            
            absolute_risk_reduction = event_rate_control - event_rate_treated
            
            if absolute_risk_reduction > 0:
                return 1 / absolute_risk_reduction
        
        return float('inf')
    
    def _to_numpy(self, data: Any) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        return np.array(data)