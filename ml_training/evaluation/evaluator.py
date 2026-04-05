"""
Model Evaluator for Comprehensive Model Assessment.

This module provides a complete evaluation pipeline for medical models,
including:
- Model evaluation on test datasets
- Cross-validation
- Hyperparameter tuning with Optuna
- Model comparison
- Results visualization
- Export of evaluation results
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold, StratifiedKFold
import optuna
from optuna.trial import Trial
import warnings

from .metrics import MetricsCalculator, CalibrationMetrics, ClinicalUtilityMetrics

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluator for medical AI models.
    
    This class handles:
    - Model evaluation with multiple metrics
    - Cross-validation
    - Hyperparameter optimization
    - Model comparison
    - Results visualization
    """
    
    def __init__(
        self,
        model,
        device: str = None,
        output_dir: str = './evaluation_results'
    ):
        """
        Initialize model evaluator.
        
        Args:
            model: PyTorch model to evaluate
            device: Device to use for evaluation
            output_dir: Directory to save evaluation results
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_calc = MetricsCalculator()
        self.calibration_metrics = CalibrationMetrics()
        self.clinical_metrics = ClinicalUtilityMetrics()
        
        self.model.to(self.device)
        
        logger.info(f"Evaluator initialized with device: {self.device}")
    
    def evaluate(
        self,
        test_loader: DataLoader,
        model_name: str = None,
        save_results: bool = True,
        plot_curves: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on test dataset.
        
        Args:
            test_loader: DataLoader for test data
            model_name: Name of the model being evaluated
            save_results: Whether to save results to file
            plot_curves: Whether to generate and save plots
            
        Returns:
            Dictionary containing evaluation results
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probas = []
        
        logger.info("Running evaluation on test set...")
        
        with torch.no_grad():
            for batch in test_loader:
                # Move data to device
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                    labels = batch['labels'].to(self.device)
                else:
                    # Assume batch is tuple (inputs, labels)
                    inputs, labels = batch
                    if isinstance(inputs, dict):
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    else:
                        inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                # Store results
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probas.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_proba = np.array(all_probas)
        
        # Calculate metrics
        results = {
            'model_name': model_name or self.model.__class__.__name__,
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(y_true)
        }
        
        # Classification metrics
        if len(np.unique(y_true)) == 2:
            # Binary classification
            classification_metrics = self.metrics_calc.calculate_classification_metrics(
                y_true, y_pred, y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba
            )
        else:
            # Multi-class classification
            classification_metrics = self.metrics_calc.calculate_classification_metrics(
                y_true, y_pred, y_proba
            )
        
        results.update(classification_metrics)
        
        # Calibration metrics
        if len(np.unique(y_true)) == 2 and y_proba.shape[1] == 2:
            results['ece'] = self.calibration_metrics.calculate_ece(y_true, y_proba[:, 1])
            results['mce'] = self.calibration_metrics.calculate_mce(y_true, y_proba[:, 1])
            
            # Get calibration data
            calibration_data = self.calibration_metrics.get_reliability_diagram_data(
                y_true, y_proba[:, 1]
            )
            results['calibration_data'] = calibration_data
        
        # Clinical utility metrics
        if len(np.unique(y_true)) == 2 and y_proba.shape[1] == 2:
            thresholds = np.linspace(0.05, 0.95, 19)
            net_benefit = self.clinical_metrics.calculate_net_benefit(
                y_true, y_proba[:, 1], thresholds
            )
            results['net_benefit_data'] = {
                'thresholds': net_benefit['thresholds'].tolist(),
                'net_benefit': net_benefit['net_benefit'].tolist()
            }
        
        # Log results
        logger.info("=" * 60)
        logger.info(f"Evaluation Results for {model_name or 'model'}:")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"F1 Score: {results['f1_score']:.4f}")
        
        if 'roc_auc' in results:
            logger.info(f"ROC AUC: {results['roc_auc']:.4f}")
            logger.info(f"ECE: {results['ece']:.4f}")
        
        logger.info("=" * 60)
        
        # Save results
        if save_results:
            self._save_results(results)
        
        # Generate plots
        if plot_curves:
            self._plot_evaluation_results(y_true, y_pred, y_proba, results)
        
        return results
    
    def cross_validate(
        self,
        dataset: Dataset,
        n_folds: int = 5,
        stratified: bool = True,
        batch_size: int = 32,
        model_class: type = None,
        model_kwargs: Dict = None,
        **train_kwargs
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            dataset: PyTorch dataset
            n_folds: Number of folds
            stratified: Whether to use stratified splitting
            batch_size: Batch size for evaluation
            model_class: Class to instantiate new models
            model_kwargs: Arguments for model instantiation
            **train_kwargs: Arguments for training function
            
        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Starting {n_folds}-fold cross-validation")
        
        # Create splitter
        if stratified:
            # Get labels for stratification
            labels = [dataset[i][1] for i in range(len(dataset))]
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            folds = list(splitter.split(np.zeros(len(dataset)), labels))
        else:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            folds = list(splitter.split(np.zeros(len(dataset))))
        
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            logger.info(f"\n{'='*60}")
            logger.info(f"Fold {fold_idx + 1}/{n_folds}")
            logger.info(f"{'='*60}")
            
            # Create data loaders
            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            val_dataset = torch.utils.data.Subset(dataset, val_idx)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Create and train model
            if model_class:
                model = model_class(**(model_kwargs or {}))
            else:
                # Clone existing model
                model = self._clone_model()
            
            # Train model (implement training function)
            trained_model = self._train_fold(model, train_loader, **train_kwargs)
            
            # Evaluate on validation set
            eval_results = self.evaluate(val_loader, save_results=False, plot_curves=False)
            
            # Store results
            eval_results['fold'] = fold_idx
            fold_results.append(eval_results)
        
        # Aggregate results
        aggregated_results = self._aggregate_cv_results(fold_results)
        
        # Save cross-validation results
        cv_results_path = self.output_dir / 'cross_validation_results.json'
        with open(cv_results_path, 'w') as f:
            json.dump({
                'n_folds': n_folds,
                'fold_results': fold_results,
                'aggregated': aggregated_results
            }, f, indent=2)
        
        logger.info(f"Cross-validation results saved to {cv_results_path}")
        
        return aggregated_results
    
    def hyperparameter_optimization(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_class: type,
        n_trials: int = 50,
        timeout: int = None,
        **fixed_params
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization using Optuna.
        
        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            model_class: Model class to instantiate
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            **fixed_params: Fixed model parameters
            
        Returns:
            Dictionary with best parameters and trial results
        """
        def objective(trial: Trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            # Create model with suggested parameters
            model_params = {
                'dropout_rate': dropout_rate,
                **fixed_params
            }
            model = model_class(**model_params)
            
            # Update batch size
            train_loader.dataset.batch_size = batch_size
            
            # Train model
            trained_model = self._train_fold(
                model, train_loader,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                **fixed_params
            )
            
            # Evaluate
            eval_results = self.evaluate(val_loader, save_results=False, plot_curves=False)
            
            # Return validation metric to optimize
            return eval_results.get('roc_auc', eval_results.get('f1_score', 0))
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name='medintel_hpo',
            storage=None
        )
        
        # Run optimization
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Save results
        best_params = study.best_params
        best_value = study.best_value
        
        hpo_results = {
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': n_trials,
            'best_trial': study.best_trial.number,
            'all_trials': [
                {
                    'params': trial.params,
                    'value': trial.value,
                    'number': trial.number
                }
                for trial in study.trials
            ]
        }
        
        # Save HPO results
        hpo_path = self.output_dir / 'hyperparameter_optimization.json'
        with open(hpo_path, 'w') as f:
            json.dump(hpo_results, f, indent=2)
        
        logger.info(f"Hyperparameter optimization completed")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best value: {best_value:.4f}")
        
        return hpo_results
    
    def compare_models(
        self,
        models: Dict[str, Any],
        test_loader: DataLoader
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same test set.
        
        Args:
            models: Dictionary mapping model names to model instances
            test_loader: DataLoader for test data
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            self.model = model
            eval_results = self.evaluate(test_loader, model_name=model_name, save_results=False)
            
            # Extract key metrics
            result = {
                'Model': model_name,
                'Accuracy': eval_results.get('accuracy', 0),
                'F1 Score': eval_results.get('f1_score', 0),
                'Precision': eval_results.get('precision', 0),
                'Recall': eval_results.get('recall', 0),
                'ROC AUC': eval_results.get('roc_auc', 0),
                'PR AUC': eval_results.get('pr_auc', 0),
                'ECE': eval_results.get('ece', 0)
            }
            results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save comparison
        comparison_path = self.output_dir / 'model_comparison.csv'
        df.to_csv(comparison_path, index=False)
        
        logger.info(f"Model comparison saved to {comparison_path}")
        
        return df
    
    def _train_fold(self, model, train_loader, **kwargs):
        """
        Train a model for one fold.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            **kwargs: Training arguments
            
        Returns:
            Trained model
        """
        # Import here to avoid circular imports
        from ..training.trainer import ModelTrainer
        
        # Get training parameters
        learning_rate = kwargs.get('learning_rate', 2e-5)
        weight_decay = kwargs.get('weight_decay', 0.01)
        num_epochs = kwargs.get('num_epochs', 3)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        # Create trainer
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=train_loader,  # Use same for validation (just for training)
            optimizer=optimizer,
            criterion=criterion,
            config={'num_epochs': num_epochs}
        )
        
        # Train
        trainer.train()
        
        return model
    
    def _clone_model(self):
        """Create a clone of the current model."""
        import copy
        return copy.deepcopy(self.model)
    
    def _aggregate_cv_results(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate cross-validation results.
        
        Args:
            fold_results: List of fold evaluation results
            
        Returns:
            Dictionary with aggregated metrics
        """
        aggregated = {
            'mean': {},
            'std': {},
            'min': {},
            'max': {}
        }
        
        # Collect all metric names
        metric_names = set()
        for result in fold_results:
            metric_names.update(result.keys())
        
        # Remove non-metric fields
        metric_names.discard('model_name')
        metric_names.discard('timestamp')
        metric_names.discard('fold')
        metric_names.discard('num_samples')
        metric_names.discard('confusion_matrix')
        metric_names.discard('classification_report')
        metric_names.discard('calibration_data')
        metric_names.discard('net_benefit_data')
        
        # Aggregate each metric
        for metric in metric_names:
            values = [r.get(metric, 0) for r in fold_results]
            aggregated['mean'][metric] = np.mean(values)
            aggregated['std'][metric] = np.std(values)
            aggregated['min'][metric] = np.min(values)
            aggregated['max'][metric] = np.max(values)
        
        return aggregated
    
    def _save_results(self, results: Dict[str, Any]):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results dictionary
        """
        # Save main results
        results_path = self.output_dir / f'evaluation_results_{results["timestamp"]}.json'
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, np.float32) or isinstance(value, np.float64):
                serializable_results[key] = float(value)
            elif isinstance(value, np.int32) or isinstance(value, np.int64):
                serializable_results[key] = int(value)
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
    
    def _plot_evaluation_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        results: Dict[str, Any]
    ):
        """
        Generate and save evaluation plots.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            results: Evaluation results
        """
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        if 'confusion_matrix' in results:
            plt.figure(figsize=(8, 6))
            cm = np.array(results['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(plots_dir / 'confusion_matrix.png')
            plt.close()
        
        # ROC Curve for binary classification
        if 'roc_auc' in results and y_proba.shape[1] == 2:
            from sklearn.metrics import roc_curve
            
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {results["roc_auc"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(plots_dir / 'roc_curve.png')
            plt.close()
        
        # Calibration curve
        if 'calibration_data' in results:
            cal_data = results['calibration_data']
            
            plt.figure(figsize=(8, 6))
            plt.plot(cal_data['bin_centers'], cal_data['confidences'], 
                     'o-', label='Confidence')
            plt.plot(cal_data['bin_centers'], cal_data['accuracies'], 
                     's-', label='Accuracy')
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            plt.xlabel('Predicted Probability')
            plt.ylabel('Observed Frequency')
            plt.title('Reliability Diagram')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(plots_dir / 'reliability_diagram.png')
            plt.close()
        
        logger.info(f"Plots saved to {plots_dir}")