"""
Clinical feature extraction module for risk prediction.

This module handles:
- Extraction of clinical features from EHR data
- Normalization and standardization
- Temporal feature engineering
- Missing data imputation
- Feature selection and dimensionality reduction
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class ClinicalFeatureExtractor:
    """
    Feature extraction and preprocessing for clinical data.
    
    Extracts and processes features from:
    - Patient demographics
    - Vital signs time series
    - Laboratory results
    - Medications
    - Diagnoses and procedures
    - Previous admissions
    
    Produces normalized feature matrices for model input.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration dictionary with parameters:
                - vital_signs: List of vital sign types to extract
                - lab_tests: List of lab tests to extract
                - demographics: List of demographic features
                - window_sizes: List of aggregation windows (hours)
                - imputation_strategy: Strategy for missing data
                - normalization: Whether to normalize features
        """
        self.config = config or {}
        
        # Feature categories
        self.vital_signs = self.config.get("vital_signs", [
            "heart_rate",
            "blood_pressure_systolic",
            "blood_pressure_diastolic",
            "respiratory_rate",
            "temperature",
            "oxygen_saturation"
        ])
        
        self.lab_tests = self.config.get("lab_tests", [
            "wbc",  # White blood cell count
            "hemoglobin",
            "platelets",
            "creatinine",
            "bun",  # Blood urea nitrogen
            "glucose",
            "sodium",
            "potassium",
            "lactate",
            "troponin",
            "crp",  # C-reactive protein
            "procalcitonin"
        ])
        
        self.demographics = self.config.get("demographics", [
            "age",
            "gender",
            "bmi",
            "race"
        ])
        
        self.comorbidities = self.config.get("comorbidities", [
            "hypertension",
            "diabetes",
            "heart_disease",
            "ckd",  # Chronic kidney disease
            "copd",  # Chronic obstructive pulmonary disease
            "cancer"
        ])
        
        # Temporal windows for aggregation (hours)
        self.window_sizes = self.config.get("window_sizes", [1, 6, 24, 48, 168])
        
        # Feature statistics for normalization
        self.feature_stats = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Missing data handler
        self.imputer = SimpleImputer(
            strategy=self.config.get("imputation_strategy", "median")
        )
        
        # Feature dimension cache
        self._feature_dim = None
        
        logger.info(f"ClinicalFeatureExtractor initialized with {len(self.vital_signs)} vital signs, "
                   f"{len(self.lab_tests)} lab tests, and {len(self.demographics)} demographics")
    
    async def extract_features(
        self,
        patient_id: str,
        patient_data: Dict[str, Any],
        sequence_length: int = 48
    ) -> torch.Tensor:
        """
        Extract and process features for a patient.
        
        Args:
            patient_id: Patient identifier
            patient_data: Raw patient data from EHR
            sequence_length: Number of time steps to include
            
        Returns:
            Feature tensor [sequence_length, feature_dim]
        """
        try:
            # Extract temporal features
            temporal_features = await self._extract_temporal_features(
                patient_id, patient_data, sequence_length
            )
            
            # Extract static features
            static_features = await self._extract_static_features(patient_data)
            
            # Combine features
            combined = await self._combine_features(
                temporal_features, static_features, sequence_length
            )
            
            # Handle missing values
            combined = self._handle_missing_values(combined)
            
            # Normalize
            normalized = self._normalize_features(combined)
            
            # Convert to tensor
            feature_tensor = torch.FloatTensor(normalized)
            
            return feature_tensor
            
        except Exception as e:
            logger.error(f"Error extracting features for patient {patient_id}: {str(e)}")
            # Return zeros as fallback
            return torch.zeros(sequence_length, self.get_feature_dim())
    
    async def _extract_temporal_features(
        self,
        patient_id: str,
        patient_data: Dict[str, Any],
        sequence_length: int
    ) -> np.ndarray:
        """
        Extract time-varying features (vitals, labs).
        
        Args:
            patient_id: Patient identifier
            patient_data: Raw patient data
            sequence_length: Desired sequence length
            
        Returns:
            Temporal features array [sequence_length, temporal_dim]
        """
        # Get time series data
        vitals_series = patient_data.get("vitals", [])
        labs_series = patient_data.get("labs", [])
        
        # Create time index (last sequence_length hours)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=sequence_length)
        time_index = pd.date_range(start=start_time, end=end_time, periods=sequence_length)
        
        # Initialize feature matrix
        n_temporal_features = len(self.vital_signs) + len(self.lab_tests)
        feature_matrix = np.zeros((sequence_length, n_temporal_features))
        
        # Process vital signs
        for i, vital_name in enumerate(self.vital_signs):
            vital_values = []
            vital_times = []
            
            for vital in vitals_series:
                if vital.get("type") == vital_name:
                    vital_values.append(vital.get("value", 0))
                    vital_times.append(datetime.fromisoformat(vital.get("timestamp")))
            
            if vital_values:
                # Create time series
                vital_series = pd.Series(vital_values, index=vital_times)
                # Resample to regular intervals
                vital_resampled = vital_series.reindex(time_index, method='ffill', limit=3)
                feature_matrix[:, i] = vital_resampled.values
        
        # Process lab results
        for i, lab_name in enumerate(self.lab_tests):
            lab_idx = len(self.vital_signs) + i
            lab_values = []
            lab_times = []
            
            for lab in labs_series:
                if lab.get("test") == lab_name:
                    lab_values.append(lab.get("value", 0))
                    lab_times.append(datetime.fromisoformat(lab.get("timestamp")))
            
            if lab_values:
                lab_series = pd.Series(lab_values, index=lab_times)
                lab_resampled = lab_series.reindex(time_index, method='ffill', limit=5)
                feature_matrix[:, lab_idx] = lab_resampled.values
        
        # Add temporal aggregations
        aggregated = await self._add_temporal_aggregations(feature_matrix)
        
        return aggregated
    
    async def _add_temporal_aggregations(
        self,
        feature_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Add aggregated features over different time windows.
        
        Args:
            feature_matrix: Base temporal features
            
        Returns:
            Enhanced feature matrix with aggregations
        """
        n_timesteps, n_features = feature_matrix.shape
        n_windows = len(self.window_sizes)
        
        # Initialize enhanced matrix
        enhanced = np.zeros((n_timesteps, n_features * (1 + 3 * n_windows)))
        
        for t in range(n_timesteps):
            # Base features at current time
            enhanced[t, :n_features] = feature_matrix[t, :]
            
            offset = n_features
            
            # For each window
            for w_idx, window in enumerate(self.window_sizes):
                start_idx = max(0, t - window)
                window_data = feature_matrix[start_idx:t+1, :]
                
                if len(window_data) > 0:
                    # Mean
                    enhanced[t, offset + w_idx * 3 * n_features: offset + (w_idx * 3 + 1) * n_features] = \
                        np.nanmean(window_data, axis=0)
                    
                    # Standard deviation
                    enhanced[t, offset + (w_idx * 3 + 1) * n_features: offset + (w_idx * 3 + 2) * n_features] = \
                        np.nanstd(window_data, axis=0)
                    
                    # Slope (linear trend)
                    if len(window_data) > 1:
                        x = np.arange(len(window_data))
                        slopes = np.polyfit(x, window_data.T, 1)[0]
                        enhanced[t, offset + (w_idx * 3 + 2) * n_features: offset + (w_idx * 3 + 3) * n_features] = slopes
        
        return enhanced
    
    async def _extract_static_features(
        self,
        patient_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extract static patient features (demographics, comorbidities).
        
        Args:
            patient_data: Raw patient data
            
        Returns:
            Static features array [static_dim]
        """
        features = []
        
        # Demographics
        demo = patient_data.get("demographics", {})
        
        # Age
        if "date_of_birth" in demo:
            dob = datetime.strptime(demo["date_of_birth"], "%Y-%m-%d")
            age = (datetime.now() - dob).days / 365.25
            features.append(age)
        else:
            features.append(0)
        
        # Gender (one-hot encode)
        gender = demo.get("gender", "Unknown")
        if gender not in self.label_encoders:
            self.label_encoders["gender"] = LabelEncoder()
            self.label_encoders["gender"].fit(["M", "F", "Other", "Unknown"])
        
        gender_encoded = self.label_encoders["gender"].transform([gender])[0]
        features.append(gender_encoded)
        
        # BMI
        height = demo.get("height", 170) / 100  # Convert cm to m
        weight = demo.get("weight", 70)
        bmi = weight / (height * height)
        features.append(bmi)
        
        # Race (one-hot simplified as index)
        race = demo.get("race", "Unknown")
        if race not in self.label_encoders:
            self.label_encoders["race"] = LabelEncoder()
            races = ["White", "Black", "Asian", "Hispanic", "Other", "Unknown"]
            self.label_encoders["race"].fit(races)
        
        race_encoded = self.label_encoders["race"].transform([race])[0]
        features.append(race_encoded)
        
        # Comorbidities
        conditions = patient_data.get("conditions", [])
        condition_set = set(conditions)
        
        for comorbidity in self.comorbidities:
            features.append(1.0 if comorbidity in condition_set else 0.0)
        
        # Previous admissions count
        previous_admissions = patient_data.get("previous_admissions", 0)
        features.append(min(previous_admissions / 10, 1.0))  # Normalize
        
        # Length of stay in previous admission
        previous_los = patient_data.get("previous_los", 0)
        features.append(min(previous_los / 30, 1.0))  # Normalize to max 30 days
        
        return np.array(features)
    
    async def _combine_features(
        self,
        temporal: np.ndarray,
        static: np.ndarray,
        sequence_length: int
    ) -> np.ndarray:
        """
        Combine temporal and static features.
        
        Args:
            temporal: Temporal features [seq_len, temporal_dim]
            static: Static features [static_dim]
            sequence_length: Expected sequence length
            
        Returns:
            Combined features [seq_len, total_dim]
        """
        # Repeat static features for each time step
        static_repeated = np.tile(static, (sequence_length, 1))
        
        # Concatenate
        combined = np.concatenate([temporal, static_repeated], axis=1)
        
        return combined
    
    def _handle_missing_values(self, features: np.ndarray) -> np.ndarray:
        """
        Handle missing values in feature matrix.
        
        Args:
            features: Feature array with possible NaN values
            
        Returns:
            Features with imputed missing values
        """
        # Check if we need to fit imputer
        if not hasattr(self.imputer, 'statistics_') and np.any(np.isnan(features)):
            # Fit on first batch
            self.imputer.fit(features)
        
        # Transform
        if np.any(np.isnan(features)):
            features = self.imputer.transform(features)
        else:
            # No missing values
            pass
        
        return features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to zero mean and unit variance.
        
        Args:
            features: Feature array
            
        Returns:
            Normalized features
        """
        if not hasattr(self.scaler, 'mean_'):
            # Fit scaler
            self.scaler.fit(features)
        
        # Transform
        normalized = self.scaler.transform(features)
        
        return normalized
    
    def get_feature_dim(self) -> int:
        """
        Get total feature dimension.
        
        Returns:
            Number of features
        """
        if self._feature_dim is None:
            # Temporal features with aggregations
            n_base_temporal = len(self.vital_signs) + len(self.lab_tests)
            n_aggregations = 3 * len(self.window_sizes)  # mean, std, slope per window
            temporal_dim = n_base_temporal * (1 + n_aggregations)
            
            # Static features
            n_demographics = len(self.demographics)
            n_comorbidities = len(self.comorbidities)
            n_previous = 2  # previous admissions and LOS
            static_dim = n_demographics + n_comorbidities + n_previous
            
            self._feature_dim = temporal_dim + static_dim
        
        return self._feature_dim
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all features.
        
        Returns:
            List of feature names
        """
        feature_names = []
        
        # Base temporal features
        for vital in self.vital_signs:
            feature_names.append(f"vital_{vital}")
        
        for lab in self.lab_tests:
            feature_names.append(f"lab_{lab}")
        
        # Temporal aggregations
        base_count = len(self.vital_signs) + len(self.lab_tests)
        for window in self.window_sizes:
            for i in range(base_count):
                feature_names.append(f"mean_{window}h_{i}")
            for i in range(base_count):
                feature_names.append(f"std_{window}h_{i}")
            for i in range(base_count):
                feature_names.append(f"slope_{window}h_{i}")
        
        # Static features
        feature_names.extend(self.demographics)
        feature_names.extend(self.comorbidities)
        feature_names.append("previous_admissions")
        feature_names.append("previous_los")
        
        return feature_names
    
    def save(self, path: Path):
        """
        Save feature extractor state.
        
        Args:
            path: Path to save directory
        """
        path.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        with open(path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        
        # Save imputer
        with open(path / "imputer.pkl", "wb") as f:
            pickle.dump(self.imputer, f)
        
        # Save label encoders
        with open(path / "label_encoders.pkl", "wb") as f:
            pickle.dump(self.label_encoders, f)
        
        # Save config
        with open(path / "config.json", "w") as f:
            import json
            json.dump(self.config, f)
        
        logger.info(f"Feature extractor saved to {path}")
    
    def load(self, path: Path):
        """
        Load feature extractor state.
        
        Args:
            path: Path to load directory
        """
        # Load scaler
        with open(path / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        
        # Load imputer
        with open(path / "imputer.pkl", "rb") as f:
            self.imputer = pickle.load(f)
        
        # Load label encoders
        with open(path / "label_encoders.pkl", "rb") as f:
            self.label_encoders = pickle.load(f)
        
        logger.info(f"Feature extractor loaded from {path}")


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance for model interpretability.
    
    Provides:
    - Feature importance scores
    - SHAP value computation
    - Partial dependence plots
    - Feature correlation analysis
    """
    
    def __init__(self, feature_names: List[str]):
        """
        Initialize analyzer.
        
        Args:
            feature_names: Names of all features
        """
        self.feature_names = feature_names
        self.importance_scores = None
    
    def compute_permutation_importance(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        metric: str = "mse",
        n_repeats: int = 5
    ) -> np.ndarray:
        """
        Compute permutation feature importance.
        
        Args:
            model: PyTorch model
            X: Input features
            y: Target values
            metric: Evaluation metric
            n_repeats: Number of permutations
            
        Returns:
            Importance scores for each feature
        """
        model.eval()
        
        # Compute baseline score
        with torch.no_grad():
            baseline_pred = model(X)
            baseline_score = self._compute_metric(baseline_pred, y, metric)
        
        n_features = X.shape[-1]
        importance = np.zeros(n_features)
        
        # Permute each feature
        for i in range(n_features):
            scores = []
            for _ in range(n_repeats):
                # Permute feature i
                X_permuted = X.clone()
                perm_idx = torch.randperm(X.shape[0])
                X_permuted[:, :, i] = X_permuted[perm_idx, :, i]
                
                # Compute score
                with torch.no_grad():
                    perm_pred = model(X_permuted)
                    perm_score = self._compute_metric(perm_pred, y, metric)
                
                scores.append(perm_score)
            
            # Importance = increase in error
            importance[i] = np.mean(scores) - baseline_score
        
        self.importance_scores = importance
        return importance
    
    def _compute_metric(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        metric: str
    ) -> float:
        """
        Compute evaluation metric.
        
        Args:
            pred: Predictions
            target: Targets
            metric: Metric name
            
        Returns:
            Metric value
        """
        if metric == "mse":
            return F.mse_loss(pred, target).item()
        elif metric == "mae":
            return F.l1_loss(pred, target).item()
        elif metric == "accuracy":
            pred_labels = pred.argmax(dim=1)
            return (pred_labels == target).float().mean().item()
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N most important features.
        
        Args:
            n: Number of features to return
            
        Returns:
            List of (feature_name, importance) tuples
        """
        if self.importance_scores is None:
            return []
        
        indices = np.argsort(self.importance_scores)[-n:][::-1]
        return [(self.feature_names[i], self.importance_scores[i]) for i in indices]