"""
Main Risk Predictor Agent for AegisMedBot.

This module implements the primary risk prediction agent that:
- Orchestrates multiple prediction models
- Provides unified risk assessment interface
- Handles feature extraction and preprocessing
- Manages model versioning and selection
- Implements confidence scoring and calibration
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import asyncio
from collections import defaultdict
import pickle

from agents.base_agent import BaseAgent, AgentMessage, AgentResponse
from agents.risk_agent.models.lstm_predictor import LSTMPredictor
from agents.risk_agent.models.transformer_predictor import TransformerPredictor
from agents.risk_agent.features import ClinicalFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskPredictor(BaseAgent):
    """
    Primary risk prediction agent that coordinates multiple prediction models.
    
    This agent provides comprehensive risk assessment for patients including:
    - Mortality risk (in-hospital, 30-day, 1-year)
    - Complication risks (sepsis, cardiac events, readmission)
    - ICU admission probability
    - Length of stay predictions
    - Real-time deterioration alerts
    
    The agent uses ensemble methods combining LSTM and Transformer models
    for robust predictions with confidence calibration.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_path: Optional[Path] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the risk predictor agent.
        
        Args:
            config: Configuration dictionary with model parameters
            model_path: Path to pre-trained model checkpoints
            device: Computation device ('cuda' or 'cpu')
        """
        # Initialize base agent
        super().__init__(
            name="risk_agent",
            role="Patient Risk Assessment Specialist",
            description="Predicts patient risks including mortality, complications, and ICU admission",
            config=config
        )
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Risk predictor initialized on device: {self.device}")
        
        # Configuration defaults
        self.config = config or {}
        self.sequence_length = self.config.get("sequence_length", 48)  # Hours of history
        self.prediction_horizons = self.config.get("prediction_horizons", [24, 48, 168])  # Hours
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        
        # Initialize feature extractor
        self.feature_extractor = ClinicalFeatureExtractor(
            config=self.config.get("feature_config", {})
        )
        
        # Initialize prediction models
        self.models = self._initialize_models(model_path)
        
        # Model performance tracking
        self.model_performance = defaultdict(list)
        self.calibration_stats = {}
        
        # Cache for recent predictions
        self.prediction_cache = {}
        self.cache_ttl = timedelta(minutes=30)
        
        logger.info(f"Risk predictor initialized with {len(self.models)} models")
    
    def _initialize_models(self, model_path: Optional[Path]) -> Dict[str, nn.Module]:
        """
        Initialize all prediction models.
        
        Args:
            model_path: Path to pre-trained model checkpoints
            
        Returns:
            Dictionary of model name to model instance
        """
        models = {}
        
        # Initialize LSTM model for time series prediction
        lstm_config = {
            "input_size": self.feature_extractor.get_feature_dim(),
            "hidden_size": self.config.get("lstm_hidden_size", 256),
            "num_layers": self.config.get("lstm_layers", 2),
            "output_size": len(self.prediction_horizons) * 5,  # 5 risk types per horizon
            "dropout": self.config.get("lstm_dropout", 0.3)
        }
        
        models["lstm"] = LSTMPredictor(lstm_config)
        logger.info(f"LSTM model initialized with config: {lstm_config}")
        
        # Initialize Transformer model for complex pattern recognition
        transformer_config = {
            "input_dim": self.feature_extractor.get_feature_dim(),
            "d_model": self.config.get("transformer_d_model", 512),
            "nhead": self.config.get("transformer_heads", 8),
            "num_layers": self.config.get("transformer_layers", 6),
            "dim_feedforward": self.config.get("transformer_ff_dim", 2048),
            "output_dim": len(self.prediction_horizons) * 5,
            "dropout": self.config.get("transformer_dropout", 0.1)
        }
        
        models["transformer"] = TransformerPredictor(transformer_config)
        logger.info(f"Transformer model initialized with config: {transformer_config}")
        
        # Load pre-trained weights if available
        if model_path and model_path.exists():
            self._load_models(models, model_path)
        
        # Move models to device
        for name, model in models.items():
            models[name] = model.to(self.device)
            models[name].eval()
        
        return models
    
    def _load_models(self, models: Dict[str, nn.Module], model_path: Path):
        """
        Load pre-trained model weights.
        
        Args:
            models: Dictionary of models to load weights into
            model_path: Path to checkpoint directory
        """
        try:
            for name, model in models.items():
                checkpoint_path = model_path / f"{name}_best.pt"
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    logger.info(f"Loaded {name} model from {checkpoint_path}")
                    
                    # Load performance metrics
                    if "performance" in checkpoint:
                        self.model_performance[name] = checkpoint["performance"]
                else:
                    logger.warning(f"No checkpoint found for {name} at {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    async def process(self, message: AgentMessage) -> AgentResponse:
        """
        Process incoming risk prediction requests.
        
        This method handles various risk-related queries:
        - Patient mortality risk assessment
        - Complication prediction
        - ICU admission probability
        - Length of stay estimation
        - Real-time deterioration alerts
        
        Args:
            message: Standardized agent message with query content
            
        Returns:
            AgentResponse with risk predictions and confidence scores
        """
        start_time = datetime.now()
        
        try:
            # Validate input
            if not self.validate_input(message):
                return AgentResponse(
                    message_id=message.message_id,
                    content={"error": "Invalid message format"},
                    confidence=0.0,
                    processing_time_ms=0
                )
            
            self.update_status("processing", message.message_id)
            
            # Extract query parameters
            query = message.content.get("query", "")
            patient_id = message.content.get("patient_id")
            prediction_type = message.content.get("prediction_type", "comprehensive")
            
            # Validate patient ID
            if not patient_id:
                return AgentResponse(
                    message_id=message.message_id,
                    content={
                        "error": "Patient ID required for risk prediction",
                        "suggestion": "Please provide a valid patient identifier"
                    },
                    confidence=0.0,
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
            
            # Check cache for recent predictions
            cache_key = f"{patient_id}_{prediction_type}"
            if cache_key in self.prediction_cache:
                cached = self.prediction_cache[cache_key]
                cache_age = datetime.now() - cached["timestamp"]
                if cache_age < self.cache_ttl:
                    logger.info(f"Returning cached prediction for patient {patient_id}")
                    cached_response = cached["response"]
                    cached_response.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                    return cached_response
            
            # Extract clinical features
            patient_data = message.content.get("patient_data", {})
            feature_matrix = await self.feature_extractor.extract_features(
                patient_id=patient_id,
                patient_data=patient_data,
                sequence_length=self.sequence_length
            )
            
            # Generate predictions based on type
            if prediction_type == "mortality":
                predictions = await self._predict_mortality(feature_matrix)
            elif prediction_type == "complications":
                predictions = await self._predict_complications(feature_matrix)
            elif prediction_type == "icu_admission":
                predictions = await self._predict_icu_admission(feature_matrix)
            elif prediction_type == "length_of_stay":
                predictions = await self._predict_length_of_stay(feature_matrix)
            elif prediction_type == "deterioration":
                predictions = await self._predict_deterioration(feature_matrix)
            else:  # comprehensive
                predictions = await self._predict_comprehensive(feature_matrix)
            
            # Calculate ensemble confidence
            confidence = self._calculate_ensemble_confidence(predictions)
            
            # Calibrate probabilities
            calibrated_predictions = self._calibrate_predictions(predictions)
            
            # Generate alerts if risk exceeds thresholds
            alerts = self._generate_risk_alerts(calibrated_predictions)
            
            # Prepare response content
            response_content = {
                "patient_id": patient_id,
                "prediction_type": prediction_type,
                "timestamp": datetime.now().isoformat(),
                "predictions": calibrated_predictions,
                "alerts": alerts,
                "recommendations": self._generate_recommendations(calibrated_predictions),
                "disclaimer": "These predictions are for clinical decision support only. Always verify with clinical assessment."
            }
            
            # Create response
            response = AgentResponse(
                message_id=message.message_id,
                content=response_content,
                confidence=confidence,
                requires_human_confirmation=confidence < self.confidence_threshold or len(alerts) > 0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            
            # Cache the response
            self.prediction_cache[cache_key] = {
                "response": response,
                "timestamp": datetime.now()
            }
            
            # Log interaction for audit
            await self.log_interaction(message, response)
            
            self.update_status("completed")
            return response
            
        except Exception as e:
            logger.error(f"Error processing risk prediction: {str(e)}", exc_info=True)
            self.update_status("error")
            
            return AgentResponse(
                message_id=message.message_id,
                content={
                    "error": f"Risk prediction failed: {str(e)}",
                    "suggestion": "Please try again or contact support"
                },
                confidence=0.0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    async def _predict_mortality(self, feature_matrix: torch.Tensor) -> Dict[str, Any]:
        """
        Predict mortality risk at different time horizons.
        
        Args:
            feature_matrix: Tensor of clinical features [sequence_length, feature_dim]
            
        Returns:
            Dictionary with mortality predictions for different horizons
        """
        # Add batch dimension
        if feature_matrix.dim() == 2:
            feature_matrix = feature_matrix.unsqueeze(0)  # [1, seq_len, features]
        
        # Move to device
        feature_matrix = feature_matrix.to(self.device)
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            with torch.no_grad():
                output = model(feature_matrix)  # [1, num_predictions]
                
                # Extract mortality predictions (first 3 outputs per horizon)
                # Assuming output structure: [mortality_24h, mortality_48h, mortality_168h, ...]
                mortality_indices = [0, 1, 2]  # Indices for mortality predictions
                
                model_pred = {
                    "24h": float(torch.sigmoid(output[0, 0]).cpu().numpy()),
                    "48h": float(torch.sigmoid(output[0, 1]).cpu().numpy()),
                    "168h": float(torch.sigmoid(output[0, 2]).cpu().numpy())
                }
                predictions[name] = model_pred
        
        # Ensemble predictions (weighted average)
        ensemble = {}
        weights = self._get_model_weights("mortality")
        
        for horizon in ["24h", "48h", "168h"]:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for name, pred in predictions.items():
                weight = weights.get(name, 1.0)
                weighted_sum += pred[horizon] * weight
                total_weight += weight
            
            ensemble[horizon] = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        return {
            "mortality_risk": ensemble,
            "model_predictions": predictions,
            "risk_level": self._categorize_risk(ensemble),
            "trend": await self._analyze_trend(feature_matrix)
        }
    
    async def _predict_complications(self, feature_matrix: torch.Tensor) -> Dict[str, Any]:
        """
        Predict risk of various complications.
        
        Args:
            feature_matrix: Tensor of clinical features
            
        Returns:
            Dictionary with complication probabilities
        """
        if feature_matrix.dim() == 2:
            feature_matrix = feature_matrix.unsqueeze(0)
        
        feature_matrix = feature_matrix.to(self.device)
        
        # Complication types to predict
        complication_types = [
            "sepsis",
            "cardiac_arrest",
            "respiratory_failure", 
            "acute_kidney_injury",
            "bleeding"
        ]
        
        predictions = {}
        
        for name, model in self.models.items():
            with torch.no_grad():
                output = model(feature_matrix)  # [1, num_predictions]
                
                # Extract complication predictions (next 5 outputs)
                comp_pred = {}
                for i, comp_type in enumerate(complication_types):
                    # Index 3-7 for complications (after mortality)
                    comp_pred[comp_type] = float(torch.sigmoid(output[0, 3 + i]).cpu().numpy())
                
                predictions[name] = comp_pred
        
        # Ensemble predictions
        ensemble = {}
        weights = self._get_model_weights("complications")
        
        for comp_type in complication_types:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for name, pred in predictions.items():
                weight = weights.get(name, 1.0)
                weighted_sum += pred[comp_type] * weight
                total_weight += weight
            
            ensemble[comp_type] = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Identify high-risk complications
        high_risk = [
            comp for comp, prob in ensemble.items() 
            if prob > self.config.get("complication_threshold", 0.3)
        ]
        
        return {
            "complication_risks": ensemble,
            "high_risk_complications": high_risk,
            "model_predictions": predictions,
            "overall_risk_score": np.mean(list(ensemble.values()))
        }
    
    async def _predict_icu_admission(self, feature_matrix: torch.Tensor) -> Dict[str, Any]:
        """
        Predict probability of ICU admission.
        
        Args:
            feature_matrix: Tensor of clinical features
            
        Returns:
            Dictionary with ICU admission predictions
        """
        if feature_matrix.dim() == 2:
            feature_matrix = feature_matrix.unsqueeze(0)
        
        feature_matrix = feature_matrix.to(self.device)
        
        predictions = {}
        
        for name, model in self.models.items():
            with torch.no_grad():
                output = model(feature_matrix)
                
                # ICU admission probability (index 8)
                predictions[name] = float(torch.sigmoid(output[0, 8]).cpu().numpy())
        
        # Ensemble prediction
        weights = self._get_model_weights("icu_admission")
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, pred in predictions.items():
            weight = weights.get(name, 1.0)
            weighted_sum += pred * weight
            total_weight += weight
        
        ensemble_prob = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        return {
            "icu_admission_probability": ensemble_prob,
            "model_predictions": predictions,
            "recommendation": "Consider ICU transfer" if ensemble_prob > 0.5 else "Regular floor care",
            "priority": "HIGH" if ensemble_prob > 0.7 else "MEDIUM" if ensemble_prob > 0.4 else "LOW"
        }
    
    async def _predict_length_of_stay(self, feature_matrix: torch.Tensor) -> Dict[str, Any]:
        """
        Predict expected length of stay.
        
        Args:
            feature_matrix: Tensor of clinical features
            
        Returns:
            Dictionary with length of stay predictions
        """
        if feature_matrix.dim() == 2:
            feature_matrix = feature_matrix.unsqueeze(0)
        
        feature_matrix = feature_matrix.to(self.device)
        
        predictions = {}
        
        for name, model in self.models.items():
            with torch.no_grad():
                output = model(feature_matrix)
                
                # LOS prediction (index 9, regression output)
                predictions[name] = float(output[0, 9].cpu().numpy())
        
        # Ensemble prediction
        weights = self._get_model_weights("length_of_stay")
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, pred in predictions.items():
            weight = weights.get(name, 1.0)
            weighted_sum += pred * weight
            total_weight += weight
        
        ensemble_los = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Calculate confidence interval
        predictions_list = list(predictions.values())
        std_dev = np.std(predictions_list) if len(predictions_list) > 1 else 0.5
        
        return {
            "predicted_los_days": round(ensemble_los, 1),
            "confidence_interval": {
                "lower": round(ensemble_los - 1.96 * std_dev, 1),
                "upper": round(ensemble_los + 1.96 * std_dev, 1)
            },
            "model_predictions": predictions,
            "prediction_quality": "HIGH" if std_dev < 1.0 else "MEDIUM" if std_dev < 2.0 else "LOW"
        }
    
    async def _predict_deterioration(self, feature_matrix: torch.Tensor) -> Dict[str, Any]:
        """
        Predict patient deterioration risk in real-time.
        
        Args:
            feature_matrix: Tensor of clinical features
            
        Returns:
            Dictionary with deterioration predictions and alerts
        """
        if feature_matrix.dim() == 2:
            feature_matrix = feature_matrix.unsqueeze(0)
        
        feature_matrix = feature_matrix.to(self.device)
        
        # Calculate early warning scores
        news_score = await self._calculate_news_score(feature_matrix)
        qsofa_score = await self._calculate_qsofa_score(feature_matrix)
        
        # Get predictions
        mortality = await self._predict_mortality(feature_matrix)
        complications = await self._predict_complications(feature_matrix)
        
        # Combine into deterioration score
        deterioration_score = (
            0.3 * mortality["mortality_risk"]["24h"] +
            0.3 * complications["overall_risk_score"] +
            0.2 * (news_score / 20) +  # Normalize NEWS (0-20)
            0.2 * (qsofa_score / 3)      # Normalize qSOFA (0-3)
        )
        
        return {
            "deterioration_score": deterioration_score,
            "early_warning_scores": {
                "news": news_score,
                "qsofa": qsofa_score,
                "mews": await self._calculate_mews_score(feature_matrix)
            },
            "risk_trajectory": await self._analyze_trend(feature_matrix),
            "next_assessment_recommended": datetime.now() + timedelta(hours=4)
        }
    
    async def _predict_comprehensive(self, feature_matrix: torch.Tensor) -> Dict[str, Any]:
        """
        Generate comprehensive risk assessment combining all predictions.
        
        Args:
            feature_matrix: Tensor of clinical features
            
        Returns:
            Comprehensive risk assessment dictionary
        """
        # Run all predictions in parallel
        mortality_task = self._predict_mortality(feature_matrix)
        complications_task = self._predict_complications(feature_matrix)
        icu_task = self._predict_icu_admission(feature_matrix)
        los_task = self._predict_length_of_stay(feature_matrix)
        deterioration_task = self._predict_deterioration(feature_matrix)
        
        # Gather results
        mortality, complications, icu, los, deterioration = await asyncio.gather(
            mortality_task, complications_task, icu_task, los_task, deterioration_task
        )
        
        # Calculate overall risk score
        overall_risk = (
            0.25 * mortality["mortality_risk"]["48h"] +
            0.25 * complications["overall_risk_score"] +
            0.25 * icu["icu_admission_probability"] +
            0.25 * deterioration["deterioration_score"]
        )
        
        return {
            "overall_risk_score": overall_risk,
            "risk_category": self._categorize_risk({"overall": overall_risk}),
            "mortality": mortality,
            "complications": complications,
            "icu_admission": icu,
            "length_of_stay": los,
            "deterioration": deterioration,
            "key_findings": self._extract_key_findings(
                mortality, complications, icu, los, deterioration
            )
        }
    
    def _calculate_ensemble_confidence(self, predictions: Dict[str, Any]) -> float:
        """
        Calculate confidence in ensemble predictions based on model agreement.
        
        Args:
            predictions: Dictionary of predictions from different models
            
        Returns:
            Confidence score between 0 and 1
        """
        if "model_predictions" not in predictions:
            return 0.7  # Default confidence
        
        model_preds = predictions["model_predictions"]
        
        if not model_preds or len(model_preds) < 2:
            return 0.6
        
        # Calculate agreement between models
        agreement_scores = []
        
        if isinstance(next(iter(model_preds.values())), dict):
            # For dictionary predictions (mortality, complications)
            for horizon in ["24h", "48h", "168h"]:
                values = [pred[horizon] for pred in model_preds.values() if horizon in pred]
                if len(values) >= 2:
                    std_dev = np.std(values)
                    agreement_scores.append(1.0 - min(std_dev, 1.0))
        else:
            # For scalar predictions (ICU, LOS)
            values = list(model_preds.values())
            if len(values) >= 2:
                std_dev = np.std(values)
                agreement_scores.append(1.0 - min(std_dev / (max(values) + 1e-8), 1.0))
        
        if agreement_scores:
            return float(np.mean(agreement_scores))
        else:
            return 0.7
    
    def _calibrate_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calibrate prediction probabilities using Platt scaling.
        
        Args:
            predictions: Raw predictions dictionary
            
        Returns:
            Calibrated predictions
        """
        calibrated = predictions.copy()
        
        # Apply calibration if we have statistics
        if "mortality_risk" in predictions:
            for horizon in ["24h", "48h", "168h"]:
                if horizon in predictions["mortality_risk"]:
                    # Platt scaling: 1 / (1 + exp(-(score - shift) / scale))
                    raw_score = predictions["mortality_risk"][horizon]
                    shift = self.calibration_stats.get(horizon, {}).get("shift", 0.0)
                    scale = self.calibration_stats.get(horizon, {}).get("scale", 1.0)
                    
                    calibrated_score = 1.0 / (1.0 + np.exp(-(raw_score - shift) / scale))
                    calibrated["mortality_risk"][horizon] = float(calibrated_score)
        
        return calibrated
    
    def _generate_risk_alerts(self, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate alerts based on risk thresholds.
        
        Args:
            predictions: Calibrated predictions dictionary
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Mortality alerts
        if "mortality" in predictions:
            mortality = predictions["mortality"]["mortality_risk"]
            for horizon, risk in mortality.items():
                if risk > self.config.get("mortality_alert_threshold", 0.3):
                    alerts.append({
                        "type": "mortality_risk",
                        "severity": "HIGH" if risk > 0.5 else "MEDIUM",
                        "message": f"Elevated {horizon} mortality risk: {risk:.1%}",
                        "timestamp": datetime.now().isoformat(),
                        "recommendation": "Consider palliative care consultation" if risk > 0.5 else "Monitor closely"
                    })
        
        # Complication alerts
        if "complications" in predictions:
            complications = predictions["complications"]["complication_risks"]
            for comp_type, risk in complications.items():
                if risk > self.config.get("complication_alert_threshold", 0.3):
                    alerts.append({
                        "type": f"{comp_type}_risk",
                        "severity": "HIGH" if risk > 0.5 else "MEDIUM",
                        "message": f"Elevated {comp_type.replace('_', ' ')} risk: {risk:.1%}",
                        "timestamp": datetime.now().isoformat(),
                        "recommendation": f"Implement {comp_type} prevention protocols"
                    })
        
        # ICU admission alert
        if "icu_admission" in predictions:
            icu_risk = predictions["icu_admission"]["icu_admission_probability"]
            if icu_risk > self.config.get("icu_alert_threshold", 0.5):
                alerts.append({
                    "type": "icu_admission",
                    "severity": "HIGH" if icu_risk > 0.7 else "MEDIUM",
                    "message": f"High probability of ICU admission: {icu_risk:.1%}",
                    "timestamp": datetime.now().isoformat(),
                    "recommendation": "Prepare ICU bed and notify intensivist"
                })
        
        return alerts
    
    def _generate_recommendations(self, predictions: Dict[str, Any]) -> List[str]:
        """
        Generate clinical recommendations based on predictions.
        
        Args:
            predictions: Calibrated predictions dictionary
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if "mortality" in predictions:
            mortality = predictions["mortality"]["mortality_risk"]
            if mortality["24h"] > 0.3:
                recommendations.append("Consider ICU transfer for close monitoring")
            if mortality["168h"] > 0.4:
                recommendations.append("Discuss goals of care and advance directives")
        
        if "complications" in predictions:
            complications = predictions["complications"]["high_risk_complications"]
            if "sepsis" in complications:
                recommendations.append("Initiate sepsis screening protocol q4h")
            if "cardiac_arrest" in complications:
                recommendations.append("Continuous cardiac monitoring recommended")
            if "acute_kidney_injury" in complications:
                recommendations.append("Monitor renal function and urine output closely")
        
        if "icu_admission" in predictions:
            if predictions["icu_admission"]["icu_admission_probability"] > 0.6:
                recommendations.append("Early ICU consultation recommended")
        
        if "deterioration" in predictions:
            if predictions["deterioration"]["deterioration_score"] > 0.5:
                recommendations.append("Increase monitoring frequency to q2h")
        
        return recommendations
    
    def _get_model_weights(self, task: str) -> Dict[str, float]:
        """
        Get model weights for ensemble based on task type.
        
        Args:
            task: Prediction task type
            
        Returns:
            Dictionary of model weights
        """
        # Default equal weights
        default_weights = {name: 1.0 for name in self.models.keys()}
        
        # Task-specific weights based on historical performance
        task_weights = {
            "mortality": {
                "lstm": 0.6,
                "transformer": 0.4
            },
            "complications": {
                "lstm": 0.4,
                "transformer": 0.6
            },
            "icu_admission": {
                "lstm": 0.5,
                "transformer": 0.5
            },
            "length_of_stay": {
                "lstm": 0.7,
                "transformer": 0.3
            }
        }
        
        return task_weights.get(task, default_weights)
    
    def _categorize_risk(self, risk_scores: Dict[str, float]) -> str:
        """
        Categorize risk level based on scores.
        
        Args:
            risk_scores: Dictionary of risk scores
            
        Returns:
            Risk category string
        """
        # Get maximum risk score
        max_risk = max(risk_scores.values()) if risk_scores else 0
        
        if max_risk < 0.2:
            return "LOW"
        elif max_risk < 0.4:
            return "LOW-MEDIUM"
        elif max_risk < 0.6:
            return "MEDIUM"
        elif max_risk < 0.8:
            return "MEDIUM-HIGH"
        else:
            return "HIGH"
    
    def _extract_key_findings(self, *args) -> List[str]:
        """
        Extract key clinical findings from predictions.
        
        Args:
            *args: Various prediction dictionaries
            
        Returns:
            List of key finding strings
        """
        findings = []
        
        for predictions in args:
            if isinstance(predictions, dict):
                if "overall_risk_score" in predictions:
                    score = predictions["overall_risk_score"]
                    if score > 0.7:
                        findings.append("Critical patient - immediate attention required")
                    elif score > 0.4:
                        findings.append("Moderate risk - increased monitoring recommended")
                
                if "high_risk_complications" in predictions:
                    if predictions["high_risk_complications"]:
                        findings.append(f"High risk for: {', '.join(predictions['high_risk_complications'])}")
        
        return findings
    
    async def _calculate_news_score(self, feature_matrix: torch.Tensor) -> float:
        """
        Calculate National Early Warning Score (NEWS).
        
        Args:
            feature_matrix: Clinical features tensor
            
        Returns:
            NEWS score
        """
        # Simplified NEWS calculation
        # In production, extract actual vitals from features
        return 3.0  # Placeholder
    
    async def _calculate_qsofa_score(self, feature_matrix: torch.Tensor) -> float:
        """
        Calculate quick SOFA score.
        
        Args:
            feature_matrix: Clinical features tensor
            
        Returns:
            qSOFA score
        """
        # Simplified qSOFA calculation
        return 1.0  # Placeholder
    
    async def _calculate_mews_score(self, feature_matrix: torch.Tensor) -> float:
        """
        Calculate Modified Early Warning Score.
        
        Args:
            feature_matrix: Clinical features tensor
            
        Returns:
            MEWS score
        """
        # Simplified MEWS calculation
        return 2.0  # Placeholder
    
    async def _analyze_trend(self, feature_matrix: torch.Tensor) -> str:
        """
        Analyze trend in clinical parameters.
        
        Args:
            feature_matrix: Clinical features tensor over time
            
        Returns:
            Trend description
        """
        # Extract last few timepoints
        if feature_matrix.shape[1] < 5:
            return "Insufficient data for trend analysis"
        
        # Calculate slope of key parameters
        # Simplified trend analysis
        return "STABLE"  # Placeholder
    
    async def can_handle(self, task_type: str, context: Dict[str, Any]) -> float:
        """
        Determine if this agent can handle the task.
        
        Args:
            task_type: Type of task requested
            context: Conversation context
            
        Returns:
            Confidence score between 0 and 1
        """
        risk_keywords = [
            "risk", "mortality", "survival", "complication", "predict",
            "prognosis", "outcome", "likelihood", "probability", "chance",
            "icu", "admission", "deterioration", "worsening", "decline"
        ]
        
        query = context.get("query", "").lower()
        
        # Check for risk-related keywords
        keyword_matches = sum(1 for keyword in risk_keywords if keyword in query)
        
        if keyword_matches >= 2:
            return 0.9
        elif keyword_matches == 1:
            return 0.6
        else:
            return 0.2
    
    def get_model_performance(self) -> Dict[str, Any]:
        """
        Get performance metrics for all models.
        
        Returns:
            Dictionary with model performance metrics
        """
        return {
            "models": dict(self.model_performance),
            "calibration_stats": self.calibration_stats,
            "cache_size": len(self.prediction_cache)
        }
    
    def save_checkpoint(self, path: Path):
        """
        Save model checkpoints and configuration.
        
        Args:
            path: Directory to save checkpoints
        """
        path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": model.get_config() if hasattr(model, "get_config") else {},
                "performance": self.model_performance.get(name, [])
            }
            
            torch.save(checkpoint, path / f"{name}_best.pt")
            logger.info(f"Saved {name} checkpoint to {path}")
        
        # Save calibration stats
        with open(path / "calibration_stats.json", "w") as f:
            json.dump(self.calibration_stats, f)
    
    def load_checkpoint(self, path: Path):
        """
        Load model checkpoints.
        
        Args:
            path: Directory containing checkpoints
        """
        self._load_models(self.models, path)
        
        # Load calibration stats
        calib_path = path / "calibration_stats.json"
        if calib_path.exists():
            with open(calib_path, "r") as f:
                self.calibration_stats = json.load(f)