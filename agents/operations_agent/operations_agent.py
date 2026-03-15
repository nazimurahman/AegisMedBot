"""
Operations Agent - Core agent for hospital operational intelligence.
This agent is responsible for monitoring and optimizing hospital resources,
predicting patient flow, and providing real-time operational insights.
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field

# Import base agent classes
from agents.base_agent import BaseAgent, AgentMessage, AgentResponse, AgentStatus

# Import specialized modules
from .bed_analyzer import BedAnalyzer, BedStatus, BedType, BedOccupancyAnalysis
from .flow_predictor import FlowPredictor, FlowPrediction, PatientFlowMetrics

# Configure logging
logger = logging.getLogger(__name__)

class OperationsPriority(Enum):
    """
    Enumeration for operational request priorities.
    Used to determine urgency of operational requests.
    """
    CRITICAL = 1  # Life-critical operational needs
    HIGH = 2      # Urgent operational requirements
    MEDIUM = 3    # Standard operational requests
    LOW = 4       # Routine inquiries
    PLANNING = 5  # Long-term planning requests

class ResourceType(Enum):
    """
    Enumeration of hospital resource types for tracking.
    """
    BED = "bed"
    VENTILATOR = "ventilator"
    ICU_EQUIPMENT = "icu_equipment"
    STAFF = "staff"
    OPERATING_ROOM = "operating_room"
    MEDICAL_SUPPLY = "medical_supply"
    AMBULANCE = "ambulance"

class OperationsMetrics(BaseModel):
    """
    Comprehensive operational metrics for hospital performance monitoring.
    This model captures all key performance indicators for hospital operations.
    """
    # Bed occupancy metrics
    total_beds: int = Field(..., description="Total number of beds in hospital")
    occupied_beds: int = Field(..., description="Currently occupied beds")
    available_beds: int = Field(..., description="Currently available beds")
    occupancy_rate: float = Field(..., description="Current occupancy rate as percentage")
    
    # ICU specific metrics
    icu_beds_total: int = Field(..., description="Total ICU beds")
    icu_beds_occupied: int = Field(..., description="Occupied ICU beds")
    icu_occupancy_rate: float = Field(..., description="ICU occupancy rate")
    
    # Patient flow metrics
    patients_in_er: int = Field(0, description="Patients currently in ER")
    patients_waiting_admission: int = Field(0, description="Patients waiting for admission")
    average_wait_time_minutes: float = Field(0.0, description="Average ER wait time")
    
    # Discharge metrics
    discharges_today: int = Field(0, description="Patients discharged today")
    expected_discharges_today: int = Field(0, description="Expected discharges today")
    
    # Resource utilization
    ventilator_utilization: float = Field(0.0, description="Ventilator utilization rate")
    or_utilization: float = Field(0.0, description="Operating room utilization rate")
    
    # Staffing metrics
    nurses_on_duty: int = Field(0, description="Nurses currently on duty")
    doctors_on_duty: int = Field(0, description="Doctors currently on duty")
    staff_to_patient_ratio: float = Field(0.0, description="Staff to patient ratio")
    
    class Config:
        """Pydantic configuration for JSON serialization."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class OperationsAgent(BaseAgent):
    """
    Operations Agent responsible for hospital operational intelligence.
    This agent monitors bed occupancy, predicts patient flow, optimizes
    resource allocation, and provides real-time operational insights.
    
    Key Responsibilities:
    1. Real-time bed occupancy monitoring and prediction
    2. Patient flow analysis and forecasting
    3. Resource allocation optimization
    4. Operational bottleneck detection
    5. Staff scheduling recommendations
    6. Emergency department wait time prediction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Operations Agent with all necessary components.
        
        Args:
            config: Configuration dictionary containing operational parameters
        """
        # Call parent class initializer with agent metadata
        super().__init__(
            name="operations_agent",
            role="Hospital Operations Intelligence Specialist",
            description="Monitors and optimizes hospital resources, predicts patient flow, and provides operational insights",
            config=config
        )
        
        logger.info("Initializing Operations Agent")
        
        # Initialize specialized analyzers and predictors
        self.bed_analyzer = BedAnalyzer(config)
        self.flow_predictor = FlowPredictor(config)
        
        # Operational data cache for quick access
        self._current_metrics: Optional[OperationsMetrics] = None
        self._historical_metrics: List[OperationsMetrics] = []
        self._resource_alerts: List[Dict[str, Any]] = []
        
        # Configuration parameters with defaults
        self.bed_refresh_interval = self.config.get('bed_refresh_interval', 300)  # 5 minutes
        self.prediction_horizon_hours = self.config.get('prediction_horizon_hours', 24)
        self.critical_occupancy_threshold = self.config.get('critical_occupancy_threshold', 0.9)  # 90%
        self.warning_occupancy_threshold = self.config.get('warning_occupancy_threshold', 0.8)  # 80%
        
        logger.info(f"Operations Agent initialized with config: {self.config}")
    
    async def process(self, message: AgentMessage) -> AgentResponse:
        """
        Process incoming operational queries and return appropriate responses.
        This is the main entry point for all operations-related requests.
        
        Args:
            message: Standardized agent message containing the query
            
        Returns:
            AgentResponse with operational insights or error information
        """
        start_time = datetime.now()
        
        # Validate incoming message format
        if not self.validate_input(message):
            logger.error(f"Invalid message format received: {message}")
            return AgentResponse(
                message_id=message.message_id,
                content={"error": "Invalid message format"},
                confidence=0.0,
                processing_time_ms=0.0
            )
        
        # Update agent status to processing
        self.update_status(AgentStatus.PROCESSING, message.message_id)
        
        try:
            # Extract query and parameters from message
            query = message.content.get("query", "")
            parameters = message.content.get("parameters", {})
            
            logger.info(f"Processing operational query: {query[:100]}...")
            
            # Determine the type of operational request
            request_type, confidence = await self._classify_request(query)
            
            # Route to appropriate handler based on request type
            if request_type == "bed_occupancy":
                response_content = await self._handle_bed_occupancy_query(query, parameters)
            elif request_type == "patient_flow":
                response_content = await self._handle_patient_flow_query(query, parameters)
            elif request_type == "resource_allocation":
                response_content = await self._handle_resource_allocation_query(query, parameters)
            elif request_type == "wait_time":
                response_content = await self._handle_wait_time_query(query, parameters)
            elif request_type == "staffing":
                response_content = await self._handle_staffing_query(query, parameters)
            elif request_type == "discharge_prediction":
                response_content = await self._handle_discharge_prediction_query(query, parameters)
            elif request_type == "operational_metrics":
                response_content = await self._handle_metrics_query(query, parameters)
            elif request_type == "bottleneck_detection":
                response_content = await self._handle_bottleneck_query(query, parameters)
            else:
                # General operational query
                response_content = await self._handle_general_operational_query(query, parameters)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Determine if response requires human confirmation
            requires_human = self._needs_human_confirmation(response_content)
            
            # Update status to completed
            self.update_status(AgentStatus.COMPLETED)
            
            # Log successful interaction
            logger.info(f"Successfully processed operational query in {processing_time:.2f}ms")
            
            return AgentResponse(
                message_id=message.message_id,
                content=response_content,
                confidence=confidence,
                requires_human_confirmation=requires_human,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing operational query: {str(e)}", exc_info=True)
            self.update_status(AgentStatus.ERROR)
            
            return AgentResponse(
                message_id=message.message_id,
                content={
                    "error": f"Operations processing error: {str(e)}",
                    "error_type": type(e).__name__
                },
                confidence=0.0,
                requires_human_confirmation=True,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    async def _classify_request(self, query: str) -> Tuple[str, float]:
        """
        Classify the type of operational request based on query content.
        Uses keyword matching and semantic analysis for classification.
        
        Args:
            query: User's query string
            
        Returns:
            Tuple of (request_type, confidence_score)
        """
        query_lower = query.lower()
        
        # Define keyword patterns for different request types
        classification_patterns = {
            "bed_occupancy": {
                "keywords": ["bed", "occupancy", "capacity", "available bed", "free bed", "bed count"],
                "weight": 0.8
            },
            "patient_flow": {
                "keywords": ["patient flow", "admission", "discharge", "throughput", "patient movement"],
                "weight": 0.85
            },
            "resource_allocation": {
                "keywords": ["resource", "equipment", "ventilator", "allocate", "distribution"],
                "weight": 0.75
            },
            "wait_time": {
                "keywords": ["wait time", "waiting", "delay", "er wait", "emergency wait"],
                "weight": 0.9
            },
            "staffing": {
                "keywords": ["staff", "nurse", "doctor", "schedule", "shift", "on duty"],
                "weight": 0.85
            },
            "discharge_prediction": {
                "keywords": ["discharge", "release", "going home", "leave hospital"],
                "weight": 0.8
            },
            "operational_metrics": {
                "keywords": ["metrics", "kpi", "performance", "statistics", "dashboard"],
                "weight": 0.7
            },
            "bottleneck_detection": {
                "keywords": ["bottleneck", "delay", "slow", "backlog", "congestion"],
                "weight": 0.85
            }
        }
        
        # Calculate scores for each request type
        scores = {}
        for req_type, pattern in classification_patterns.items():
            score = 0.0
            matches = 0
            
            for keyword in pattern["keywords"]:
                if keyword in query_lower:
                    matches += 1
            
            if matches > 0:
                # Calculate score based on matches and weight
                score = (matches / len(pattern["keywords"])) * pattern["weight"]
                scores[req_type] = score
        
        if not scores:
            return "general", 0.3  # Low confidence for general queries
        
        # Select the best matching request type
        best_match = max(scores.items(), key=lambda x: x[1])
        return best_match[0], best_match[1]
    
    async def _handle_bed_occupancy_query(
        self,
        query: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle queries related to bed occupancy and availability.
        
        Args:
            query: Original user query
            parameters: Additional parameters from the request
            
        Returns:
            Dictionary with bed occupancy information
        """
        logger.info("Handling bed occupancy query")
        
        # Extract bed type from query if specified
        bed_type = self._extract_bed_type(query)
        department = parameters.get("department") or self._extract_department(query)
        
        # Get current bed occupancy analysis
        occupancy_analysis = await self.bed_analyzer.analyze_current_occupancy(
            bed_type=bed_type,
            department=department
        )
        
        # Get occupancy predictions
        predictions = await self.bed_analyzer.predict_occupancy(
            hours_ahead=self.prediction_horizon_hours,
            bed_type=bed_type
        )
        
        # Check for critical occupancy levels
        alerts = await self._check_occupancy_alerts(occupancy_analysis)
        
        # Format response
        response = {
            "query_type": "bed_occupancy",
            "current_occupancy": {
                "total_beds": occupancy_analysis.total_beds,
                "occupied_beds": occupancy_analysis.occupied_beds,
                "available_beds": occupancy_analysis.available_beds,
                "occupancy_rate": occupancy_analysis.occupancy_rate,
                "by_department": occupancy_analysis.department_breakdown,
                "by_bed_type": occupancy_analysis.bed_type_breakdown
            },
            "predictions": {
                "next_6_hours": predictions.get("6h", {}),
                "next_12_hours": predictions.get("12h", {}),
                "next_24_hours": predictions.get("24h", {})
            },
            "alerts": alerts,
            "recommendations": self._generate_bed_recommendations(occupancy_analysis),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add bed type specific information if requested
        if bed_type:
            response["bed_type_specific"] = {
                "type": bed_type.value,
                "available": occupancy_analysis.bed_type_breakdown.get(bed_type.value, {}).get("available", 0)
            }
        
        return response
    
    def _extract_bed_type(self, query: str) -> Optional[BedType]:
        """
        Extract bed type from query using keyword matching.
        
        Args:
            query: User query string
            
        Returns:
            BedType enum value or None if not specified
        """
        query_lower = query.lower()
        
        bed_type_mapping = {
            "icu": BedType.ICU,
            "intensive care": BedType.ICU,
            "critical care": BedType.ICU,
            "emergency": BedType.EMERGENCY,
            "er": BedType.EMERGENCY,
            "med surg": BedType.MEDICAL_SURGICAL,
            "medical surgical": BedType.MEDICAL_SURGICAL,
            "pediatric": BedType.PEDIATRIC,
            "children": BedType.PEDIATRIC,
            "nicu": BedType.NICU,
            "neonatal": BedType.NICU
        }
        
        for keyword, bed_type in bed_type_mapping.items():
            if keyword in query_lower:
                return bed_type
        
        return None
    
    def _extract_department(self, query: str) -> Optional[str]:
        """
        Extract department name from query.
        
        Args:
            query: User query string
            
        Returns:
            Department name or None
        """
        departments = ["cardiology", "oncology", "neurology", "orthopedics", 
                      "pediatrics", "surgery", "internal medicine"]
        
        query_lower = query.lower()
        for dept in departments:
            if dept in query_lower:
                return dept
        
        return None
    
    async def _check_occupancy_alerts(
        self,
        analysis: BedOccupancyAnalysis
    ) -> List[Dict[str, Any]]:
        """
        Check for critical occupancy levels and generate alerts.
        
        Args:
            analysis: Bed occupancy analysis data
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Check overall occupancy
        if analysis.occupancy_rate >= self.critical_occupancy_threshold:
            alerts.append({
                "level": "CRITICAL",
                "type": "overall_occupancy",
                "message": f"Hospital at critical occupancy: {analysis.occupancy_rate:.1%}",
                "threshold": self.critical_occupancy_threshold,
                "current_value": analysis.occupancy_rate
            })
        elif analysis.occupancy_rate >= self.warning_occupancy_threshold:
            alerts.append({
                "level": "WARNING",
                "type": "overall_occupancy",
                "message": f"Hospital approaching critical occupancy: {analysis.occupancy_rate:.1%}",
                "threshold": self.warning_occupancy_threshold,
                "current_value": analysis.occupancy_rate
            })
        
        # Check ICU occupancy specifically
        icu_occupancy = analysis.bed_type_breakdown.get(BedType.ICU.value, {}).get("occupancy_rate", 0)
        if icu_occupancy >= self.critical_occupancy_threshold:
            alerts.append({
                "level": "CRITICAL",
                "type": "icu_occupancy",
                "message": f"ICU at critical occupancy: {icu_occupancy:.1%}",
                "threshold": self.critical_occupancy_threshold,
                "current_value": icu_occupancy
            })
        
        return alerts
    
    def _generate_bed_recommendations(
        self,
        analysis: BedOccupancyAnalysis
    ) -> List[str]:
        """
        Generate recommendations based on bed occupancy analysis.
        
        Args:
            analysis: Bed occupancy analysis data
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if analysis.occupancy_rate > 0.85:
            recommendations.append("Consider activating surge capacity protocol")
            recommendations.append("Review pending discharges for today")
            recommendations.append("Assess possibility of early discharges")
        
        if analysis.occupancy_rate < 0.4:
            recommendations.append("Consider consolidating units for efficiency")
            recommendations.append("Review staffing levels for potential reduction")
        
        # Check specific department needs
        for dept, dept_data in analysis.department_breakdown.items():
            if dept_data["occupancy_rate"] > 0.9:
                recommendations.append(f"Critical occupancy in {dept} department - consider patient redistribution")
        
        return recommendations
    
    async def _handle_patient_flow_query(
        self,
        query: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle queries related to patient flow analysis and prediction.
        
        Args:
            query: Original user query
            parameters: Additional parameters from the request
            
        Returns:
            Dictionary with patient flow information
        """
        logger.info("Handling patient flow query")
        
        # Get current flow metrics
        current_flow = await self.flow_predictor.get_current_flow_metrics()
        
        # Generate flow predictions
        flow_predictions = await self.flow_predictor.predict_patient_flow(
            hours_ahead=self.prediction_horizon_hours
        )
        
        # Identify flow bottlenecks
        bottlenecks = await self.flow_predictor.identify_bottlenecks()
        
        response = {
            "query_type": "patient_flow",
            "current_metrics": {
                "patients_in_er": current_flow.er_patients,
                "patients_waiting_admission": current_flow.waiting_admission,
                "average_wait_time_minutes": current_flow.average_wait_time,
                "admissions_today": current_flow.admissions_today,
                "discharges_today": current_flow.discharges_today,
                "boarded_patients": current_flow.boarded_patients
            },
            "predictions": {
                "next_6_hours": self._format_flow_predictions(flow_predictions, 6),
                "next_12_hours": self._format_flow_predictions(flow_predictions, 12),
                "next_24_hours": self._format_flow_predictions(flow_predictions, 24)
            },
            "bottlenecks": bottlenecks,
            "flow_efficiency_score": self._calculate_flow_efficiency(current_flow),
            "recommendations": self._generate_flow_recommendations(current_flow, bottlenecks),
            "timestamp": datetime.now().isoformat()
        }
        
        return response
    
    def _format_flow_predictions(
        self,
        predictions: Dict[str, Any],
        hours: int
    ) -> Dict[str, Any]:
        """
        Format flow predictions for a specific time horizon.
        
        Args:
            predictions: All flow predictions
            hours: Hours ahead for prediction
            
        Returns:
            Formatted predictions for specified horizon
        """
        return {
            "er_arrivals": predictions.get(f"{hours}h_er_arrivals", 0),
            "admissions": predictions.get(f"{hours}h_admissions", 0),
            "discharges": predictions.get(f"{hours}h_discharges", 0),
            "peak_occupancy": predictions.get(f"{hours}h_peak_occupancy", 0),
            "peak_time": predictions.get(f"{hours}h_peak_time", "unknown"),
            "confidence_interval": predictions.get(f"{hours}h_confidence", [0, 0])
        }
    
    def _calculate_flow_efficiency(self, metrics: PatientFlowMetrics) -> float:
        """
        Calculate patient flow efficiency score (0-100).
        
        Args:
            metrics: Current patient flow metrics
            
        Returns:
            Efficiency score as percentage
        """
        score = 100.0
        
        # Deduct for wait times
        if metrics.average_wait_time > 30:
            score -= min(30, (metrics.average_wait_time - 30) / 2)
        
        # Deduct for boarded patients
        if metrics.boarded_patients > 5:
            score -= min(20, metrics.boarded_patients * 2)
        
        # Deduct for high ER census
        if metrics.er_patients > 20:
            score -= min(15, (metrics.er_patients - 20) * 0.5)
        
        return max(0, score)
    
    def _generate_flow_recommendations(
        self,
        metrics: PatientFlowMetrics,
        bottlenecks: List[str]
    ) -> List[str]:
        """
        Generate recommendations for improving patient flow.
        
        Args:
            metrics: Current flow metrics
            bottlenecks: Identified bottlenecks
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if metrics.average_wait_time > 60:
            recommendations.append("Consider opening additional triage stations")
            recommendations.append("Implement fast-track for low-acuity patients")
        
        if metrics.boarded_patients > 10:
            recommendations.append("Expedite discharge process for stable patients")
            recommendations.append("Consider transferring boarded patients to step-down units")
        
        for bottleneck in bottlenecks:
            recommendations.append(f"Address bottleneck: {bottleneck}")
        
        return recommendations
    
    async def _handle_resource_allocation_query(
        self,
        query: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle queries about resource allocation and optimization.
        
        Args:
            query: Original user query
            parameters: Additional parameters from the request
            
        Returns:
            Dictionary with resource allocation information
        """
        logger.info("Handling resource allocation query")
        
        # Get current resource utilization
        resource_utilization = await self._get_resource_utilization()
        
        # Generate optimization recommendations
        optimization_recommendations = await self._optimize_resource_allocation(resource_utilization)
        
        # Check for resource shortages
        shortages = await self._identify_resource_shortages(resource_utilization)
        
        response = {
            "query_type": "resource_allocation",
            "current_utilization": resource_utilization,
            "optimization_recommendations": optimization_recommendations,
            "critical_shortages": shortages,
            "projected_needs": await self._project_resource_needs(),
            "timestamp": datetime.now().isoformat()
        }
        
        return response
    
    async def _get_resource_utilization(self) -> Dict[str, Any]:
        """
        Get current utilization rates for all resource types.
        
        Returns:
            Dictionary with resource utilization data
        """
        # This would connect to hospital systems in production
        # For now, return simulated data
        
        return {
            ResourceType.BED.value: {
                "utilization_rate": 0.78,
                "total": 500,
                "in_use": 390,
                "available": 110
            },
            ResourceType.VENTILATOR.value: {
                "utilization_rate": 0.65,
                "total": 50,
                "in_use": 32,
                "available": 18
            },
            ResourceType.STAFF.value: {
                "nurses": {
                    "scheduled": 120,
                    "on_duty": 98,
                    "utilization": 0.82
                },
                "doctors": {
                    "scheduled": 45,
                    "on_duty": 38,
                    "utilization": 0.84
                }
            },
            ResourceType.OPERATING_ROOM.value: {
                "utilization_rate": 0.72,
                "total_rooms": 15,
                "in_use": 11,
                "available": 4
            }
        }
    
    async def _optimize_resource_allocation(
        self,
        current_utilization: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate resource allocation optimization recommendations.
        
        Args:
            current_utilization: Current resource utilization data
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Optimize bed allocation
        if current_utilization.get(ResourceType.BED.value, {}).get("utilization_rate", 0) > 0.9:
            recommendations.append({
                "resource": "beds",
                "action": "redistribute",
                "description": "Redistribute patients from high-occupancy units",
                "potential_impact": "Create 15-20 additional bed capacity"
            })
        
        # Optimize staff allocation
        nurse_util = current_utilization.get(ResourceType.STAFF.value, {}).get("nurses", {}).get("utilization", 0)
        if nurse_util > 0.9:
            recommendations.append({
                "resource": "nurses",
                "action": "call_in",
                "description": "Call in 5 additional nurses from on-call list",
                "potential_impact": "Reduce nurse-patient ratio to 1:4"
            })
        
        return recommendations
    
    async def _identify_resource_shortages(
        self,
        current_utilization: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify critical resource shortages.
        
        Args:
            current_utilization: Current resource utilization data
            
        Returns:
            List of identified shortages
        """
        shortages = []
        
        # Check ventilator availability
        vent_data = current_utilization.get(ResourceType.VENTILATOR.value, {})
        if vent_data.get("available", 0) < 5:
            shortages.append({
                "resource": "ventilators",
                "severity": "critical",
                "current_available": vent_data.get("available", 0),
                "minimum_required": 5,
                "deficit": 5 - vent_data.get("available", 0)
            })
        
        # Check ICU bed availability
        # This would come from bed_analyzer in production
        
        return shortages
    
    async def _project_resource_needs(self) -> Dict[str, Any]:
        """
        Project resource needs for the next 24-48 hours.
        
        Returns:
            Dictionary with projected resource needs
        """
        # This would use predictive models in production
        # For now, return simulated projections
        
        return {
            "next_24h": {
                "beds_needed": 45,
                "nurses_needed": 12,
                "ventilators_needed": 3,
                "or_slots_needed": 8
            },
            "next_48h": {
                "beds_needed": 80,
                "nurses_needed": 20,
                "ventilators_needed": 5,
                "or_slots_needed": 15
            }
        }
    
    async def _handle_wait_time_query(
        self,
        query: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle queries about wait times in various departments.
        
        Args:
            query: Original user query
            parameters: Additional parameters from the request
            
        Returns:
            Dictionary with wait time information
        """
        logger.info("Handling wait time query")
        
        # Extract department from query
        department = self._extract_department(query) or parameters.get("department", "emergency")
        
        # Get current wait times
        wait_times = await self._get_current_wait_times(department)
        
        # Get wait time predictions
        predictions = await self._predict_wait_times(department)
        
        response = {
            "query_type": "wait_time",
            "department": department,
            "current_wait_times": wait_times,
            "predictions": predictions,
            "factors_affecting_wait": await self._analyze_wait_factors(department),
            "recommendations": self._generate_wait_time_recommendations(wait_times),
            "timestamp": datetime.now().isoformat()
        }
        
        return response
    
    async def _get_current_wait_times(self, department: str) -> Dict[str, Any]:
        """
        Get current wait times for specified department.
        
        Args:
            department: Department name
            
        Returns:
            Dictionary with wait time data
        """
        # This would connect to hospital systems
        # For now, return simulated data
        
        base_times = {
            "emergency": {"average": 45, "min": 15, "max": 120, "patients_waiting": 12},
            "cardiology": {"average": 30, "min": 10, "max": 60, "patients_waiting": 5},
            "orthopedics": {"average": 25, "min": 5, "max": 45, "patients_waiting": 3},
            "pediatrics": {"average": 20, "min": 5, "max": 40, "patients_waiting": 4}
        }
        
        return base_times.get(department, base_times["emergency"])
    
    async def _predict_wait_times(self, department: str) -> Dict[str, Any]:
        """
        Predict wait times for the next few hours.
        
        Args:
            department: Department name
            
        Returns:
            Dictionary with wait time predictions
        """
        # This would use time series models in production
        
        return {
            "next_hour": {"expected": 40, "range": [30, 50]},
            "next_2_hours": {"expected": 35, "range": [25, 45]},
            "next_4_hours": {"expected": 30, "range": [20, 40]}
        }
    
    async def _analyze_wait_factors(self, department: str) -> List[str]:
        """
        Analyze factors contributing to wait times.
        
        Args:
            department: Department name
            
        Returns:
            List of factor descriptions
        """
        # This would analyze hospital data
        
        return [
            "High patient volume during peak hours (2-6 PM)",
            "Limited number of available beds for admission",
            "Laboratory result delays averaging 45 minutes",
            "Staff shortage during shift changes"
        ]
    
    def _generate_wait_time_recommendations(self, wait_times: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations to reduce wait times.
        
        Args:
            wait_times: Current wait time data
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if wait_times.get("average", 0) > 60:
            recommendations.append("Implement provider-in-triage model to initiate care earlier")
            recommendations.append("Add additional triage nurse during peak hours")
        
        if wait_times.get("patients_waiting", 0) > 15:
            recommendations.append("Open additional treatment bays")
            recommendations.append("Fast-track low-acuity patients to urgent care")
        
        return recommendations
    
    async def _handle_staffing_query(
        self,
        query: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle queries about staff scheduling and optimization.
        
        Args:
            query: Original user query
            parameters: Additional parameters from the request
            
        Returns:
            Dictionary with staffing information
        """
        logger.info("Handling staffing query")
        
        # Get current staffing levels
        current_staffing = await self._get_current_staffing()
        
        # Get shift coverage
        shift_coverage = await self._get_shift_coverage()
        
        # Generate staffing recommendations
        staffing_recs = await self._optimize_staffing(current_staffing)
        
        response = {
            "query_type": "staffing",
            "current_staffing": current_staffing,
            "shift_coverage": shift_coverage,
            "staffing_gaps": await self._identify_staffing_gaps(current_staffing),
            "optimization_recommendations": staffing_recs,
            "projected_needs": await self._project_staffing_needs(),
            "timestamp": datetime.now().isoformat()
        }
        
        return response
    
    async def _get_current_staffing(self) -> Dict[str, Any]:
        """
        Get current staffing levels by role and department.
        
        Returns:
            Dictionary with staffing data
        """
        # This would connect to HR/scheduling system
        # For now, return simulated data
        
        return {
            "nurses": {
                "total_scheduled": 120,
                "currently_on_duty": 98,
                "by_department": {
                    "emergency": 25,
                    "icu": 18,
                    "med_surg": 35,
                    "surgery": 12,
                    "other": 8
                }
            },
            "doctors": {
                "total_scheduled": 45,
                "currently_on_duty": 38,
                "by_department": {
                    "emergency": 8,
                    "icu": 6,
                    "med_surg": 12,
                    "surgery": 7,
                    "other": 5
                }
            },
            "support_staff": {
                "total_scheduled": 80,
                "currently_on_duty": 65
            }
        }
    
    async def _get_shift_coverage(self) -> Dict[str, Any]:
        """
        Get coverage for current and upcoming shifts.
        
        Returns:
            Dictionary with shift coverage data
        """
        now = datetime.now()
        
        return {
            "current_shift": {
                "time": f"{now.hour}:00 - {(now.hour + 8) % 24}:00",
                "coverage": 0.85
            },
            "next_shift": {
                "time": f"{(now.hour + 8) % 24}:00 - {(now.hour + 16) % 24}:00",
                "coverage": 0.75,
                "gaps": ["2 ICU nurses", "1 ER doctor"]
            }
        }
    
    async def _identify_staffing_gaps(self, staffing: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify gaps in staffing coverage.
        
        Args:
            staffing: Current staffing data
            
        Returns:
            List of staffing gaps
        """
        gaps = []
        
        # Define minimum staffing requirements
        min_requirements = {
            "emergency": {"nurses": 8, "doctors": 3},
            "icu": {"nurses": 6, "doctors": 2},
            "med_surg": {"nurses": 12, "doctors": 4}
        }
        
        # Check each department
        for dept, requirements in min_requirements.items():
            current_nurses = staffing.get("nurses", {}).get("by_department", {}).get(dept, 0)
            current_doctors = staffing.get("doctors", {}).get("by_department", {}).get(dept, 0)
            
            if current_nurses < requirements["nurses"]:
                gaps.append({
                    "department": dept,
                    "role": "nurses",
                    "current": current_nurses,
                    "required": requirements["nurses"],
                    "deficit": requirements["nurses"] - current_nurses
                })
            
            if current_doctors < requirements["doctors"]:
                gaps.append({
                    "department": dept,
                    "role": "doctors",
                    "current": current_doctors,
                    "required": requirements["doctors"],
                    "deficit": requirements["doctors"] - current_doctors
                })
        
        return gaps
    
    async def _optimize_staffing(self, staffing: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate staffing optimization recommendations.
        
        Args:
            staffing: Current staffing data
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Analyze nurse-to-patient ratios
        nurse_ratios = self._calculate_nurse_patient_ratios(staffing)
        
        for dept, ratio in nurse_ratios.items():
            if ratio > 5:  # Too many patients per nurse
                recommendations.append({
                    "department": dept,
                    "issue": "High patient-to-nurse ratio",
                    "current_ratio": ratio,
                    "target_ratio": 4,
                    "recommendation": f"Add {ratio - 4} nurses in {dept}",
                    "priority": "high"
                })
        
        return recommendations
    
    def _calculate_nurse_patient_ratios(self, staffing: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate nurse-to-patient ratios by department.
        
        Args:
            staffing: Current staffing data
            
        Returns:
            Dictionary of ratios by department
        """
        # This would use patient census data
        # For now, return simulated ratios
        
        return {
            "emergency": 4.5,
            "icu": 2.0,
            "med_surg": 5.5,
            "surgery": 3.0
        }
    
    async def _project_staffing_needs(self) -> Dict[str, Any]:
        """
        Project staffing needs for upcoming shifts.
        
        Returns:
            Dictionary with projected staffing needs
        """
        # This would use predictive models
        # For now, return simulated projections
        
        return {
            "next_24h": {
                "nurses_needed": 15,
                "doctors_needed": 5,
                "by_department": {
                    "emergency": {"nurses": 4, "doctors": 1},
                    "icu": {"nurses": 3, "doctors": 1},
                    "med_surg": {"nurses": 5, "doctors": 2}
                }
            }
        }
    
    async def _handle_discharge_prediction_query(
        self,
        query: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle queries about discharge predictions and planning.
        
        Args:
            query: Original user query
            parameters: Additional parameters from the request
            
        Returns:
            Dictionary with discharge prediction information
        """
        logger.info("Handling discharge prediction query")
        
        # Get current discharge predictions
        predictions = await self.flow_predictor.predict_discharges(
            hours_ahead=48
        )
        
        # Get patients ready for discharge
        ready_for_discharge = await self._get_ready_for_discharge()
        
        response = {
            "query_type": "discharge_prediction",
            "predictions": {
                "next_6_hours": predictions.get("6h", 0),
                "next_12_hours": predictions.get("12h", 0),
                "next_24_hours": predictions.get("24h", 0),
                "next_48_hours": predictions.get("48h", 0)
            },
            "currently_ready_for_discharge": ready_for_discharge,
            "bottlenecks": await self._identify_discharge_bottlenecks(),
            "recommendations": await self._generate_discharge_recommendations(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
        return response
    
    async def _get_ready_for_discharge(self) -> List[Dict[str, Any]]:
        """
        Get list of patients ready for discharge.
        
        Returns:
            List of patients ready for discharge
        """
        # This would query patient management system
        
        return [
            {
                "patient_id": "P12345",
                "name": "John Doe",
                "department": "Med/Surg",
                "ready_since": "2024-01-15T10:30:00",
                "waiting_for": "Discharge medications"
            },
            {
                "patient_id": "P12346",
                "name": "Jane Smith",
                "department": "Cardiology",
                "ready_since": "2024-01-15T09:15:00",
                "waiting_for": "Transportation"
            }
        ]
    
    async def _identify_discharge_bottlenecks(self) -> List[str]:
        """
        Identify bottlenecks in the discharge process.
        
        Returns:
            List of bottleneck descriptions
        """
        return [
            "Pharmacy processing time averaging 90 minutes",
            "Transportation availability limited",
            "Discharge summary completion delayed"
        ]
    
    async def _generate_discharge_recommendations(
        self,
        predictions: Dict[str, int]
    ) -> List[str]:
        """
        Generate recommendations for discharge optimization.
        
        Args:
            predictions: Discharge predictions
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if predictions.get("24h", 0) > 20:
            recommendations.append("Schedule additional discharge planning staff for tomorrow")
            recommendations.append("Coordinate with pharmacy for early medication dispensing")
        
        return recommendations
    
    async def _handle_metrics_query(
        self,
        query: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle queries for comprehensive operational metrics.
        
        Args:
            query: Original user query
            parameters: Additional parameters from the request
            
        Returns:
            Dictionary with comprehensive metrics
        """
        logger.info("Handling operational metrics query")
        
        # Get current metrics from all sources
        bed_metrics = await self.bed_analyzer.get_metrics()
        flow_metrics = await self.flow_predictor.get_current_flow_metrics()
        resource_metrics = await self._get_resource_utilization()
        staffing_metrics = await self._get_current_staffing()
        
        # Create comprehensive metrics object
        metrics = OperationsMetrics(
            total_beds=bed_metrics.get("total_beds", 500),
            occupied_beds=bed_metrics.get("occupied_beds", 390),
            available_beds=bed_metrics.get("available_beds", 110),
            occupancy_rate=bed_metrics.get("occupancy_rate", 0.78),
            icu_beds_total=bed_metrics.get("icu_beds_total", 50),
            icu_beds_occupied=bed_metrics.get("icu_beds_occupied", 42),
            icu_occupancy_rate=bed_metrics.get("icu_occupancy_rate", 0.84),
            patients_in_er=flow_metrics.er_patients,
            patients_waiting_admission=flow_metrics.waiting_admission,
            average_wait_time_minutes=flow_metrics.average_wait_time,
            discharges_today=flow_metrics.discharges_today,
            expected_discharges_today=flow_metrics.expected_discharges,
            ventilator_utilization=resource_metrics.get(ResourceType.VENTILATOR.value, {}).get("utilization_rate", 0),
            or_utilization=resource_metrics.get(ResourceType.OPERATING_ROOM.value, {}).get("utilization_rate", 0),
            nurses_on_duty=staffing_metrics.get("nurses", {}).get("currently_on_duty", 0),
            doctors_on_duty=staffing_metrics.get("doctors", {}).get("currently_on_duty", 0),
            staff_to_patient_ratio=self._calculate_staff_patient_ratio(staffing_metrics, bed_metrics)
        )
        
        # Cache metrics
        self._current_metrics = metrics
        self._historical_metrics.append(metrics)
        if len(self._historical_metrics) > 100:
            self._historical_metrics.pop(0)
        
        response = {
            "query_type": "operational_metrics",
            "metrics": metrics.dict(),
            "trends": self._calculate_trends(),
            "alerts": await self._check_occupancy_alerts(
                await self.bed_analyzer.analyze_current_occupancy()
            ),
            "timestamp": datetime.now().isoformat()
        }
        
        return response
    
    def _calculate_staff_patient_ratio(
        self,
        staffing_metrics: Dict[str, Any],
        bed_metrics: Dict[str, Any]
    ) -> float:
        """
        Calculate overall staff to patient ratio.
        
        Args:
            staffing_metrics: Current staffing data
            bed_metrics: Current bed occupancy data
            
        Returns:
            Staff to patient ratio
        """
        total_staff = (
            staffing_metrics.get("nurses", {}).get("currently_on_duty", 0) +
            staffing_metrics.get("doctors", {}).get("currently_on_duty", 0)
        )
        
        total_patients = bed_metrics.get("occupied_beds", 1)  # Avoid division by zero
        
        return total_staff / total_patients if total_patients > 0 else 0
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """
        Calculate operational trends from historical metrics.
        
        Returns:
            Dictionary with trend analysis
        """
        if len(self._historical_metrics) < 2:
            return {}
        
        # Calculate trends for key metrics
        recent = self._historical_metrics[-1]
        previous = self._historical_metrics[-2]
        
        return {
            "occupancy_trend": recent.occupancy_rate - previous.occupancy_rate,
            "wait_time_trend": recent.average_wait_time_minutes - previous.average_wait_time_minutes,
            "discharge_trend": recent.discharges_today - previous.discharges_today
        }
    
    async def _handle_bottleneck_query(
        self,
        query: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle queries for operational bottleneck detection.
        
        Args:
            query: Original user query
            parameters: Additional parameters from the request
            
        Returns:
            Dictionary with bottleneck analysis
        """
        logger.info("Handling bottleneck detection query")
        
        # Get bottlenecks from various sources
        flow_bottlenecks = await self.flow_predictor.identify_bottlenecks()
        discharge_bottlenecks = await self._identify_discharge_bottlenecks()
        
        # Analyze system constraints
        system_constraints = await self._analyze_system_constraints()
        
        response = {
            "query_type": "bottleneck_detection",
            "flow_bottlenecks": flow_bottlenecks,
            "discharge_bottlenecks": discharge_bottlenecks,
            "system_constraints": system_constraints,
            "impact_analysis": await self._analyze_bottleneck_impact(flow_bottlenecks),
            "recommendations": await self._generate_bottleneck_recommendations(flow_bottlenecks),
            "timestamp": datetime.now().isoformat()
        }
        
        return response
    
    async def _analyze_system_constraints(self) -> List[Dict[str, Any]]:
        """
        Analyze system-wide constraints affecting operations.
        
        Returns:
            List of system constraints
        """
        return [
            {
                "constraint": "Bed capacity",
                "severity": "high",
                "current_state": "85% occupied",
                "impact": "Delays in admissions from ED"
            },
            {
                "constraint": "Staff availability",
                "severity": "medium",
                "current_state": "8% vacancy rate",
                "impact": "Extended patient wait times"
            }
        ]
    
    async def _analyze_bottleneck_impact(self, bottlenecks: List[str]) -> Dict[str, Any]:
        """
        Analyze the impact of identified bottlenecks.
        
        Args:
            bottlenecks: List of identified bottlenecks
            
        Returns:
            Dictionary with impact analysis
        """
        # This would use simulation models in production
        
        return {
            "average_delay_increase": "45 minutes",
            "patients_affected": "30-40 per day",
            "financial_impact": "$15,000 per day",
            "quality_impact": "Decreased patient satisfaction scores"
        }
    
    async def _generate_bottleneck_recommendations(
        self,
        bottlenecks: List[str]
    ) -> List[str]:
        """
        Generate recommendations for addressing bottlenecks.
        
        Args:
            bottlenecks: List of identified bottlenecks
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        for bottleneck in bottlenecks:
            if "bed" in bottleneck.lower():
                recommendations.append("Implement rapid discharge protocol for stable patients")
                recommendations.append("Activate surge capacity plan")
            elif "staff" in bottleneck.lower():
                recommendations.append("Initiate call-in list for additional staff")
                recommendations.append("Redistribute staff from low-acuity areas")
        
        return recommendations
    
    async def _handle_general_operational_query(
        self,
        query: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle general operational queries that don't fit specific categories.
        
        Args:
            query: Original user query
            parameters: Additional parameters from the request
            
        Returns:
            Dictionary with general operational information
        """
        logger.info("Handling general operational query")
        
        # Get comprehensive operational snapshot
        metrics = await self._handle_metrics_query(query, parameters)
        
        response = {
            "query_type": "general_operational",
            "summary": "Here's a summary of current hospital operations:",
            "key_highlights": {
                "occupancy": f"{metrics['metrics']['occupancy_rate']:.1%} overall",
                "wait_time": f"{metrics['metrics']['average_wait_time_minutes']} minutes average ER wait",
                "discharges": f"{metrics['metrics']['discharges_today']} discharged today",
                "icu_occupancy": f"{metrics['metrics']['icu_occupancy_rate']:.1%}"
            },
            "detailed_metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        return response
    
    def _needs_human_confirmation(self, response_content: Dict[str, Any]) -> bool:
        """
        Determine if response requires human confirmation based on content.
        
        Args:
            response_content: Response content to evaluate
            
        Returns:
            True if human confirmation needed, False otherwise
        """
        # Check for critical alerts
        if response_content.get("alerts"):
            for alert in response_content.get("alerts", []):
                if alert.get("level") == "CRITICAL":
                    return True
        
        # Check for resource shortage recommendations
        if response_content.get("critical_shortages"):
            return True
        
        # Check for critical operational decisions
        if response_content.get("recommendations"):
            for rec in response_content.get("recommendations", []):
                if isinstance(rec, dict) and rec.get("priority") == "high":
                    return True
        
        return False
    
    async def get_operational_dashboard(self) -> Dict[str, Any]:
        """
        Generate comprehensive operational dashboard data.
        This method aggregates all operational data for display in dashboards.
        
        Returns:
            Dictionary with complete operational dashboard data
        """
        return {
            "current_metrics": self._current_metrics.dict() if self._current_metrics else {},
            "resource_alerts": self._resource_alerts,
            "predictions": {
                "occupancy": await self.bed_analyzer.predict_occupancy(self.prediction_horizon_hours),
                "flow": await self.flow_predictor.predict_patient_flow(self.prediction_horizon_hours)
            },
            "recommendations": self._generate_operational_recommendations(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_operational_recommendations(self) -> List[str]:
        """
        Generate operational recommendations based on current metrics.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not self._current_metrics:
            return recommendations
        
        if self._current_metrics.occupancy_rate > 0.85:
            recommendations.append("Activate surge capacity protocol - occupancy exceeds 85%")
        
        if self._current_metrics.icu_occupancy_rate > 0.9:
            recommendations.append("ICU at critical levels - review transfer criteria")
        
        if self._current_metrics.average_wait_time_minutes > 60:
            recommendations.append("ER wait times exceed 60 minutes - consider opening additional triage")
        
        if self._current_metrics.staff_to_patient_ratio < 0.2:
            recommendations.append("Low staff-to-patient ratio - request additional staff")
        
        return recommendations
    
    async def can_handle(self, task_type: str, context: Dict[str, Any]) -> float:
        """
        Determine if this agent can handle the given task.
        
        Args:
            task_type: Type of task to evaluate
            context: Additional context for decision
            
        Returns:
            Confidence score (0-1) for handling capability
        """
        operational_keywords = [
            "operational", "bed", "occupancy", "capacity", "flow",
            "resource", "staff", "schedule", "wait time", "discharge",
            "admission", "emergency", "er", "icu", "bottleneck",
            "efficiency", "throughput", "utilization"
        ]
        
        # Check if task type matches
        if task_type in ["operations", "resource", "bed_management"]:
            return 0.95
        
        # Check context for relevant keywords
        context_text = str(context).lower()
        matches = sum(1 for keyword in operational_keywords if keyword in context_text)
        
        if matches > 0:
            return min(0.5 + (matches * 0.1), 0.9)
        
        return 0.1