"""
Director Intelligence Agent - Main Class

This agent serves as the primary strategic advisor for hospital leadership,
integrating data from various sources to provide actionable insights,
performance analysis, and strategic recommendations.

The agent uses a combination of:
    - Historical data analysis
    - Real-time KPI monitoring
    - Predictive modeling
    - Industry benchmarking
    - Natural language generation for reports
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import json
import hashlib
from collections import defaultdict

# Configure logging for the director agent
logger = logging.getLogger(__name__)

class StrategicPriority(Enum):
    """
    Strategic priorities for hospital management.
    These represent the key focus areas for leadership decisions.
    """
    PATIENT_SAFETY = "patient_safety"
    CLINICAL_QUALITY = "clinical_quality" 
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    FINANCIAL_PERFORMANCE = "financial_performance"
    STAFF_SATISFACTION = "staff_satisfaction"
    PATIENT_EXPERIENCE = "patient_experience"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    INNOVATION = "innovation"

class PerformanceTrend(Enum):
    """
    Trend indicators for performance metrics.
    Used to show direction and magnitude of changes.
    """
    SIGNIFICANT_IMPROVEMENT = "significant_improvement"
    MODERATE_IMPROVEMENT = "moderate_improvement"
    STABLE = "stable"
    MODERATE_DECLINE = "moderate_decline"
    SIGNIFICANT_DECLINE = "significant_decline"
    INSUFFICIENT_DATA = "insufficient_data"

@dataclass
class KPIThresholds:
    """
    Threshold definitions for Key Performance Indicators.
    These define target ranges and alert levels for each metric.
    """
    warning_lower: float
    warning_upper: float
    critical_lower: float
    critical_upper: float
    target_min: float
    target_max: float
    
    def get_status(self, value: float) -> str:
        """
        Determine the status of a KPI based on its value.
        
        Args:
            value: Current KPI value
            
        Returns:
            Status string: 'critical', 'warning', 'good', or 'excellent'
        """
        if value < self.critical_lower or value > self.critical_upper:
            return "critical"
        elif value < self.warning_lower or value > self.warning_upper:
            return "warning"
        elif self.target_min <= value <= self.target_max:
            return "excellent"
        else:
            return "good"

@dataclass
class DepartmentMetrics:
    """
    Comprehensive metrics for a hospital department.
    This class holds all performance data for a single department.
    """
    department_id: str
    department_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    historical_data: Dict[str, List[float]] = field(default_factory=dict)
    benchmarks: Dict[str, float] = field(default_factory=dict)
    trends: Dict[str, PerformanceTrend] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

class DirectorIntelligenceAgent:
    """
    Main Director Intelligence Agent class.
    
    This agent provides strategic intelligence for hospital leadership by:
        1. Aggregating data from multiple hospital systems
        2. Analyzing KPIs across departments and time periods
        3. Identifying trends and anomalies
        4. Generating strategic recommendations
        5. Creating executive reports
        6. Benchmarking against industry standards
        7. Forecasting future performance
        8. Alerting on critical issues
    
    The agent maintains a comprehensive view of hospital operations,
    clinical quality, financial performance, and strategic initiatives.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str = "Director Intelligence Agent",
        config_path: Optional[str] = None,
        data_sources: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Director Intelligence Agent.
        
        Args:
            agent_id: Unique identifier for this agent instance
            name: Human-readable name for the agent
            config_path: Path to configuration file (optional)
            data_sources: Dictionary of data source connections (optional)
        
        The initialization process:
            1. Sets up basic agent identity and configuration
            2. Initializes data structures for metrics storage
            3. Loads KPI definitions and thresholds
            4. Establishes connections to data sources
            5. Sets up caching for performance
            6. Initializes reporting templates
        """
        # Basic agent identification
        self.agent_id = agent_id
        self.name = name
        self.created_at = datetime.now()
        
        # Configuration
        self.config = self._load_configuration(config_path) if config_path else {}
        
        # Data source connections (would be actual DB connections in production)
        self.data_sources = data_sources or {}
        
        # Core data structures
        self.departments: Dict[str, DepartmentMetrics] = {}
        self.hospital_wide_metrics: Dict[str, float] = {}
        self.kpi_definitions: Dict[str, Dict[str, Any]] = self._initialize_kpi_definitions()
        self.thresholds: Dict[str, KPIThresholds] = self._initialize_thresholds()
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.alert_history: List[Dict[str, Any]] = []
        self.report_cache: Dict[str, Dict[str, Any]] = {}
        
        # Strategic planning
        self.current_priorities: List[StrategicPriority] = [
            StrategicPriority.PATIENT_SAFETY,
            StrategicPriority.CLINICAL_QUALITY,
            StrategicPriority.OPERATIONAL_EFFICIENCY
        ]
        
        # Initialize connections to other systems
        self._initialize_data_pipeline()
        
        # Start background monitoring tasks
        self.monitoring_task = None
        
        logger.info(f"Director Intelligence Agent initialized with ID: {agent_id}")
        
    def _load_configuration(self, config_path: str) -> Dict[str, Any]:
        """
        Load agent configuration from file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            Dictionary of configuration settings
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            return {}
    
    def _initialize_kpi_definitions(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize definitions for all Key Performance Indicators.
        
        These definitions specify:
            - What the KPI measures
            - How it's calculated
            - Data sources needed
            - Target values and thresholds
            - Reporting frequency
            
        Returns:
            Dictionary of KPI definitions
        """
        return {
            # Clinical Quality KPIs
            "mortality_rate": {
                "name": "Hospital Mortality Rate",
                "description": "Risk-adjusted mortality rate for all inpatients",
                "category": "clinical_quality",
                "unit": "percentage",
                "calculation": "(deaths / total_discharges) * 100",
                "data_sources": ["patient_records", "discharge_data"],
                "frequency": "daily",
                "direction": "lower_is_better"
            },
            "readmission_rate": {
                "name": "30-Day Readmission Rate",
                "description": "Percentage of patients readmitted within 30 days",
                "category": "clinical_quality",
                "unit": "percentage",
                "calculation": "(readmissions_30d / total_discharges) * 100",
                "data_sources": ["patient_records", "admissions_data"],
                "frequency": "weekly",
                "direction": "lower_is_better"
            },
            "hospital_acquired_infections": {
                "name": "Hospital-Acquired Infections",
                "description": "Rate of infections acquired during hospital stay",
                "category": "patient_safety",
                "unit": "rate_per_1000_patient_days",
                "calculation": "(infections / total_patient_days) * 1000",
                "data_sources": ["infection_control", "patient_days"],
                "frequency": "daily",
                "direction": "lower_is_better"
            },
            
            # Operational Efficiency KPIs
            "average_length_of_stay": {
                "name": "Average Length of Stay",
                "description": "Average number of days patients spend in hospital",
                "category": "operational_efficiency",
                "unit": "days",
                "calculation": "total_patient_days / total_discharges",
                "data_sources": ["admissions_data", "discharge_data"],
                "frequency": "daily",
                "direction": "moderate_is_better"
            },
            "bed_occupancy_rate": {
                "name": "Bed Occupancy Rate",
                "description": "Percentage of available beds that are occupied",
                "category": "operational_efficiency",
                "unit": "percentage",
                "calculation": "(occupied_beds / total_beds) * 100",
                "data_sources": ["bed_management"],
                "frequency": "hourly",
                "direction": "moderate_is_better"
            },
            "emergency_department_wait_time": {
                "name": "ED Wait Time",
                "description": "Average time patients wait in emergency department",
                "category": "patient_experience",
                "unit": "minutes",
                "calculation": "avg(time_from_arrival_to_treatment)",
                "data_sources": ["ed_tracking"],
                "frequency": "hourly",
                "direction": "lower_is_better"
            },
            
            # Financial KPIs
            "revenue_per_patient_day": {
                "name": "Revenue per Patient Day",
                "description": "Average revenue generated per patient day",
                "category": "financial_performance",
                "unit": "dollars",
                "calculation": "total_revenue / total_patient_days",
                "data_sources": ["billing", "patient_days"],
                "frequency": "daily",
                "direction": "higher_is_better"
            },
            "cost_per_discharge": {
                "name": "Cost per Discharge",
                "description": "Average cost per patient discharge",
                "category": "financial_performance",
                "unit": "dollars",
                "calculation": "total_costs / total_discharges",
                "data_sources": ["cost_accounting", "discharge_data"],
                "frequency": "weekly",
                "direction": "lower_is_better"
            },
            "staff_overtime_percentage": {
                "name": "Staff Overtime Percentage",
                "description": "Percentage of total hours that are overtime",
                "category": "staff_satisfaction",
                "unit": "percentage",
                "calculation": "(overtime_hours / total_hours) * 100",
                "data_sources": ["hr_system", "scheduling"],
                "frequency": "weekly",
                "direction": "lower_is_better"
            },
            
            # Patient Experience KPIs
            "patient_satisfaction_score": {
                "name": "Patient Satisfaction Score",
                "description": "Average patient satisfaction rating from surveys",
                "category": "patient_experience",
                "unit": "score",
                "calculation": "avg(survey_scores)",
                "data_sources": ["patient_surveys"],
                "frequency": "monthly",
                "direction": "higher_is_better"
            },
            
            # Compliance KPIs
            "compliance_score": {
                "name": "Regulatory Compliance Score",
                "description": "Percentage of regulatory requirements met",
                "category": "regulatory_compliance",
                "unit": "percentage",
                "calculation": "(met_requirements / total_requirements) * 100",
                "data_sources": ["compliance_tracking"],
                "frequency": "weekly",
                "direction": "higher_is_better"
            },
            
            # Innovation KPIs
            "research_publications": {
                "name": "Research Publications",
                "description": "Number of research publications by staff",
                "category": "innovation",
                "unit": "count",
                "calculation": "count(publications)",
                "data_sources": ["research_tracking"],
                "frequency": "monthly",
                "direction": "higher_is_better"
            },
            "clinical_trials_active": {
                "name": "Active Clinical Trials",
                "description": "Number of ongoing clinical trials",
                "category": "innovation",
                "unit": "count",
                "calculation": "count(active_trials)",
                "data_sources": ["research_tracking"],
                "frequency": "weekly",
                "direction": "higher_is_better"
            }
        }
    
    def _initialize_thresholds(self) -> Dict[str, KPIThresholds]:
        """
        Initialize threshold values for all KPIs.
        
        These thresholds define acceptable ranges and alert conditions
        for each KPI based on industry standards and hospital policies.
        
        Returns:
            Dictionary mapping KPI names to KPIThresholds objects
        """
        thresholds = {}
        
        # Mortality rate thresholds (percentage)
        thresholds["mortality_rate"] = KPIThresholds(
            warning_lower=0.0,  # Not applicable for lower bound
            warning_upper=1.5,   # Warning at 1.5
            critical_lower=0.0,
            critical_upper=2.0,  # Critical at 2.0
            target_min=0.0,
            target_max=1.0        # Target range 0-1
        )
        
        # Readmission rate thresholds (percentage)
        thresholds["readmission_rate"] = KPIThresholds(
            warning_lower=0.0,
            warning_upper=15.0,
            critical_lower=0.0,
            critical_upper=20.0,
            target_min=0.0,
            target_max=12.0
        )
        
        # Hospital-acquired infections (per 1000 patient days)
        thresholds["hospital_acquired_infections"] = KPIThresholds(
            warning_lower=0.0,
            warning_upper=2.0,
            critical_lower=0.0,
            critical_upper=3.0,
            target_min=0.0,
            target_max=1.0
        )
        
        # Average length of stay (days)
        thresholds["average_length_of_stay"] = KPIThresholds(
            warning_lower=3.0,   # Too low might indicate premature discharge
            warning_upper=7.0,   # Too high indicates inefficiency
            critical_lower=2.0,
            critical_upper=10.0,
            target_min=4.0,
            target_max=6.0
        )
        
        # Bed occupancy rate (percentage)
        thresholds["bed_occupancy_rate"] = KPIThresholds(
            warning_lower=70.0,   # Below 70 might be underutilized
            warning_upper=90.0,   # Above 90 might be overcrowded
            critical_lower=60.0,
            critical_upper=95.0,
            target_min=75.0,
            target_max=85.0
        )
        
        # ED wait time (minutes)
        thresholds["emergency_department_wait_time"] = KPIThresholds(
            warning_lower=0.0,
            warning_upper=30.0,
            critical_lower=0.0,
            critical_upper=45.0,
            target_min=0.0,
            target_max=20.0
        )
        
        # Revenue per patient day (dollars)
        thresholds["revenue_per_patient_day"] = KPIThresholds(
            warning_lower=2000.0,
            warning_upper=float('inf'),
            critical_lower=1500.0,
            critical_upper=float('inf'),
            target_min=2500.0,
            target_max=float('inf')
        )
        
        # Cost per discharge (dollars)
        thresholds["cost_per_discharge"] = KPIThresholds(
            warning_lower=0.0,
            warning_upper=15000.0,
            critical_lower=0.0,
            critical_upper=20000.0,
            target_min=0.0,
            target_max=12000.0
        )
        
        # Staff overtime percentage
        thresholds["staff_overtime_percentage"] = KPIThresholds(
            warning_lower=0.0,
            warning_upper=10.0,
            critical_lower=0.0,
            critical_upper=15.0,
            target_min=0.0,
            target_max=5.0
        )
        
        # Patient satisfaction score (1-10 scale)
        thresholds["patient_satisfaction_score"] = KPIThresholds(
            warning_lower=7.0,
            warning_upper=float('inf'),
            critical_lower=5.0,
            critical_upper=float('inf'),
            target_min=8.5,
            target_max=10.0
        )
        
        # Compliance score (percentage)
        thresholds["compliance_score"] = KPIThresholds(
            warning_lower=90.0,
            warning_upper=float('inf'),
            critical_lower=85.0,
            critical_upper=float('inf'),
            target_min=95.0,
            target_max=100.0
        )
        
        return thresholds
    
    def _initialize_data_pipeline(self) -> None:
        """
        Initialize the data pipeline connections.
        
        This method sets up connections to various hospital data systems:
            - EHR/EMR systems for clinical data
            - ADT system for admissions/discharges/transfers
            - Billing system for financial data
            - HR system for staffing data
            - Patient satisfaction survey system
            - Regulatory compliance tracking system
        """
        logger.info("Initializing data pipeline connections...")
        
        # In production, these would be actual database connections
        # or API clients. Here we simulate with configuration.
        
        self.data_pipeline = {
            "ehr_system": self.data_sources.get("ehr", {}),
            "adt_system": self.data_sources.get("adt", {}),
            "billing_system": self.data_sources.get("billing", {}),
            "hr_system": self.data_sources.get("hr", {}),
            "survey_system": self.data_sources.get("surveys", {}),
            "compliance_system": self.data_sources.get("compliance", {})
        }
        
        logger.info("Data pipeline initialized successfully")
    
    async def start_monitoring(self) -> None:
        """
        Start background monitoring of KPIs.
        
        This method initiates continuous monitoring of all hospital KPIs,
        running periodic analyses and generating alerts when thresholds are exceeded.
        """
        logger.info("Starting KPI monitoring...")
        
        while True:
            try:
                # Refresh all KPIs
                await self.refresh_all_metrics()
                
                # Check for alerts
                alerts = await self.check_for_alerts()
                
                # Generate trend analysis
                trends = await self.analyze_trends()
                
                # Log monitoring cycle
                logger.info(f"Monitoring cycle complete. Alerts: {len(alerts)}, Trends analyzed: {len(trends)}")
                
                # Wait for next monitoring interval (configurable)
                await asyncio.sleep(self.config.get("monitoring_interval", 3600))  # Default: 1 hour
                
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def refresh_all_metrics(self) -> Dict[str, Any]:
        """
        Refresh all hospital metrics from data sources.
        
        This method:
            1. Fetches latest data from all connected systems
            2. Calculates current KPI values
            3. Updates department-level metrics
            4. Updates hospital-wide metrics
            5. Stores historical data
            
        Returns:
            Dictionary containing all updated metrics
        """
        logger.info("Refreshing all hospital metrics...")
        
        try:
            # Simulate data fetching from various systems
            # In production, these would be actual database queries
            
            # Fetch clinical data
            clinical_data = await self._fetch_clinical_data()
            
            # Fetch operational data
            operational_data = await self._fetch_operational_data()
            
            # Fetch financial data
            financial_data = await self._fetch_financial_data()
            
            # Fetch staffing data
            staffing_data = await self._fetch_staffing_data()
            
            # Combine all data
            all_data = {
                "clinical": clinical_data,
                "operational": operational_data,
                "financial": financial_data,
                "staffing": staffing_data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Calculate KPIs
            kpis = await self._calculate_all_kpis(all_data)
            
            # Update department metrics
            await self._update_department_metrics(all_data, kpis)
            
            # Update hospital-wide metrics
            self.hospital_wide_metrics = kpis
            
            # Store in history
            self._store_in_history(kpis)
            
            logger.info(f"Metrics refresh complete. {len(kpis)} KPIs updated.")
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "metrics": kpis,
                "departments": len(self.departments)
            }
            
        except Exception as e:
            logger.error(f"Failed to refresh metrics: {str(e)}")
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def _fetch_clinical_data(self) -> Dict[str, Any]:
        """
        Fetch clinical data from EHR/EMR systems.
        
        Returns:
            Dictionary containing patient outcomes, quality metrics, etc.
        """
        # Simulate data fetching
        # In production, this would query actual databases
        
        return {
            "total_patients": 1250,
            "discharges_last_24h": 85,
            "deaths_last_24h": 2,
            "readmissions_30d": 12,
            "hospital_acquired_infections": 3,
            "total_patient_days": 850,
            "icu_patients": 45,
            "ventilator_patients": 28
        }
    
    async def _fetch_operational_data(self) -> Dict[str, Any]:
        """
        Fetch operational data from ADT and bed management systems.
        
        Returns:
            Dictionary containing bed occupancy, wait times, etc.
        """
        return {
            "total_beds": 500,
            "occupied_beds": 425,
            "ed_arrivals_24h": 145,
            "average_ed_wait": 28.5,  # minutes
            "average_los": 5.2,  # days
            "surgery_count": 32,
            "cancelled_surgeries": 3
        }
    
    async def _fetch_financial_data(self) -> Dict[str, Any]:
        """
        Fetch financial data from billing and accounting systems.
        
        Returns:
            Dictionary containing revenue, costs, etc.
        """
        return {
            "daily_revenue": 1250000.00,
            "daily_costs": 980000.00,
            "total_billed": 15000000.00,
            "collections_rate": 0.92,
            "denial_rate": 0.05,
            "charity_care": 45000.00
        }
    
    async def _fetch_staffing_data(self) -> Dict[str, Any]:
        """
        Fetch staffing data from HR and scheduling systems.
        
        Returns:
            Dictionary containing staff metrics, overtime, etc.
        """
        return {
            "total_staff": 2150,
            "nurses_on_duty": 320,
            "doctors_on_duty": 85,
            "overtime_hours_yesterday": 450,
            "total_hours_yesterday": 8500,
            "vacancy_rate": 0.08,
            "turnover_rate_yearly": 0.12
        }
    
    async def _calculate_all_kpis(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate all KPI values from raw data.
        
        Args:
            data: Combined data from all sources
            
        Returns:
            Dictionary of KPI names to calculated values
        """
        kpis = {}
        
        # Clinical KPIs
        clinical = data.get("clinical", {})
        if clinical.get("discharges_last_24h", 0) > 0:
            kpis["mortality_rate"] = (
                clinical.get("deaths_last_24h", 0) / 
                clinical.get("discharges_last_24h", 1) * 100
            )
        
        if clinical.get("discharges_last_24h", 0) > 0:
            kpis["readmission_rate"] = (
                clinical.get("readmissions_30d", 0) / 
                clinical.get("discharges_last_24h", 1) * 100
            )
        
        if clinical.get("total_patient_days", 0) > 0:
            kpis["hospital_acquired_infections"] = (
                clinical.get("hospital_acquired_infections", 0) / 
                clinical.get("total_patient_days", 1) * 1000
            )
        
        # Operational KPIs
        operational = data.get("operational", {})
        if operational.get("discharges_last_24h", 0) > 0:
            kpis["average_length_of_stay"] = (
                clinical.get("total_patient_days", 0) / 
                operational.get("discharges_last_24h", 1)
            )
        
        if operational.get("total_beds", 0) > 0:
            kpis["bed_occupancy_rate"] = (
                operational.get("occupied_beds", 0) / 
                operational.get("total_beds", 1) * 100
            )
        
        kpis["emergency_department_wait_time"] = operational.get("average_ed_wait", 0)
        
        # Financial KPIs
        financial = data.get("financial", {})
        if clinical.get("total_patient_days", 0) > 0:
            kpis["revenue_per_patient_day"] = (
                financial.get("daily_revenue", 0) / 
                clinical.get("total_patient_days", 1)
            )
        
        if operational.get("discharges_last_24h", 0) > 0:
            kpis["cost_per_discharge"] = (
                financial.get("daily_costs", 0) / 
                operational.get("discharges_last_24h", 1)
            )
        
        # Staffing KPIs
        staffing = data.get("staffing", {})
        if staffing.get("total_hours_yesterday", 0) > 0:
            kpis["staff_overtime_percentage"] = (
                staffing.get("overtime_hours_yesterday", 0) / 
                staffing.get("total_hours_yesterday", 1) * 100
            )
        
        # Add some simulated values for KPIs that need longer timeframes
        kpis["patient_satisfaction_score"] = 8.7  # Simulated
        kpis["compliance_score"] = 94.5  # Simulated
        kpis["research_publications"] = 8  # Simulated
        kpis["clinical_trials_active"] = 15  # Simulated
        
        return kpis
    
    async def _update_department_metrics(
        self, 
        data: Dict[str, Any], 
        kpis: Dict[str, float]
    ) -> None:
        """
        Update metrics for each department.
        
        Args:
            data: Raw data from all sources
            kpis: Calculated KPI values
        """
        # Define departments (in production, these would come from database)
        departments = [
            {"id": "cardio", "name": "Cardiology"},
            {"id": "emergency", "name": "Emergency Department"},
            {"id": "icu", "name": "Intensive Care Unit"},
            {"id": "surgery", "name": "Surgery"},
            {"id": "pediatrics", "name": "Pediatrics"},
            {"id": "oncology", "name": "Oncology"},
            {"id": "ortho", "name": "Orthopedics"},
            {"id": "radiology", "name": "Radiology"}
        ]
        
        for dept in departments:
            if dept["id"] not in self.departments:
                # Create new department metrics
                self.departments[dept["id"]] = DepartmentMetrics(
                    department_id=dept["id"],
                    department_name=dept["name"]
                )
            
            # Update department metrics (simulated distribution)
            dept_metrics = {}
            for kpi_name, value in kpis.items():
                # Add some random variation for departments
                variation = np.random.normal(1.0, 0.15)  # 15% standard deviation
                dept_metrics[kpi_name] = value * variation
            
            # Store historical data
            dept_obj = self.departments[dept["id"]]
            for kpi_name, value in dept_metrics.items():
                if kpi_name not in dept_obj.historical_data:
                    dept_obj.historical_data[kpi_name] = []
                dept_obj.historical_data[kpi_name].append(value)
                # Keep last 30 days of data
                if len(dept_obj.historical_data[kpi_name]) > 30:
                    dept_obj.historical_data[kpi_name].pop(0)
            
            dept_obj.metrics = dept_metrics
            dept_obj.last_updated = datetime.now()
    
    def _store_in_history(self, kpis: Dict[str, float]) -> None:
        """
        Store KPI values in history for trend analysis.
        
        Args:
            kpis: Current KPI values
        """
        timestamp = datetime.now()
        
        for kpi_name, value in kpis.items():
            self.performance_history[kpi_name].append({
                "timestamp": timestamp,
                "value": value
            })
            
            # Keep only last 90 days of data
            if len(self.performance_history[kpi_name]) > 90:
                self.performance_history[kpi_name].pop(0)
    
    async def check_for_alerts(self) -> List[Dict[str, Any]]:
        """
        Check all KPIs for threshold violations and generate alerts.
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        for kpi_name, value in self.hospital_wide_metrics.items():
            if kpi_name in self.thresholds:
                threshold = self.thresholds[kpi_name]
                status = threshold.get_status(value)
                
                if status in ["critical", "warning"]:
                    alert = {
                        "alert_id": hashlib.md5(f"{kpi_name}_{datetime.now()}".encode()).hexdigest()[:8],
                        "kpi": kpi_name,
                        "value": value,
                        "status": status,
                        "thresholds": {
                            "warning": {"lower": threshold.warning_lower, "upper": threshold.warning_upper},
                            "critical": {"lower": threshold.critical_lower, "upper": threshold.critical_upper},
                            "target": {"min": threshold.target_min, "max": threshold.target_max}
                        },
                        "timestamp": datetime.now().isoformat(),
                        "acknowledged": False,
                        "resolved": False
                    }
                    
                    alerts.append(alert)
                    self.alert_history.append(alert)
                    
                    logger.warning(f"Alert generated: {kpi_name} = {value} ({status})")
        
        return alerts
    
    async def analyze_trends(self, days: int = 30) -> Dict[str, PerformanceTrend]:
        """
        Analyze trends in KPIs over time.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary mapping KPI names to PerformanceTrend values
        """
        trends = {}
        
        for kpi_name, history in self.performance_history.items():
            if len(history) < 7:  # Need at least a week of data
                trends[kpi_name] = PerformanceTrend.INSUFFICIENT_DATA
                continue
            
            # Get values for the specified period
            cutoff = datetime.now() - timedelta(days=days)
            relevant_history = [h for h in history if h["timestamp"] >= cutoff]
            
            if len(relevant_history) < 7:
                trends[kpi_name] = PerformanceTrend.INSUFFICIENT_DATA
                continue
            
            # Calculate trend using linear regression
            values = [h["value"] for h in relevant_history]
            x = np.arange(len(values))
            z = np.polyfit(x, values, 1)
            slope = z[0]  # Slope of the trend line
            
            # Normalize slope by mean value for relative comparison
            mean_value = np.mean(values)
            if mean_value > 0:
                relative_slope = slope / mean_value
            else:
                relative_slope = 0
            
            # Determine trend direction and significance
            kpi_def = self.kpi_definitions.get(kpi_name, {})
            direction = kpi_def.get("direction", "lower_is_better")
            
            # Determine trend category
            if abs(relative_slope) < 0.01:  # Less than 1% change
                trend = PerformanceTrend.STABLE
            elif relative_slope > 0:
                if direction == "higher_is_better":
                    trend = PerformanceTrend.SIGNIFICANT_IMPROVEMENT if relative_slope > 0.05 else PerformanceTrend.MODERATE_IMPROVEMENT
                else:
                    trend = PerformanceTrend.SIGNIFICANT_DECLINE if relative_slope > 0.05 else PerformanceTrend.MODERATE_DECLINE
            else:  # slope < 0
                if direction == "higher_is_better":
                    trend = PerformanceTrend.SIGNIFICANT_DECLINE if abs(relative_slope) > 0.05 else PerformanceTrend.MODERATE_DECLINE
                else:
                    trend = PerformanceTrend.SIGNIFICANT_IMPROVEMENT if abs(relative_slope) > 0.05 else PerformanceTrend.MODERATE_IMPROVEMENT
            
            trends[kpi_name] = trend
        
        return trends
    
    async def get_strategic_insights(self) -> Dict[str, Any]:
        """
        Generate strategic insights based on current performance.
        
        This method analyzes all available data to provide:
            - Key areas of concern
            - Opportunities for improvement
            - Strategic recommendations
            - Risk assessments
            - Performance forecasts
            
        Returns:
            Dictionary containing strategic insights
        """
        insights = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "areas_of_concern": [],
            "opportunities": [],
            "recommendations": [],
            "risks": [],
            "forecasts": {}
        }
        
        # Analyze current KPIs
        for kpi_name, value in self.hospital_wide_metrics.items():
            if kpi_name in self.thresholds:
                status = self.thresholds[kpi_name].get_status(value)
                
                if status == "critical":
                    insights["areas_of_concern"].append({
                        "kpi": kpi_name,
                        "value": value,
                        "severity": "critical",
                        "description": self.kpi_definitions[kpi_name]["description"]
                    })
                elif status == "warning":
                    insights["areas_of_concern"].append({
                        "kpi": kpi_name,
                        "value": value,
                        "severity": "warning",
                        "description": self.kpi_definitions[kpi_name]["description"]
                    })
        
        # Analyze trends
        trends = await self.analyze_trends()
        for kpi_name, trend in trends.items():
            if trend in [PerformanceTrend.SIGNIFICANT_IMPROVEMENT, PerformanceTrend.MODERATE_IMPROVEMENT]:
                insights["opportunities"].append({
                    "kpi": kpi_name,
                    "trend": trend.value,
                    "description": f"Positive trend in {self.kpi_definitions[kpi_name]['name']}"
                })
        
        # Generate recommendations based on analysis
        insights["recommendations"] = self._generate_recommendations()
        
        # Assess risks
        insights["risks"] = self._assess_risks()
        
        # Generate forecasts
        insights["forecasts"] = await self._generate_forecasts()
        
        # Overall summary
        insights["summary"] = {
            "total_kpis": len(self.hospital_wide_metrics),
            "critical_alerts": len([a for a in insights["areas_of_concern"] if a["severity"] == "critical"]),
            "warning_alerts": len([a for a in insights["areas_of_concern"] if a["severity"] == "warning"]),
            "positive_trends": len(insights["opportunities"]),
            "overall_health": self._calculate_overall_health()
        }
        
        return insights
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate strategic recommendations based on current performance.
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Check specific KPIs for recommendations
        for kpi_name, value in self.hospital_wide_metrics.items():
            if kpi_name not in self.thresholds:
                continue
            
            status = self.thresholds[kpi_name].get_status(value)
            
            if status == "critical":
                if kpi_name == "mortality_rate":
                    recommendations.append({
                        "priority": "high",
                        "category": "clinical_quality",
                        "title": "Review Mortality Cases",
                        "description": "Conduct immediate review of all mortality cases from the last 30 days",
                        "action_items": [
                            "Schedule mortality review committee meeting",
                            "Analyze patterns in mortality cases",
                            "Review care protocols for affected conditions"
                        ],
                        "expected_impact": "Identify systemic issues and reduce preventable deaths"
                    })
                
                elif kpi_name == "readmission_rate":
                    recommendations.append({
                        "priority": "high",
                        "category": "clinical_quality",
                        "title": "Enhance Discharge Planning",
                        "description": "Implement comprehensive discharge planning program",
                        "action_items": [
                            "Assign discharge planners to high-risk patients",
                            "Improve patient education on post-discharge care",
                            "Schedule follow-up calls within 48 hours"
                        ],
                        "expected_impact": "Reduce 30-day readmissions by 15-20%"
                    })
                
                elif kpi_name == "emergency_department_wait_time":
                    recommendations.append({
                        "priority": "high",
                        "category": "operational_efficiency",
                        "title": "Optimize ED Workflow",
                        "description": "Implement lean methodology in emergency department",
                        "action_items": [
                            "Analyze patient flow bottlenecks",
                            "Implement rapid assessment protocol",
                            "Add fast-track for minor conditions"
                        ],
                        "expected_impact": "Reduce wait times by 25-30%"
                    })
                
                elif kpi_name == "staff_overtime_percentage":
                    recommendations.append({
                        "priority": "medium",
                        "category": "staff_satisfaction",
                        "title": "Address Staff Burnout",
                        "description": "Implement staff wellness and retention program",
                        "action_items": [
                            "Review staffing ratios",
                            "Implement flexible scheduling",
                            "Add wellness resources and support"
                        ],
                        "expected_impact": "Reduce turnover and improve staff satisfaction"
                    })
        
        return recommendations
    
    def _assess_risks(self) -> List[Dict[str, Any]]:
        """
        Assess current and future risks to the hospital.
        
        Returns:
            List of risk assessment dictionaries
        """
        risks = []
        
        # Financial risk assessment
        if "revenue_per_patient_day" in self.hospital_wide_metrics:
            revenue = self.hospital_wide_metrics["revenue_per_patient_day"]
            if revenue < 2000:
                risks.append({
                    "risk_level": "high",
                    "category": "financial",
                    "description": "Revenue per patient day below sustainable levels",
                    "mitigation": "Review pricing strategy and payer mix optimization"
                })
        
        # Operational risk assessment
        if "bed_occupancy_rate" in self.hospital_wide_metrics:
            occupancy = self.hospital_wide_metrics["bed_occupancy_rate"]
            if occupancy > 90:
                risks.append({
                    "risk_level": "high",
                    "category": "operational",
                    "description": "Bed occupancy rate critically high, risk of diversion",
                    "mitigation": "Implement surge capacity plan and expedite discharges"
                })
            elif occupancy < 70:
                risks.append({
                    "risk_level": "medium",
                    "category": "operational",
                    "description": "Low bed occupancy impacting revenue",
                    "mitigation": "Review referral patterns and marketing efforts"
                })
        
        # Clinical risk assessment
        if "mortality_rate" in self.hospital_wide_metrics:
            mortality = self.hospital_wide_metrics["mortality_rate"]
            if mortality > 2.0:
                risks.append({
                    "risk_level": "critical",
                    "category": "clinical",
                    "description": "Elevated mortality rate requires immediate attention",
                    "mitigation": "Immediate mortality review and protocol audit"
                })
        
        # Staffing risk assessment
        if "staff_overtime_percentage" in self.hospital_wide_metrics:
            overtime = self.hospital_wide_metrics["staff_overtime_percentage"]
            if overtime > 12:
                risks.append({
                    "risk_level": "high",
                    "category": "staffing",
                    "description": "High overtime indicates staffing shortage and burnout risk",
                    "mitigation": "Accelerate hiring and improve retention"
                })
        
        return risks
    
    async def _generate_forecasts(self) -> Dict[str, Any]:
        """
        Generate forecasts for key metrics using time series analysis.
        
        Returns:
            Dictionary of forecasts for various metrics
        """
        forecasts = {}
        
        for kpi_name, history in self.performance_history.items():
            if len(history) < 14:  # Need at least 14 days for forecasting
                continue
            
            # Get values
            values = [h["value"] for h in history[-30:]]  # Last 30 days
            
            # Simple moving average forecast
            if len(values) >= 7:
                # Use last 7 days to forecast next 7 days
                recent_trend = np.mean(values[-7:])
                overall_trend = np.mean(values)
                
                # Simple linear projection
                x = np.arange(len(values))
                z = np.polyfit(x, values, 1)
                slope = z[0]
                
                # Generate 7-day forecast
                forecast_values = []
                last_value = values[-1]
                for i in range(1, 8):
                    forecast_values.append(last_value + slope * i)
                
                forecasts[kpi_name] = {
                    "current_value": values[-1],
                    "forecast_7day": forecast_values[-1],
                    "trend_direction": "up" if slope > 0 else "down",
                    "confidence": 0.7 if len(values) > 30 else 0.5,
                    "values": forecast_values
                }
        
        return forecasts
    
    def _calculate_overall_health(self) -> Dict[str, Any]:
        """
        Calculate overall hospital health score.
        
        Returns:
            Dictionary with health metrics
        """
        total_kpis = 0
        weighted_score = 0
        
        # Weight categories
        category_weights = {
            "clinical_quality": 0.30,
            "patient_safety": 0.25,
            "operational_efficiency": 0.20,
            "financial_performance": 0.15,
            "staff_satisfaction": 0.05,
            "patient_experience": 0.05
        }
        
        category_scores = defaultdict(list)
        
        for kpi_name, value in self.hospital_wide_metrics.items():
            if kpi_name in self.thresholds:
                kpi_def = self.kpi_definitions.get(kpi_name, {})
                category = kpi_def.get("category", "other")
                
                # Calculate normalized score (0-100)
                threshold = self.thresholds[kpi_name]
                direction = kpi_def.get("direction", "lower_is_better")
                
                if direction == "higher_is_better":
                    if value >= threshold.target_max:
                        score = 100
                    elif value <= threshold.critical_lower:
                        score = 0
                    else:
                        # Linear interpolation
                        score = (value - threshold.critical_lower) / (threshold.target_max - threshold.critical_lower) * 100
                else:  # lower_is_better
                    if value <= threshold.target_min:
                        score = 100
                    elif value >= threshold.critical_upper:
                        score = 0
                    else:
                        score = (threshold.critical_upper - value) / (threshold.critical_upper - threshold.target_min) * 100
                
                # Cap at 0-100
                score = max(0, min(100, score))
                
                category_scores[category].append(score)
                total_kpis += 1
        
        # Calculate weighted score
        for category, scores in category_scores.items():
            if scores:
                avg_score = np.mean(scores)
                weighted_score += avg_score * category_weights.get(category, 0.05)
        
        return {
            "overall_score": round(weighted_score, 1),
            "category_scores": {
                cat: round(np.mean(scores), 1) 
                for cat, scores in category_scores.items() if scores
            },
            "total_kpis_measured": total_kpis,
            "assessment": self._get_health_assessment(weighted_score)
        }
    
    def _get_health_assessment(self, score: float) -> str:
        """
        Get text assessment of overall health based on score.
        
        Args:
            score: Overall health score (0-100)
            
        Returns:
            Assessment string
        """
        if score >= 90:
            return "Excellent - Hospital performing at peak levels"
        elif score >= 75:
            return "Good - Meeting most targets, minor improvements needed"
        elif score >= 60:
            return "Fair - Several areas need attention"
        elif score >= 40:
            return "Poor - Multiple areas require immediate action"
        else:
            return "Critical - Urgent intervention required"
    
    async def get_executive_summary(self) -> Dict[str, Any]:
        """
        Generate an executive summary for the Medical Director.
        
        Returns:
            Dictionary containing executive summary
        """
        insights = await self.get_strategic_insights()
        trends = await self.analyze_trends()
        
        # Get top performing and underperforming areas
        top_performers = []
        underperformers = []
        
        for kpi_name, value in self.hospital_wide_metrics.items():
            if kpi_name in self.thresholds:
                status = self.thresholds[kpi_name].get_status(value)
                kpi_def = self.kpi_definitions.get(kpi_name, {})
                
                item = {
                    "kpi": kpi_name,
                    "name": kpi_def.get("name", kpi_name),
                    "value": value,
                    "status": status,
                    "trend": trends.get(kpi_name, PerformanceTrend.INSUFFICIENT_DATA).value
                }
                
                if status == "excellent":
                    top_performers.append(item)
                elif status in ["critical", "warning"]:
                    underperformers.append(item)
        
        summary = {
            "generated_at": datetime.now().isoformat(),
            "period": "Last 24 hours",
            "overall_health": insights["summary"]["overall_health"],
            "critical_issues": insights["summary"]["critical_alerts"],
            "key_metrics": {
                "mortality_rate": self.hospital_wide_metrics.get("mortality_rate"),
                "readmission_rate": self.hospital_wide_metrics.get("readmission_rate"),
                "bed_occupancy": self.hospital_wide_metrics.get("bed_occupancy_rate"),
                "patient_satisfaction": self.hospital_wide_metrics.get("patient_satisfaction_score")
            },
            "top_performers": sorted(top_performers, key=lambda x: x["value"], reverse=True)[:3],
            "areas_needing_attention": underperformers[:5],
            "key_recommendations": [r for r in insights["recommendations"] if r["priority"] == "high"][:3],
            "risks": insights["risks"],
            "forecast": {
                k: v["forecast_7day"] 
                for k, v in insights["forecasts"].items() 
                if k in ["bed_occupancy_rate", "emergency_department_wait_time"]
            }
        }
        
        return summary
    
    async def get_department_performance(
        self, 
        department_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get performance data for departments.
        
        Args:
            department_id: Optional specific department ID
            
        Returns:
            List of department performance dictionaries
        """
        if department_id:
            depts = [self.departments.get(department_id)]
        else:
            depts = list(self.departments.values())
        
        results = []
        
        for dept in depts:
            if dept:
                # Calculate trends for this department
                dept_trends = {}
                for kpi_name, history in dept.historical_data.items():
                    if len(history) >= 7:
                        values = history[-7:]
                        slope = np.polyfit(range(len(values)), values, 1)[0]
                        dept_trends[kpi_name] = "up" if slope > 0 else "down"
                    else:
                        dept_trends[kpi_name] = "insufficient_data"
                
                results.append({
                    "department_id": dept.department_id,
                    "department_name": dept.department_name,
                    "metrics": dept.metrics,
                    "trends": dept_trends,
                    "alerts": dept.alerts[-5:] if dept.alerts else [],
                    "last_updated": dept.last_updated.isoformat()
                })
        
        return results

# Utility functions for external use

async def create_director_agent(
    agent_id: str,
    config_path: Optional[str] = None
) -> DirectorIntelligenceAgent:
    """
    Factory function to create and initialize a Director Intelligence Agent.
    
    Args:
        agent_id: Unique identifier for the agent
        config_path: Path to configuration file
        
    Returns:
        Initialized DirectorIntelligenceAgent instance
    """
    agent = DirectorIntelligenceAgent(agent_id, config_path=config_path)
    
    # Initial data load
    await agent.refresh_all_metrics()
    
    return agent

if __name__ == "__main__":
    # Example usage
    async def main():
        agent = await create_director_agent("director_001")
        summary = await agent.get_executive_summary()
        print(json.dumps(summary, indent=2))
    
    asyncio.run(main())