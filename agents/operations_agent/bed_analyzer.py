"""
Bed Analyzer Module - Specialized component for bed occupancy analysis.
This module provides detailed analysis of hospital bed utilization,
predicts future occupancy, and identifies optimization opportunities.
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class BedType(Enum):
    """
    Enumeration of different bed types in the hospital.
    """
    ICU = "icu"
    CCU = "ccu"  # Cardiac Care Unit
    MICU = "micu"  # Medical ICU
    SICU = "sicu"  # Surgical ICU
    NICU = "nicu"  # Neonatal ICU
    PICU = "picu"  # Pediatric ICU
    EMERGENCY = "emergency"
    MEDICAL_SURGICAL = "medical_surgical"
    TELEMETRY = "telemetry"
    STEP_DOWN = "step_down"
    PEDIATRIC = "pediatric"
    MATERNITY = "maternity"
    PSYCHIATRIC = "psychiatric"
    REHABILITATION = "rehabilitation"

class BedStatus(Enum):
    """
    Enumeration of possible bed statuses.
    """
    AVAILABLE = "available"
    OCCUPIED = "occupied"
    RESERVED = "reserved"
    CLEANING = "cleaning"
    MAINTENANCE = "maintenance"
    OUT_OF_SERVICE = "out_of_service"

class BedOccupancyAnalysis(BaseModel):
    """
    Comprehensive bed occupancy analysis results.
    """
    total_beds: int = Field(..., description="Total number of beds in hospital")
    occupied_beds: int = Field(..., description="Currently occupied beds")
    available_beds: int = Field(..., description="Currently available beds")
    occupancy_rate: float = Field(..., description="Overall occupancy rate")
    
    # Breakdown by department
    department_breakdown: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Occupancy by department"
    )
    
    # Breakdown by bed type
    bed_type_breakdown: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Occupancy by bed type"
    )
    
    # Occupancy predictions
    predicted_occupancy_6h: float = Field(0.0, description="Predicted occupancy in 6 hours")
    predicted_occupancy_12h: float = Field(0.0, description="Predicted occupancy in 12 hours")
    predicted_occupancy_24h: float = Field(0.0, description="Predicted occupancy in 24 hours")
    
    # Historical trends
    occupancy_trend_24h: List[float] = Field(
        default_factory=list,
        description="Occupancy trend over last 24 hours"
    )
    
    # Analysis timestamp
    analysis_time: datetime = Field(default_factory=datetime.now)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class BedData(BaseModel):
    """
    Individual bed data structure.
    """
    bed_id: str = Field(..., description="Unique bed identifier")
    bed_type: BedType = Field(..., description="Type of bed")
    department: str = Field(..., description="Department where bed is located")
    room_number: str = Field(..., description="Room number")
    status: BedStatus = Field(..., description="Current bed status")
    last_updated: datetime = Field(default_factory=datetime.now)
    current_patient_id: Optional[str] = Field(None, description="Current patient if occupied")
    estimated_discharge_time: Optional[datetime] = Field(None, description="Estimated discharge if occupied")
    cleaning_estimate_minutes: int = Field(30, description="Estimated cleaning time in minutes")
    equipment_available: List[str] = Field(default_factory=list, description="Available equipment")

class BedAnalyzer:
    """
    Specialized analyzer for bed occupancy data.
    Provides real-time analysis, predictions, and optimization recommendations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Bed Analyzer with configuration.
        
        Args:
            config: Configuration dictionary containing analysis parameters
        """
        self.config = config or {}
        
        # Configuration parameters
        self.bed_data_refresh_interval = self.config.get('bed_refresh_interval', 300)  # 5 minutes
        self.prediction_window_hours = self.config.get('prediction_window_hours', 24)
        self.historical_data_days = self.config.get('historical_data_days', 30)
        
        # Initialize data storage
        self._bed_inventory: Dict[str, BedData] = {}
        self._historical_occupancy: List[Dict[str, Any]] = []
        self._last_refresh: Optional[datetime] = None
        
        logger.info("BedAnalyzer initialized successfully")
        
        # Load initial bed data (would connect to hospital system in production)
        asyncio.create_task(self._initialize_bed_data())
    
    async def _initialize_bed_data(self):
        """
        Initialize bed data from hospital information system.
        In production, this would connect to the hospital's bed management system.
        """
        logger.info("Initializing bed data...")
        
        # Simulate loading bed data
        # In production, this would be an API call to the hospital's bed management system
        
        # Create sample bed data for demonstration
        departments = ["Emergency", "ICU", "Med/Surg", "Cardiology", "Pediatrics"]
        bed_types = [
            BedType.EMERGENCY, BedType.ICU, BedType.MEDICAL_SURGICAL,
            BedType.CCU, BedType.PEDIATRIC
        ]
        
        bed_id_counter = 1
        for dept in departments:
            for room_num in range(1, 21):  # 20 beds per department
                bed_id = f"BED{bed_id_counter:05d}"
                bed_type = bed_types[bed_id_counter % len(bed_types)]
                
                # Randomly assign status for demonstration
                status_choices = [
                    BedStatus.AVAILABLE,
                    BedStatus.OCCUPIED,
                    BedStatus.CLEANING
                ]
                import random
                status = random.choice(status_choices)
                
                self._bed_inventory[bed_id] = BedData(
                    bed_id=bed_id,
                    bed_type=bed_type,
                    department=dept,
                    room_number=f"{dept[:3]}{room_num:03d}",
                    status=status,
                    current_patient_id=f"P{random.randint(10000, 99999)}" if status == BedStatus.OCCUPIED else None,
                    estimated_discharge_time=datetime.now() + timedelta(hours=random.randint(2, 48)) if status == BedStatus.OCCUPIED else None
                )
                bed_id_counter += 1
        
        self._last_refresh = datetime.now()
        logger.info(f"Initialized {len(self._bed_inventory)} beds")
    
    async def analyze_current_occupancy(
        self,
        bed_type: Optional[BedType] = None,
        department: Optional[str] = None
    ) -> BedOccupancyAnalysis:
        """
        Analyze current bed occupancy with optional filtering.
        
        Args:
            bed_type: Optional filter for specific bed type
            department: Optional filter for specific department
            
        Returns:
            BedOccupancyAnalysis object with detailed occupancy data
        """
        logger.info(f"Analyzing bed occupancy - bed_type: {bed_type}, department: {department}")
        
        # Refresh data if needed
        await self._refresh_if_needed()
        
        # Filter beds based on criteria
        filtered_beds = self._filter_beds(bed_type, department)
        
        # Calculate basic statistics
        total_beds = len(filtered_beds)
        occupied_beds = sum(1 for bed in filtered_beds.values() if bed.status == BedStatus.OCCUPIED)
        available_beds = sum(1 for bed in filtered_beds.values() if bed.status == BedStatus.AVAILABLE)
        
        occupancy_rate = occupied_beds / total_beds if total_beds > 0 else 0
        
        # Calculate department breakdown
        dept_breakdown = self._calculate_department_breakdown(filtered_beds)
        
        # Calculate bed type breakdown
        bed_type_breakdown = self._calculate_bed_type_breakdown(filtered_beds)
        
        # Get predictions
        predictions = await self.predict_occupancy(hours_ahead=24, bed_type=bed_type)
        
        # Get historical trend
        historical_trend = await self._get_historical_trend(bed_type, department, hours=24)
        
        analysis = BedOccupancyAnalysis(
            total_beds=total_beds,
            occupied_beds=occupied_beds,
            available_beds=available_beds,
            occupancy_rate=occupancy_rate,
            department_breakdown=dept_breakdown,
            bed_type_breakdown=bed_type_breakdown,
            predicted_occupancy_6h=predictions.get("6h", {}).get("occupancy_rate", occupancy_rate),
            predicted_occupancy_12h=predictions.get("12h", {}).get("occupancy_rate", occupancy_rate),
            predicted_occupancy_24h=predictions.get("24h", {}).get("occupancy_rate", occupancy_rate),
            occupancy_trend_24h=historical_trend,
            analysis_time=datetime.now()
        )
        
        # Store for historical analysis
        self._historical_occupancy.append({
            "timestamp": datetime.now(),
            "analysis": analysis.dict()
        })
        
        # Trim historical data
        cutoff = datetime.now() - timedelta(days=self.historical_data_days)
        self._historical_occupancy = [
            entry for entry in self._historical_occupancy
            if entry["timestamp"] > cutoff
        ]
        
        return analysis
    
    async def predict_occupancy(
        self,
        hours_ahead: int,
        bed_type: Optional[BedType] = None
    ) -> Dict[str, Any]:
        """
        Predict bed occupancy for future time periods.
        
        Args:
            hours_ahead: Number of hours to predict ahead
            bed_type: Optional filter for specific bed type
            
        Returns:
            Dictionary with occupancy predictions
        """
        logger.info(f"Predicting occupancy for next {hours_ahead} hours")
        
        # In production, this would use ML models trained on historical data
        # For now, use a simple time series model based on historical patterns
        
        predictions = {}
        
        # Get historical patterns
        historical_pattern = await self._get_historical_pattern(bed_type)
        
        # Current occupancy
        current_analysis = await self.analyze_current_occupancy(bed_type)
        current_occupied = current_analysis.occupied_beds
        
        # Generate predictions for different time horizons
        time_points = [6, 12, 24]
        for hours in time_points:
            if hours <= hours_ahead:
                # Simple prediction using historical patterns and current occupancy
                pattern_factor = historical_pattern.get(str(hours), 1.0)
                predicted_occupied = current_occupied * pattern_factor
                
                # Add some randomness and constraints
                import random
                predicted_occupied += random.uniform(-0.05, 0.05) * current_occupied
                predicted_occupied = max(0, min(current_analysis.total_beds, predicted_occupied))
                
                predictions[f"{hours}h"] = {
                    "occupied_beds": int(predicted_occupied),
                    "occupancy_rate": predicted_occupied / current_analysis.total_beds,
                    "confidence": 0.7 if hours <= 12 else 0.5,  # Lower confidence for longer horizons
                    "range_low": int(predicted_occupied * 0.9),
                    "range_high": int(predicted_occupied * 1.1)
                }
        
        return predictions
    
    async def _get_historical_pattern(self, bed_type: Optional[BedType] = None) -> Dict[str, float]:
        """
        Get historical occupancy patterns from stored data.
        
        Args:
            bed_type: Optional filter for specific bed type
            
        Returns:
            Dictionary with pattern factors for different time horizons
        """
        if len(self._historical_occupancy) < 24:  # Need at least 24 hours of data
            # Return default patterns
            return {
                "6": 1.02,  # Slight increase in next 6 hours
                "12": 1.05,  # Moderate increase in next 12 hours
                "24": 1.08   # Larger increase in next 24 hours
            }
        
        # Analyze historical data to find patterns
        # This would calculate average changes at different times of day
        # For now, return default patterns
        return {
            "6": 1.02,
            "12": 1.05,
            "24": 1.08
        }
    
    def _filter_beds(
        self,
        bed_type: Optional[BedType],
        department: Optional[str]
    ) -> Dict[str, BedData]:
        """
        Filter beds based on criteria.
        
        Args:
            bed_type: Optional bed type filter
            department: Optional department filter
            
        Returns:
            Filtered dictionary of beds
        """
        filtered = {}
        
        for bed_id, bed in self._bed_inventory.items():
            if bed_type and bed.bed_type != bed_type:
                continue
            if department and bed.department.lower() != department.lower():
                continue
            filtered[bed_id] = bed
        
        return filtered
    
    def _calculate_department_breakdown(
        self,
        beds: Dict[str, BedData]
    ) -> Dict[str, Dict[str, int]]:
        """
        Calculate occupancy breakdown by department.
        
        Args:
            beds: Dictionary of beds to analyze
            
        Returns:
            Department breakdown dictionary
        """
        breakdown = {}
        
        for bed in beds.values():
            dept = bed.department
            if dept not in breakdown:
                breakdown[dept] = {
                    "total": 0,
                    "occupied": 0,
                    "available": 0,
                    "cleaning": 0
                }
            
            breakdown[dept]["total"] += 1
            
            if bed.status == BedStatus.OCCUPIED:
                breakdown[dept]["occupied"] += 1
            elif bed.status == BedStatus.AVAILABLE:
                breakdown[dept]["available"] += 1
            elif bed.status == BedStatus.CLEANING:
                breakdown[dept]["cleaning"] += 1
        
        # Add occupancy rates
        for dept in breakdown:
            total = breakdown[dept]["total"]
            if total > 0:
                breakdown[dept]["occupancy_rate"] = breakdown[dept]["occupied"] / total
        
        return breakdown
    
    def _calculate_bed_type_breakdown(
        self,
        beds: Dict[str, BedData]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate occupancy breakdown by bed type.
        
        Args:
            beds: Dictionary of beds to analyze
            
        Returns:
            Bed type breakdown dictionary
        """
        breakdown = {}
        
        for bed in beds.values():
            bed_type = bed.bed_type.value
            if bed_type not in breakdown:
                breakdown[bed_type] = {
                    "total": 0,
                    "occupied": 0,
                    "available": 0,
                    "cleaning": 0
                }
            
            breakdown[bed_type]["total"] += 1
            
            if bed.status == BedStatus.OCCUPIED:
                breakdown[bed_type]["occupied"] += 1
            elif bed.status == BedStatus.AVAILABLE:
                breakdown[bed_type]["available"] += 1
            elif bed.status == BedStatus.CLEANING:
                breakdown[bed_type]["cleaning"] += 1
        
        # Add occupancy rates
        for bed_type in breakdown:
            total = breakdown[bed_type]["total"]
            if total > 0:
                breakdown[bed_type]["occupancy_rate"] = breakdown[bed_type]["occupied"] / total
        
        return breakdown
    
    async def _get_historical_trend(
        self,
        bed_type: Optional[BedType],
        department: Optional[str],
        hours: int
    ) -> List[float]:
        """
        Get historical occupancy trend for specified period.
        
        Args:
            bed_type: Optional bed type filter
            department: Optional department filter
            hours: Number of hours to look back
            
        Returns:
            List of occupancy rates at hourly intervals
        """
        trend = []
        
        # Filter historical data
        relevant_data = [
            entry for entry in self._historical_occupancy
            if entry["timestamp"] > datetime.now() - timedelta(hours=hours)
        ]
        
        # Sort by timestamp
        relevant_data.sort(key=lambda x: x["timestamp"])
        
        # Extract occupancy rates
        for entry in relevant_data:
            analysis = entry["analysis"]
            
            # Apply filters if needed
            if bed_type or department:
                # Would need to recalculate for filtered data
                # For now, use overall rate
                trend.append(analysis["occupancy_rate"])
            else:
                trend.append(analysis["occupancy_rate"])
        
        return trend
    
    async def get_available_beds(
        self,
        bed_type: Optional[BedType] = None,
        department: Optional[str] = None,
        include_cleaning: bool = False
    ) -> List[BedData]:
        """
        Get list of available beds matching criteria.
        
        Args:
            bed_type: Optional bed type filter
            department: Optional department filter
            include_cleaning: Whether to include beds being cleaned
            
        Returns:
            List of available beds
        """
        await self._refresh_if_needed()
        
        available_beds = []
        
        for bed in self._bed_inventory.values():
            if bed_type and bed.bed_type != bed_type:
                continue
            if department and bed.department.lower() != department.lower():
                continue
            
            if bed.status == BedStatus.AVAILABLE:
                available_beds.append(bed)
            elif include_cleaning and bed.status == BedStatus.CLEANING:
                # Calculate when bed will be available
                cleaning_complete = bed.last_updated + timedelta(minutes=bed.cleaning_estimate_minutes)
                if cleaning_complete <= datetime.now():
                    bed.status = BedStatus.AVAILABLE
                    available_beds.append(bed)
        
        return available_beds
    
    async def find_optimal_bed(
        self,
        required_type: BedType,
        preferred_department: Optional[str] = None,
        required_equipment: Optional[List[str]] = None
    ) -> Optional[BedData]:
        """
        Find the optimal bed for a patient based on requirements.
        
        Args:
            required_type: Required bed type
            preferred_department: Preferred department
            required_equipment: List of required equipment
            
        Returns:
            Optimal bed if found, None otherwise
        """
        available_beds = await self.get_available_beds(
            bed_type=required_type,
            department=preferred_department
        )
        
        if not available_beds:
            return None
        
        # Score beds based on criteria
        best_bed = None
        best_score = -1
        
        for bed in available_beds:
            score = 100  # Base score
            
            # Prefer beds in preferred department
            if preferred_department and bed.department == preferred_department:
                score += 50
            
            # Check equipment availability
            if required_equipment:
                equipment_match = sum(1 for eq in required_equipment if eq in bed.equipment_available)
                score += equipment_match * 10
            
            # Prefer beds that have been available longer
            time_available = (datetime.now() - bed.last_updated).total_seconds() / 3600
            score += min(20, time_available)  # Max 20 points for time available
            
            if score > best_score:
                best_score = score
                best_bed = bed
        
        return best_bed
    
    async def reserve_bed(self, bed_id: str, patient_id: str) -> bool:
        """
        Reserve a bed for a patient.
        
        Args:
            bed_id: ID of bed to reserve
            patient_id: ID of patient
            
        Returns:
            True if reservation successful, False otherwise
        """
        if bed_id not in self._bed_inventory:
            logger.error(f"Bed {bed_id} not found")
            return False
        
        bed = self._bed_inventory[bed_id]
        
        if bed.status != BedStatus.AVAILABLE:
            logger.warning(f"Bed {bed_id} is not available (status: {bed.status})")
            return False
        
        bed.status = BedStatus.OCCUPIED
        bed.current_patient_id = patient_id
        bed.last_updated = datetime.now()
        
        logger.info(f"Bed {bed_id} reserved for patient {patient_id}")
        return True
    
    async def release_bed(self, bed_id: str, start_cleaning: bool = True) -> bool:
        """
        Release a bed (patient discharged).
        
        Args:
            bed_id: ID of bed to release
            start_cleaning: Whether to start cleaning process
            
        Returns:
            True if release successful, False otherwise
        """
        if bed_id not in self._bed_inventory:
            logger.error(f"Bed {bed_id} not found")
            return False
        
        bed = self._bed_inventory[bed_id]
        
        if bed.status != BedStatus.OCCUPIED:
            logger.warning(f"Bed {bed_id} is not occupied")
            return False
        
        bed.current_patient_id = None
        bed.estimated_discharge_time = None
        
        if start_cleaning:
            bed.status = BedStatus.CLEANING
            logger.info(f"Bed {bed_id} released, cleaning started")
        else:
            bed.status = BedStatus.AVAILABLE
            logger.info(f"Bed {bed_id} released and available")
        
        bed.last_updated = datetime.now()
        return True
    
    async def update_bed_status(self, bed_id: str, new_status: BedStatus) -> bool:
        """
        Update the status of a bed.
        
        Args:
            bed_id: ID of bed to update
            new_status: New status
            
        Returns:
            True if update successful, False otherwise
        """
        if bed_id not in self._bed_inventory:
            logger.error(f"Bed {bed_id} not found")
            return False
        
        bed = self._bed_inventory[bed_id]
        bed.status = new_status
        bed.last_updated = datetime.now()
        
        logger.info(f"Bed {bed_id} status updated to {new_status.value}")
        return True
    
    async def get_occupancy_forecast(
        self,
        hours: int = 48,
        interval_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Generate detailed occupancy forecast at regular intervals.
        
        Args:
            hours: Forecast horizon in hours
            interval_minutes: Interval between forecasts
            
        Returns:
            List of forecast points
        """
        forecasts = []
        
        for hour in range(0, hours + 1, interval_minutes // 60):
            if hour == 0:
                # Current occupancy
                analysis = await self.analyze_current_occupancy()
                forecast = {
                    "timestamp": datetime.now(),
                    "hour": hour,
                    "occupied": analysis.occupied_beds,
                    "occupancy_rate": analysis.occupancy_rate,
                    "type": "current"
                }
            else:
                # Predicted occupancy
                predictions = await self.predict_occupancy(hours_ahead=hour)
                
                # Find the closest prediction
                if hour <= 6:
                    pred = predictions.get("6h", {})
                elif hour <= 12:
                    pred = predictions.get("12h", {})
                else:
                    pred = predictions.get("24h", {})
                
                forecast = {
                    "timestamp": datetime.now() + timedelta(hours=hour),
                    "hour": hour,
                    "occupied": pred.get("occupied_beds", 0),
                    "occupancy_rate": pred.get("occupancy_rate", 0),
                    "confidence": pred.get("confidence", 0.5),
                    "range_low": pred.get("range_low", 0),
                    "range_high": pred.get("range_high", 0),
                    "type": "predicted"
                }
            
            forecasts.append(forecast)
        
        return forecasts
    
    async def identify_peak_occupancy_times(self) -> Dict[str, Any]:
        """
        Identify peak occupancy times based on historical data.
        
        Returns:
            Dictionary with peak occupancy analysis
        """
        if len(self._historical_occupancy) < 24 * 7:  # Less than a week of data
            return {
                "peak_hour": "14:00",
                "peak_occupancy": 0.85,
                "lowest_hour": "04:00",
                "lowest_occupancy": 0.65,
                "average_by_hour": {}
            }
        
        # Group by hour of day
        hourly_occupancy = {}
        
        for entry in self._historical_occupancy:
            hour = entry["timestamp"].hour
            if hour not in hourly_occupancy:
                hourly_occupancy[hour] = []
            
            hourly_occupancy[hour].append(entry["analysis"]["occupancy_rate"])
        
        # Calculate averages
        avg_by_hour = {}
        for hour in range(24):
            if hour in hourly_occupancy and hourly_occupancy[hour]:
                avg_by_hour[hour] = sum(hourly_occupancy[hour]) / len(hourly_occupancy[hour])
            else:
                avg_by_hour[hour] = 0.75  # Default
        
        # Find peak and lowest hours
        peak_hour = max(avg_by_hour.items(), key=lambda x: x[1])
        lowest_hour = min(avg_by_hour.items(), key=lambda x: x[1])
        
        return {
            "peak_hour": f"{peak_hour[0]:02d}:00",
            "peak_occupancy": peak_hour[1],
            "lowest_hour": f"{lowest_hour[0]:02d}:00",
            "lowest_occupancy": lowest_hour[1],
            "average_by_hour": avg_by_hour
        }
    
    async def _refresh_if_needed(self):
        """
        Refresh bed data if cache is stale.
        """
        if not self._last_refresh:
            await self._initialize_bed_data()
            return
        
        time_since_refresh = (datetime.now() - self._last_refresh).total_seconds()
        
        if time_since_refresh > self.bed_data_refresh_interval:
            logger.info("Refreshing bed data...")
            await self._initialize_bed_data()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive bed metrics for dashboards.
        
        Returns:
            Dictionary with bed metrics
        """
        analysis = await self.analyze_current_occupancy()
        
        return {
            "total_beds": analysis.total_beds,
            "occupied_beds": analysis.occupied_beds,
            "available_beds": analysis.available_beds,
            "occupancy_rate": analysis.occupancy_rate,
            "icu_beds_total": sum(1 for bed in self._bed_inventory.values() 
                                  if bed.bed_type in [BedType.ICU, BedType.MICU, BedType.SICU]),
            "icu_beds_occupied": sum(1 for bed in self._bed_inventory.values()
                                     if bed.bed_type in [BedType.ICU, BedType.MICU, BedType.SICU]
                                     and bed.status == BedStatus.OCCUPIED),
            "icu_occupancy_rate": self._calculate_icu_occupancy(),
            "department_breakdown": analysis.department_breakdown,
            "bed_type_breakdown": analysis.bed_type_breakdown
        }
    
    def _calculate_icu_occupancy(self) -> float:
        """
        Calculate ICU occupancy rate.
        
        Returns:
            ICU occupancy rate
        """
        icu_beds = [bed for bed in self._bed_inventory.values()
                   if bed.bed_type in [BedType.ICU, BedType.MICU, BedType.SICU, BedType.CCU]]
        
        if not icu_beds:
            return 0.0
        
        occupied = sum(1 for bed in icu_beds if bed.status == BedStatus.OCCUPIED)
        return occupied / len(icu_beds)