"""
Resource Model for AegisMedBot

This module defines hospital resource models including:
- Beds and room inventory
- Medical equipment tracking
- Operating room schedules
- Resource utilization metrics

These models support operational analytics and capacity planning.
"""

from sqlalchemy import (
    Column, String, Integer, DateTime, Boolean, JSON, 
    Float, ForeignKey, Text, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from .admission import Base

class ResourceType(enum.Enum):
    """Types of hospital resources tracked"""
    BED = "bed"                      # Patient bed
    ICU_BED = "icu_bed"              # Intensive care bed
    VENTILATOR = "ventilator"        # Mechanical ventilator
    INFUSION_PUMP = "infusion_pump"  # IV infusion pump
    MONITOR = "monitor"              # Patient monitor
    SURGICAL_ROOM = "surgical_room"  # Operating room
    IMAGING_EQUIPMENT = "imaging_equipment"  # MRI, CT, X-ray
    LAB_EQUIPMENT = "lab_equipment"  # Laboratory analyzers

class ResourceStatus(enum.Enum):
    """Current operational status of resources"""
    AVAILABLE = "available"          # Ready for use
    IN_USE = "in_use"                # Currently being used
    MAINTENANCE = "maintenance"      # Undergoing maintenance
    CLEANING = "cleaning"            # Being cleaned
    RESERVED = "reserved"            # Reserved for scheduled use
    OUT_OF_SERVICE = "out_of_service"  # Not usable
    DECONTAMINATION = "decontamination"  # Being decontaminated

class BedType(enum.Enum):
    """Specialized bed types for patient care"""
    MEDICAL_SURGICAL = "medical_surgical"  # General med-surg bed
    ICU = "icu"                            # Intensive care
    CCU = "ccu"                            # Cardiac care
    PEDIATRIC = "pediatric"                # Children's bed
    NICU = "nicu"                          # Neonatal ICU
    BARIATRIC = "bariatric"                # Weight capacity bed
    ISOLATION = "isolation"                # Negative pressure isolation

class Bed(Base):
    """
    Bed Resource Model
    
    Tracks all patient beds in the hospital including:
    - Bed location and type
    - Current occupancy status
    - Patient assignment
    - Maintenance and cleaning status
    
    Critical for real-time bed management and patient flow optimization.
    """
    
    __tablename__ = "beds"
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Bed identification
    bed_number = Column(
        String(20), 
        nullable=False,
        comment="Unique bed identifier within hospital"
    )
    
    room_number = Column(
        String(20), 
        nullable=False,
        index=True,
        comment="Room containing this bed"
    )
    
    unit = Column(
        String(100), 
        nullable=False,
        index=True,
        comment="Nursing unit or department"
    )
    
    floor = Column(
        Integer, 
        nullable=False,
        comment="Floor number in hospital"
    )
    
    # Bed specifications
    bed_type = Column(
        String(50), 
        nullable=False,
        index=True,
        comment="Type of bed - see BedType enum"
    )
    
    is_private = Column(
        Boolean, 
        default=True,
        comment="Private room or shared"
    )
    
    has_negative_pressure = Column(
        Boolean, 
        default=False,
        comment="Isolation capability for airborne diseases"
    )
    
    has_bariatric_capacity = Column(
        Boolean, 
        default=False,
        comment="Weight capacity for bariatric patients"
    )
    
    # Equipment available at bed
    has_ventilator = Column(Boolean, default=False)
    has_cardiac_monitor = Column(Boolean, default=False)
    has_oxygen = Column(Boolean, default=True)
    has_suction = Column(Boolean, default=True)
    
    # Current status
    status = Column(
        String(50), 
        nullable=False,
        default=ResourceStatus.AVAILABLE.value,
        index=True,
        comment="Current operational status"
    )
    
    is_occupied = Column(
        Boolean, 
        default=False,
        index=True,
        comment="Currently has patient assigned"
    )
    
    current_patient_id = Column(
        String(36), 
        ForeignKey("patients.id", ondelete="SET NULL"),
        nullable=True,
        comment="Patient currently in this bed, if any"
    )
    
    current_admission_id = Column(
        String(36), 
        ForeignKey("admissions.id", ondelete="SET NULL"),
        nullable=True,
        comment="Current admission occupying this bed"
    )
    
    # Status tracking
    last_cleaned = Column(
        DateTime, 
        nullable=True,
        comment="Last time bed was terminally cleaned"
    )
    
    last_maintenance = Column(
        DateTime, 
        nullable=True,
        comment="Last preventive maintenance date"
    )
    
    expected_cleaning_duration_minutes = Column(
        Integer, 
        default=30,
        comment="Average time to clean between patients"
    )
    
    # Utilization metrics
    total_occupancy_hours = Column(
        Float, 
        default=0,
        comment="Total hours occupied since tracking started"
    )
    
    times_used = Column(
        Integer, 
        default=0,
        comment="Number of times bed has been used"
    )
    
    # Metadata
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    
    def is_available(self):
        """
        Check if bed is available for new patient assignment
        
        Returns:
            bool: True if bed can accept new patient
        """
        return (self.status == ResourceStatus.AVAILABLE.value and 
                not self.is_occupied)
    
    def assign_patient(self, patient_id, admission_id):
        """
        Assign a patient to this bed
        
        Args:
            patient_id: Patient identifier
            admission_id: Admission record identifier
            
        Returns:
            bool: True if assignment successful
        """
        if not self.is_available():
            return False
            
        self.current_patient_id = patient_id
        self.current_admission_id = admission_id
        self.is_occupied = True
        self.status = ResourceStatus.IN_USE.value
        self.updated_at = datetime.now()
        return True
    
    def discharge_patient(self):
        """
        Remove patient from bed and prepare for cleaning
        
        Updates utilization metrics and marks bed for cleaning.
        """
        if self.is_occupied:
            # Calculate occupancy duration for this stay
            # Would need start time tracking for accurate calculation
            self.total_occupancy_hours += 1  # Simplified
            self.times_used += 1
            
            # Clear patient assignment
            self.current_patient_id = None
            self.current_admission_id = None
            self.is_occupied = False
            
            # Mark for cleaning
            self.status = ResourceStatus.CLEANING.value
            self.updated_at = datetime.now()
    
    def mark_cleaned(self):
        """
        Mark bed as cleaned and available for use
        """
        if self.status == ResourceStatus.CLEANING.value:
            self.status = ResourceStatus.AVAILABLE.value
            self.last_cleaned = datetime.now()
            self.updated_at = datetime.now()
    
    def get_utilization_rate(self, time_period_hours=24):
        """
        Calculate bed utilization rate over time period
        
        Args:
            time_period_hours: Hours to calculate utilization for
            
        Returns:
            float: Utilization percentage 0-100
        """
        # Simplified - would need occupancy log for accurate calculation
        if self.total_occupancy_hours > 0:
            return min((self.total_occupancy_hours / time_period_hours) * 100, 100)
        return 0
    
    def to_dict(self):
        """
        Convert bed to dictionary for API responses
        """
        return {
            "id": self.id,
            "bed_number": self.bed_number,
            "room_number": self.room_number,
            "unit": self.unit,
            "floor": self.floor,
            "bed_type": self.bed_type,
            "is_private": self.is_private,
            "has_negative_pressure": self.has_negative_pressure,
            "status": self.status,
            "is_occupied": self.is_occupied,
            "is_available": self.is_available(),
            "current_patient_id": self.current_patient_id,
            "last_cleaned": self.last_cleaned.isoformat() if self.last_cleaned else None,
            "utilization_rate": self.get_utilization_rate(),
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class MedicalEquipment(Base):
    """
    Medical Equipment Model
    
    Tracks movable medical equipment including:
    - Ventilators
    - Infusion pumps
    - Patient monitors
    - Mobile imaging units
    
    Critical for resource allocation and maintenance tracking.
    """
    
    __tablename__ = "medical_equipment"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Equipment identification
    equipment_id = Column(
        String(50), 
        unique=True, 
        nullable=False,
        index=True,
        comment="Asset tracking number"
    )
    
    equipment_type = Column(
        String(50), 
        nullable=False,
        index=True,
        comment="Type of equipment - see ResourceType enum"
    )
    
    manufacturer = Column(String(100), nullable=True)
    model = Column(String(100), nullable=True)
    serial_number = Column(String(100), unique=True, nullable=True)
    
    # Location tracking
    current_location = Column(
        String(100), 
        nullable=True,
        comment="Current location in hospital"
    )
    
    department = Column(
        String(50), 
        nullable=True,
        index=True,
        comment="Assigned department"
    )
    
    # Status
    status = Column(
        String(50), 
        nullable=False,
        default=ResourceStatus.AVAILABLE.value,
        index=True
    )
    
    is_assigned = Column(
        Boolean, 
        default=False,
        comment="Currently assigned to patient or room"
    )
    
    assigned_to_patient_id = Column(
        String(36), 
        ForeignKey("patients.id", ondelete="SET NULL"),
        nullable=True
    )
    
    assigned_to_room = Column(
        String(20), 
        nullable=True,
        comment="Room equipment is assigned to"
    )
    
    # Maintenance
    last_calibration = Column(DateTime, nullable=True)
    next_calibration_due = Column(DateTime, nullable=True)
    last_maintenance = Column(DateTime, nullable=True)
    
    maintenance_frequency_days = Column(
        Integer, 
        default=180,
        comment="Days between scheduled maintenance"
    )
    
    # Utilization
    total_usage_hours = Column(Float, default=0)
    last_used = Column(DateTime, nullable=True)
    
    # Metadata
    purchase_date = Column(DateTime, nullable=True)
    purchase_cost = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    
    def is_available(self):
        """
        Check if equipment is available for use
        
        Returns:
            bool: True if equipment can be used
        """
        return (self.status == ResourceStatus.AVAILABLE.value and 
                not self.is_assigned)
    
    def assign_to_patient(self, patient_id, location):
        """
        Assign equipment to a patient
        
        Args:
            patient_id: Patient identifier
            location: Room or location where equipment is placed
            
        Returns:
            bool: True if assignment successful
        """
        if not self.is_available():
            return False
            
        self.is_assigned = True
        self.assigned_to_patient_id = patient_id
        self.assigned_to_room = location
        self.current_location = location
        self.status = ResourceStatus.IN_USE.value
        self.updated_at = datetime.now()
        return True
    
    def return_to_pool(self):
        """
        Return equipment to available pool
        """
        self.is_assigned = False
        self.assigned_to_patient_id = None
        self.status = ResourceStatus.AVAILABLE.value
        self.updated_at = datetime.now()
    
    def needs_maintenance(self):
        """
        Check if equipment needs maintenance
        
        Returns:
            bool: True if maintenance is due
        """
        if self.next_calibration_due:
            return datetime.now() >= self.next_calibration_due
        return False
    
    def to_dict(self):
        """
        Convert equipment to dictionary for API responses
        """
        return {
            "id": self.id,
            "equipment_id": self.equipment_id,
            "equipment_type": self.equipment_type,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "current_location": self.current_location,
            "status": self.status,
            "is_available": self.is_available(),
            "needs_maintenance": self.needs_maintenance(),
            "total_usage_hours": self.total_usage_hours,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class OperatingRoom(Base):
    """
    Operating Room Model
    
    Tracks surgical suite availability and scheduling.
    Critical for surgical services planning and resource allocation.
    """
    
    __tablename__ = "operating_rooms"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Room identification
    room_number = Column(
        String(10), 
        unique=True, 
        nullable=False,
        index=True,
        comment="OR room number"
    )
    
    suite_name = Column(
        String(100), 
        nullable=True,
        comment="Suite name if applicable"
    )
    
    # Room specifications
    room_type = Column(
        String(50), 
        nullable=False,
        comment="General, Cardiac, Neuro, Ortho, Hybrid, etc."
    )
    
    is_hybrid_or = Column(
        Boolean, 
        default=False,
        comment="Has imaging capabilities"
    )
    
    is_minimally_invasive = Column(
        Boolean, 
        default=False,
        comment="Equipped for laparoscopic/robotic surgery"
    )
    
    has_robotics = Column(Boolean, default=False)
    has_mri = Column(Boolean, default=False)
    has_ct = Column(Boolean, default=False)
    
    # Capacity
    max_staff_capacity = Column(Integer, default=10)
    
    # Current status
    status = Column(
        String(50), 
        nullable=False,
        default=ResourceStatus.AVAILABLE.value,
        index=True
    )
    
    current_procedure_id = Column(
        String(36), 
        nullable=True,
        comment="Currently ongoing procedure ID"
    )
    
    next_scheduled_procedure = Column(
        DateTime, 
        nullable=True,
        comment="Time of next scheduled surgery"
    )
    
    # Utilization
    total_surgical_minutes = Column(Integer, default=0)
    total_procedures = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    def is_available(self, start_time=None, duration_minutes=None):
        """
        Check if OR is available at specified time
        
        Args:
            start_time: Desired start time, defaults to now
            duration_minutes: Expected procedure duration
            
        Returns:
            bool: True if OR is available
        """
        if self.status != ResourceStatus.AVAILABLE.value:
            return False
            
        # In production, would check against scheduled procedures
        return True
    
    def schedule_procedure(self, procedure_id, start_time, duration_minutes):
        """
        Schedule a procedure in this OR
        
        Args:
            procedure_id: Procedure identifier
            start_time: Scheduled start time
            duration_minutes: Expected duration
            
        Returns:
            bool: True if scheduling successful
        """
        if not self.is_available(start_time, duration_minutes):
            return False
            
        self.status = ResourceStatus.RESERVED.value
        self.next_scheduled_procedure = start_time
        self.updated_at = datetime.now()
        return True
    
    def to_dict(self):
        """
        Convert OR to dictionary for API responses
        """
        return {
            "id": self.id,
            "room_number": self.room_number,
            "room_type": self.room_type,
            "has_robotics": self.has_robotics,
            "has_mri": self.has_mri,
            "status": self.status,
            "is_available": self.is_available(),
            "next_scheduled_procedure": self.next_scheduled_procedure.isoformat() if self.next_scheduled_procedure else None,
            "total_procedures": self.total_procedures,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }