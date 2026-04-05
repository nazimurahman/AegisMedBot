"""
Admission Model for AegisMedBot Hospital Intelligence Platform

This module defines the database schema for patient admissions, including
all clinical and administrative data related to hospital stays.
It tracks patient movement, bed assignments, diagnoses, and treatment progress.
"""

from sqlalchemy import (
    Column, String, Integer, DateTime, Date, JSON, Boolean, 
    Float, ForeignKey, Text, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid
import enum

# Create base class for all SQLAlchemy ORM models
# This base class maintains the connection between Python classes and database tables
Base = declarative_base()

# Define enumeration for admission types to ensure data consistency
# Using enum prevents invalid admission types from being stored in database
class AdmissionType(enum.Enum):
    """Standardized admission types used across hospital systems"""
    EMERGENCY = "emergency"      # Patient admitted through emergency department
    ELECTIVE = "elective"         # Scheduled admission for planned procedure
    URGENT = "urgent"             # Urgent but not emergency admission
    TRANSFER = "transfer"         # Transferred from another facility
    OBSERVATION = "observation"   # Under observation not formally admitted

class AdmissionStatus(enum.Enum):
    """Current status of patient admission"""
    ACTIVE = "active"                 # Currently admitted
    DISCHARGED = "discharged"         # Successfully discharged
    TRANSFERRED = "transferred"       # Transferred to another facility
    DECEASED = "deceased"             # Patient passed away during admission
    CANCELLED = "cancelled"           # Admission was cancelled
    PENDING = "pending"               # Awaiting bed assignment

class DischargeDisposition(enum.Enum):
    """Where patient goes after discharge - critical for care coordination"""
    HOME = "home"                     # Discharged to home
    HOME_HEALTH = "home_health"       # Home with health services
    REHAB = "rehab"                   # Rehabilitation facility
    SKILLED_NURSING = "skilled_nursing"  # Skilled nursing facility
    LONG_TERM_CARE = "long_term_care"    # Long term acute care
    HOSPICE = "hospice"               # Hospice care
    AGAINST_MEDICAL_ADVICE = "against_medical_advice"  # Left AMA
    EXPIRED = "expired"               # Patient expired
    TRANSFER = "transfer"             # Transferred to another hospital

class Admission(Base):
    """
    Patient Admission Model
    
    This table stores comprehensive information about each patient's hospital stay.
    It serves as the central hub for patient tracking, resource allocation,
    and clinical documentation.
    
    Key Features:
    - Tracks patient location within hospital (room, bed)
    - Records clinical team assignments
    - Maintains diagnosis and treatment information
    - Supports length of stay calculations
    - Enables resource utilization analysis
    """
    
    __tablename__ = "admissions"
    # Explicit table name for database - follows naming convention
    
    # Primary Key - Unique identifier for each admission record
    # Using UUID instead of auto-increment integer for:
    # 1. Distributed system compatibility
    # 2. Security through non-sequential IDs
    # 3. Easy merging of data from multiple sources
    id = Column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Foreign key linking to patient table
    # This establishes the relationship between admissions and patients
    # Each admission belongs to exactly one patient
    patient_id = Column(
        String(36), 
        ForeignKey("patients.id", ondelete="RESTRICT"),  # RESTRICT prevents orphaned records
        nullable=False,  # Every admission must have a patient
        index=True       # Index for fast patient history queries
    )
    
    # Temporal Fields - Track admission timeline
    # These are critical for length of stay calculations and billing
    admission_date = Column(
        DateTime, 
        nullable=False,  # Required field
        default=datetime.now,  # Auto-set to current time if not provided
        index=True       # Indexed for date range queries
    )
    
    discharge_date = Column(
        DateTime, 
        nullable=True,   # Null until patient is discharged
        index=True       # Index for discharge analysis
    )
    
    # Clinical Classification Fields
    admission_type = Column(
        SQLEnum(AdmissionType),  # Uses database enum for data integrity
        nullable=False,
        default=AdmissionType.ELECTIVE
    )
    
    # Department and Location Tracking
    # These fields enable resource allocation and patient flow analysis
    department = Column(
        String(100), 
        nullable=False, 
        index=True,      # Indexed for department-specific queries
        comment="Clinical department like Cardiology, Neurology, etc."
    )
    
    unit = Column(
        String(100), 
        nullable=True,
        comment="Specific unit within department e.g., Cardiac ICU, Step-down"
    )
    
    room_number = Column(
        String(20), 
        nullable=True,
        index=True,      # Indexed for room availability queries
        comment="Room identifier for location tracking"
    )
    
    bed_number = Column(
        String(20), 
        nullable=True,
        comment="Specific bed number within room"
    )
    
    # Clinical Information - JSON fields allow flexible structure
    # JSON is ideal for clinical data that varies per patient
    diagnosis = Column(
        JSON, 
        nullable=True,
        comment="Primary and secondary diagnoses with ICD-10 codes"
    )
    # Example structure:
    # {
    #   "primary": {"code": "I10", "description": "Essential hypertension"},
    #   "secondary": [
    #     {"code": "E11.9", "description": "Type 2 diabetes"},
    #     {"code": "N18.3", "description": "Chronic kidney disease"}
    #   ]
    # }
    
    procedures = Column(
        JSON, 
        nullable=True,
        comment="Procedures performed during admission with CPT codes"
    )
    # Example structure:
    # [
    #   {"code": "93458", "description": "Cardiac catheterization", "date": "2024-01-15"},
    #   {"code": "92980", "description": "Stent placement", "date": "2024-01-15"}
    # ]
    
    # Care Team Assignment
    # These fields track who is responsible for patient care
    admitting_physician = Column(
        String(100), 
        nullable=True,
        comment="Physician who authorized admission"
    )
    
    attending_physician = Column(
        String(100), 
        nullable=False,  # Every admission must have an attending
        index=True,      # Indexed for physician workload analysis
        comment="Primary physician responsible for care"
    )
    
    consulting_physicians = Column(
        JSON, 
        nullable=True,
        default=list,    # Default to empty list
        comment="List of consulting physicians involved"
    )
    
    primary_nurse = Column(
        String(100), 
        nullable=True,
        comment="Primary nurse assigned to patient"
    )
    
    # Status Tracking
    status = Column(
        SQLEnum(AdmissionStatus), 
        nullable=False,
        default=AdmissionStatus.ACTIVE,
        index=True       # Indexed for active admissions queries
    )
    
    discharge_disposition = Column(
        SQLEnum(DischargeDisposition), 
        nullable=True,
        comment="Where patient goes after discharge - critical for outcomes"
    )
    
    # Clinical Scores and Risk Assessment
    # These are used by ML models for risk prediction
    acuity_score = Column(
        Integer, 
        nullable=True,
        comment="Clinical acuity score (1-5, higher = more acute)"
    )
    
    fall_risk_score = Column(
        Integer, 
        nullable=True,
        comment="Fall risk assessment score"
    )
    
    pressure_ulcer_risk = Column(
        String(20), 
        nullable=True,
        comment="Braden score or equivalent for pressure ulcer risk"
    )
    
    # Resource Utilization
    # Tracks resources used during admission for costing and planning
    ventilator_days = Column(
        Integer, 
        default=0,
        comment="Number of days on mechanical ventilation"
    )
    
    icu_days = Column(
        Integer, 
        default=0,
        comment="Number of days in ICU"
    )
    
    total_medications = Column(
        Integer, 
        default=0,
        comment="Count of unique medications administered"
    )
    
    # Financial and Administrative
    financial_class = Column(
        String(50), 
        nullable=True,
        comment="Insurance type: Medicare, Medicaid, Commercial, Self-pay"
    )
    
    authorization_number = Column(
        String(100), 
        nullable=True,
        comment="Insurance authorization reference number"
    )
    
    # Metadata for audit and tracking
    created_at = Column(
        DateTime, 
        default=datetime.now, 
        nullable=False,
        comment="Record creation timestamp"
    )
    
    updated_at = Column(
        DateTime, 
        default=datetime.now, 
        onupdate=datetime.now,  # Automatically updates on record change
        nullable=False,
        comment="Last update timestamp"
    )
    
    created_by = Column(
        String(100), 
        nullable=True,
        comment="User who created this record"
    )
    
    # Soft delete flag - preserves data for audit while hiding from active queries
    is_deleted = Column(
        Boolean, 
        default=False, 
        nullable=False,
        index=True,      # Indexed for filtering deleted records
        comment="Soft delete flag for data retention"
    )
    
    # Relationships - Define how this table connects to others
    # These enable SQLAlchemy to automatically load related data
    patient = relationship(
        "Patient",  # References the Patient model class
        back_populates="admissions",  # Creates bidirectional relationship
        lazy="joined"  # Loads patient data automatically with admission
    )
    
    # One-to-many relationship to vital signs
    # A single admission can have many vital sign measurements
    vital_signs = relationship(
        "VitalSign",
        back_populates="admission",
        lazy="dynamic",  # Returns query object for filtering
        cascade="all, delete-orphan"  # Deletes vital signs when admission deleted
    )
    
    # One-to-many relationship to lab results
    lab_results = relationship(
        "LabResult",
        back_populates="admission",
        lazy="dynamic",
        cascade="all, delete-orphan"
    )
    
    # One-to-many relationship to medications administered
    medications = relationship(
        "Medication",
        back_populates="admission",
        lazy="dynamic",
        cascade="all, delete-orphan"
    )
    
    def calculate_length_of_stay(self):
        """
        Calculate total length of stay in days
        
        This method computes how long the patient has been or was admitted.
        Used for utilization review, billing, and outcome analysis.
        
        Returns:
            float: Length of stay in days, or None if not applicable
            
        Logic:
            - If discharged: uses discharge_date - admission_date
            - If still active: uses current time - admission_date
            - Returns 0 if admission date is in the future (data error)
        """
        if not self.admission_date:
            return None
            
        end_date = self.discharge_date if self.discharge_date else datetime.now()
        
        # Handle future admission date (likely data error)
        if end_date < self.admission_date:
            return 0
            
        # Calculate difference in days with decimal precision
        difference = end_date - self.admission_date
        return difference.total_seconds() / (24 * 3600)  # Convert seconds to days
    
    def calculate_icu_ratio(self):
        """
        Calculate proportion of stay spent in ICU
        
        This metric is important for:
        - Resource planning
        - Severity adjustment in outcomes analysis
        - Cost prediction models
        
        Returns:
            float: Ratio of ICU days to total length of stay (0 to 1)
        """
        total_los = self.calculate_length_of_stay()
        if not total_los or total_los == 0:
            return 0
        return min(self.icu_days / total_los, 1.0)  # Cap at 1.0
    
    def is_currently_admitted(self):
        """
        Check if patient is currently admitted
        
        Returns:
            bool: True if patient is currently in hospital
            
        Used for:
        - Real-time dashboards
        - Bed management systems
        - Alerting for long stays
        """
        return (self.status == AdmissionStatus.ACTIVE and 
                not self.discharge_date)
    
    def is_readmission(self, previous_admissions):
        """
        Determine if this admission is a readmission
        
        Readmissions are a key quality metric tracked by CMS.
        
        Args:
            previous_admissions: List of patient's previous admissions
            
        Returns:
            bool: True if this is a readmission within 30 days
        """
        if not previous_admissions:
            return False
            
        for prev in previous_admissions:
            if prev.id == self.id:  # Skip self
                continue
                
            if prev.discharge_date:
                # Check if readmitted within 30 days
                days_between = (self.admission_date - prev.discharge_date).days
                if 0 <= days_between <= 30:
                    return True
        return False
    
    def to_dict(self):
        """
        Convert admission object to dictionary for API responses
        
        This method ensures data is properly formatted for JSON serialization.
        Handles datetime objects, enums, and JSON fields appropriately.
        
        Returns:
            dict: Dictionary representation of admission data
        """
        result = {
            "id": self.id,
            "patient_id": self.patient_id,
            "admission_date": self.admission_date.isoformat() if self.admission_date else None,
            "discharge_date": self.discharge_date.isoformat() if self.discharge_date else None,
            "admission_type": self.admission_type.value if self.admission_type else None,
            "department": self.department,
            "unit": self.unit,
            "room_number": self.room_number,
            "bed_number": self.bed_number,
            "diagnosis": self.diagnosis,
            "procedures": self.procedures,
            "admitting_physician": self.admitting_physician,
            "attending_physician": self.attending_physician,
            "consulting_physicians": self.consulting_physicians,
            "primary_nurse": self.primary_nurse,
            "status": self.status.value if self.status else None,
            "discharge_disposition": self.discharge_disposition.value if self.discharge_disposition else None,
            "acuity_score": self.acuity_score,
            "fall_risk_score": self.fall_risk_score,
            "ventilator_days": self.ventilator_days,
            "icu_days": self.icu_days,
            "length_of_stay": self.calculate_length_of_stay(),
            "icu_ratio": self.calculate_icu_ratio(),
            "is_currently_admitted": self.is_currently_admitted(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        return result
    
    def __repr__(self):
        """
        String representation for debugging and logging
        
        Returns human-readable string with key admission information
        """
        return f"<Admission(id={self.id}, patient_id={self.patient_id}, " \
               f"department={self.department}, status={self.status})>"