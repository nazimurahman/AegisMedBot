"""
Staff Model for AegisMedBot

This module defines healthcare staff and provider models including:
- Physicians and surgeons
- Nurses and nursing staff
- Administrative personnel
- Staff scheduling and credentials

These models support workforce management and clinical team coordination.
"""

from sqlalchemy import (
    Column, String, Integer, DateTime, Boolean, JSON, 
    ForeignKey, Table, Text
)
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from .admission import Base

class StaffRole(enum.Enum):
    """Healthcare staff roles and positions"""
    PHYSICIAN = "physician"              # Medical doctor
    SURGEON = "surgeon"                  # Surgical specialist
    NURSE = "nurse"                      # Registered nurse
    NURSE_PRACTITIONER = "nurse_practitioner"
    PHYSICIAN_ASSISTANT = "physician_assistant"
    PHARMACIST = "pharmacist"
    THERAPIST = "therapist"              # Physical, occupational, respiratory
    TECHNICIAN = "technician"            # Lab, radiology, surgical tech
    ADMINISTRATOR = "administrator"
    SUPPORT_STAFF = "support_staff"

class Department(enum.Enum):
    """Hospital departments"""
    EMERGENCY = "emergency"
    CARDIOLOGY = "cardiology"
    NEUROLOGY = "neurology"
    ONCOLOGY = "oncology"
    PEDIATRICS = "pediatrics"
    OBSTETRICS = "obstetrics"
    ORTHOPEDICS = "orthopedics"
    SURGERY = "surgery"
    ICU = "intensive_care"
    RADIOLOGY = "radiology"
    LABORATORY = "laboratory"
    PHARMACY = "pharmacy"
    ADMINISTRATION = "administration"

class ShiftType(enum.Enum):
    """Work shift types"""
    DAY = "day"          # 7 AM - 3 PM
    EVENING = "evening"  # 3 PM - 11 PM
    NIGHT = "night"      # 11 PM - 7 AM
    ON_CALL = "on_call"  # Available if needed

class Staff(Base):
    """
    Healthcare Staff Model
    
    Tracks all hospital personnel, their roles, credentials,
    and scheduling information.
    """
    
    __tablename__ = "staff"
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Basic identification
    employee_id = Column(
        String(50), 
        unique=True, 
        nullable=False,
        index=True,
        comment="Hospital employee identification number"
    )
    
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    
    # Professional information
    role = Column(
        String(50),  # Using String instead of Enum for flexibility
        nullable=False,
        index=True,
        comment="Staff role - see StaffRole enum for standard values"
    )
    
    primary_department = Column(
        String(50), 
        nullable=False,
        index=True,
        comment="Primary department assignment"
    )
    
    secondary_departments = Column(
        JSON, 
        nullable=True,
        default=list,
        comment="Additional departments staff can work in"
    )
    
    # Credentials and licensing
    license_number = Column(
        String(50), 
        unique=True, 
        nullable=True,
        comment="Professional license number"
    )
    
    board_certifications = Column(
        JSON, 
        nullable=True,
        default=list,
        comment="Board certifications held"
    )
    
    dea_number = Column(
        String(50), 
        unique=True, 
        nullable=True,
        comment="DEA registration number for prescribing"
    )
    
    npi_number = Column(
        String(20), 
        unique=True, 
        nullable=True,
        comment="National Provider Identifier"
    )
    
    # Contact information
    email = Column(String(200), nullable=False, unique=True)
    phone = Column(String(20), nullable=True)
    pager = Column(String(20), nullable=True)
    
    # Scheduling
    default_shift = Column(
        String(20), 
        nullable=False,
        default=ShiftType.DAY.value,
        comment="Preferred shift type"
    )
    
    max_hours_per_week = Column(
        Integer, 
        nullable=True,
        default=80,
        comment="Maximum contracted work hours per week"
    )
    
    # Clinical privileges
    clinical_privileges = Column(
        JSON, 
        nullable=True,
        default=list,
        comment="Approved clinical procedures and privileges"
    )
    
    admitting_privileges = Column(
        Boolean, 
        default=False,
        comment="Can admit patients to hospital"
    )
    
    # Performance metrics
    patient_satisfaction_score = Column(
        Float, 
        nullable=True,
        comment="Average patient satisfaction rating 0-100"
    )
    
    quality_metrics = Column(
        JSON, 
        nullable=True,
        comment="Clinical quality and outcome metrics"
    )
    
    # Employment status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    hire_date = Column(DateTime, nullable=False)
    termination_date = Column(DateTime, nullable=True)
    
    # Relationships
    # Many-to-many relationship with patients through admissions
    # This allows tracking which staff cared for which patients
    patients = relationship(
        "Patient",
        secondary="staff_patient_assignment",
        back_populates="care_team",
        lazy="dynamic"
    )
    
    # Shifts scheduled
    shifts = relationship(
        "StaffShift",
        back_populates="staff",
        lazy="dynamic",
        cascade="all, delete-orphan"
    )
    
    # Metadata
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    created_by = Column(String(100), nullable=True)
    
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    
    def get_full_name(self):
        """
        Get staff member's full name
        
        Returns:
            str: Formatted full name
        """
        return f"{self.first_name} {self.last_name}"
    
    def can_perform_procedure(self, procedure_code):
        """
        Check if staff has privileges for specific procedure
        
        Args:
            procedure_code: CPT or procedure code to check
            
        Returns:
            bool: True if staff has required privileges
        """
        if not self.clinical_privileges:
            return False
        return procedure_code in self.clinical_privileges
    
    def get_current_shift(self):
        """
        Get staff member's current shift if working
        
        Returns:
            StaffShift: Current shift or None if not working
        """
        now = datetime.now()
        for shift in self.shifts:
            if shift.start_time <= now <= shift.end_time:
                return shift
        return None
    
    def is_on_call(self):
        """
        Check if staff member is currently on call
        
        Returns:
            bool: True if on call status active
        """
        current_shift = self.get_current_shift()
        if current_shift:
            return current_shift.shift_type == ShiftType.ON_CALL.value
        return False
    
    def to_dict(self):
        """
        Convert staff to dictionary for API responses
        """
        return {
            "id": self.id,
            "employee_id": self.employee_id,
            "name": self.get_full_name(),
            "role": self.role,
            "primary_department": self.primary_department,
            "email": self.email,
            "phone": self.phone,
            "default_shift": self.default_shift,
            "is_active": self.is_active,
            "patient_satisfaction_score": self.patient_satisfaction_score,
            "hire_date": self.hire_date.isoformat() if self.hire_date else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class StaffShift(Base):
    """
    Staff Shift Model
    
    Tracks individual staff work shifts for scheduling and labor tracking.
    """
    
    __tablename__ = "staff_shifts"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Foreign key to staff
    staff_id = Column(
        String(36), 
        ForeignKey("staff.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Shift details
    shift_type = Column(
        String(20), 
        nullable=False,
        comment="Day, Evening, Night, or On-call"
    )
    
    department = Column(
        String(50), 
        nullable=False,
        comment="Department working in during this shift"
    )
    
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=False, index=True)
    
    # Shift status
    status = Column(
        String(20), 
        nullable=False,
        default="scheduled",
        comment="Scheduled, In Progress, Completed, Cancelled"
    )
    
    actual_start_time = Column(DateTime, nullable=True)
    actual_end_time = Column(DateTime, nullable=True)
    
    # Notes
    notes = Column(Text, nullable=True)
    
    # Relationships
    staff = relationship("Staff", back_populates="shifts")
    
    # Metadata
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    
    def calculate_hours_worked(self):
        """
        Calculate actual hours worked in this shift
        
        Returns:
            float: Hours worked, or None if not completed
        """
        if self.actual_start_time and self.actual_end_time:
            duration = self.actual_end_time - self.actual_start_time
            return duration.total_seconds() / 3600
        return None
    
    def is_currently_working(self):
        """
        Check if staff is currently working this shift
        
        Returns:
            bool: True if shift is in progress
        """
        now = datetime.now()
        return (self.status == "scheduled" and 
                self.start_time <= now <= self.end_time)
    
    def to_dict(self):
        """
        Convert shift to dictionary for API responses
        """
        return {
            "id": self.id,
            "staff_id": self.staff_id,
            "shift_type": self.shift_type,
            "department": self.department,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "hours_worked": self.calculate_hours_worked(),
            "is_currently_working": self.is_currently_working(),
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

# Association table for many-to-many relationship between staff and patients
# This table links staff members to patients they have cared for
staff_patient_assignment = Table(
    "staff_patient_assignment",
    Base.metadata,
    Column(
        "staff_id", 
        String(36), 
        ForeignKey("staff.id", ondelete="CASCADE"),
        primary_key=True
    ),
    Column(
        "patient_id", 
        String(36), 
        ForeignKey("patients.id", ondelete="CASCADE"),
        primary_key=True
    ),
    Column(
        "assignment_date", 
        DateTime, 
        default=datetime.now,
        comment="When staff was assigned to patient"
    ),
    Column(
        "role_in_care", 
        String(50),
        comment="Specific role for this patient e.g., Primary, Consulting"
    )
)