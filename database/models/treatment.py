"""
Treatment Model for AegisMedBot

This module defines treatment-related database models including:
- Medications administered to patients
- Procedures performed
- Treatment plans
- Clinical interventions

These models support clinical decision making and outcomes analysis.
"""

from sqlalchemy import (
    Column, String, Integer, DateTime, Float, ForeignKey, 
    Text, Boolean, JSON, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

# Import Base from the same module structure
# In production, this would be imported from a shared base module
from .admission import Base

class MedicationRoute(enum.Enum):
    """Standard routes of medication administration"""
    ORAL = "oral"                    # By mouth
    INTRAVENOUS = "intravenous"      # Into vein
    INTRAMUSCULAR = "intramuscular"  # Into muscle
    SUBCUTANEOUS = "subcutaneous"    # Under skin
    TOPICAL = "topical"              # Applied to skin
    INHALATION = "inhalation"        # Inhaled
    RECTAL = "rectal"                # Rectal
    OPHTHALMIC = "ophthalmic"        # Eye drops
    OTIC = "otic"                    # Ear drops
    NASAL = "nasal"                  # Nasal spray

class MedicationStatus(enum.Enum):
    """Current status of medication order"""
    ORDERED = "ordered"              # Prescribed but not started
    ACTIVE = "active"                # Currently being administered
    COMPLETED = "completed"          # Course completed
    DISCONTINUED = "discontinued"    # Stopped before completion
    HOLD = "hold"                    # Temporarily paused
    EXPIRED = "expired"              # Order expired

class Medication(Base):
    """
    Medication Administration Model
    
    Tracks all medications prescribed and administered to patients.
    Critical for:
    - Clinical decision support (drug interactions)
    - Patient safety monitoring
    - Billing and inventory management
    - Pharmacovigilance
    """
    
    __tablename__ = "medications"
    
    # Primary key - unique identifier for each medication order
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Foreign keys to link medication to patient and admission
    patient_id = Column(
        String(36), 
        ForeignKey("patients.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
        comment="Patient receiving medication"
    )
    
    admission_id = Column(
        String(36), 
        ForeignKey("admissions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Admission during which medication was prescribed"
    )
    
    # Drug identification
    drug_name = Column(
        String(200), 
        nullable=False,
        index=True,
        comment="Generic or brand name of medication"
    )
    
    drug_class = Column(
        String(100), 
        nullable=True,
        comment="Therapeutic class e.g., Beta-blocker, Antibiotic"
    )
    
    rxnorm_code = Column(
        String(50), 
        nullable=True,
        comment="Standard RxNorm identifier for interoperability"
    )
    
    ndc_code = Column(
        String(20), 
        nullable=True,
        comment="National Drug Code for inventory tracking"
    )
    
    # Dosage and administration details
    dosage = Column(
        String(100), 
        nullable=False,
        comment="Dosage amount e.g., 10mg, 500mg"
    )
    
    dosage_unit = Column(
        String(20), 
        nullable=True,
        comment="Unit of measurement: mg, mcg, g, mL"
    )
    
    route = Column(
        SQLEnum(MedicationRoute), 
        nullable=False,
        comment="How medication is administered"
    )
    
    frequency = Column(
        String(50), 
        nullable=False,
        comment="Administration frequency: BID, TID, QD, Q4H"
    )
    
    duration_days = Column(
        Integer, 
        nullable=True,
        comment="Expected duration of treatment in days"
    )
    
    # Timing information
    start_date = Column(
        DateTime, 
        nullable=False,
        default=datetime.now,
        index=True,
        comment="When medication was first administered"
    )
    
    end_date = Column(
        DateTime, 
        nullable=True,
        comment="When medication was discontinued or completed"
    )
    
    # Ordering and administration
    ordering_physician = Column(
        String(100), 
        nullable=False,
        index=True,
        comment="Physician who prescribed the medication"
    )
    
    administering_nurse = Column(
        String(100), 
        nullable=True,
        comment="Nurse who administered the medication"
    )
    
    status = Column(
        SQLEnum(MedicationStatus), 
        nullable=False,
        default=MedicationStatus.ORDERED,
        index=True,
        comment="Current status of medication order"
    )
    
    # Clinical decision support fields
    indication = Column(
        Text, 
        nullable=True,
        comment="Reason medication was prescribed"
    )
    
    clinical_notes = Column(
        Text, 
        nullable=True,
        comment="Additional clinical notes about administration"
    )
    
    # Safety monitoring
    requires_renal_adjustment = Column(
        Boolean, 
        default=False,
        comment="Dose adjustment needed for kidney function"
    )
    
    requires_hepatic_adjustment = Column(
        Boolean, 
        default=False,
        comment="Dose adjustment needed for liver function"
    )
    
    therapeutic_duplication_alert = Column(
        Boolean, 
        default=False,
        comment="Potential duplication of therapy detected"
    )
    
    # Adverse reaction tracking
    adverse_reactions = Column(
        JSON, 
        nullable=True,
        default=list,
        comment="Recorded adverse reactions or side effects"
    )
    # Example: [
    #   {"reaction": "Nausea", "severity": "moderate", "date": "2024-01-15"},
    #   {"reaction": "Rash", "severity": "mild", "date": "2024-01-16"}
    # ]
    
    # Metadata
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    created_by = Column(String(100), nullable=True)
    
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    
    # Relationships
    patient = relationship("Patient", back_populates="medications")
    admission = relationship("Admission", back_populates="medications")
    
    def is_active(self):
        """
        Determine if medication is currently active
        
        Returns:
            bool: True if medication is currently being administered
        """
        return (self.status == MedicationStatus.ACTIVE and 
                not self.end_date)
    
    def calculate_duration(self):
        """
        Calculate actual duration of medication administration
        
        Returns:
            float: Duration in days, or None if not applicable
        """
        if not self.start_date:
            return None
            
        end = self.end_date if self.end_date else datetime.now()
        difference = end - self.start_date
        return difference.total_seconds() / (24 * 3600)
    
    def to_dict(self):
        """
        Convert medication to dictionary for API responses
        """
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "admission_id": self.admission_id,
            "drug_name": self.drug_name,
            "drug_class": self.drug_class,
            "dosage": f"{self.dosage} {self.dosage_unit}" if self.dosage_unit else self.dosage,
            "route": self.route.value if self.route else None,
            "frequency": self.frequency,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "ordering_physician": self.ordering_physician,
            "status": self.status.value if self.status else None,
            "indication": self.indication,
            "adverse_reactions": self.adverse_reactions,
            "is_active": self.is_active(),
            "duration_days": self.calculate_duration(),
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class Procedure(Base):
    """
    Procedure Model
    
    Tracks surgical and interventional procedures performed on patients.
    Critical for:
    - Operating room scheduling
    - Resource allocation
    - Clinical outcomes tracking
    - Revenue cycle management
    """
    
    __tablename__ = "procedures"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Foreign keys
    patient_id = Column(
        String(36), 
        ForeignKey("patients.id", ondelete="RESTRICT"),
        nullable=False,
        index=True
    )
    
    admission_id = Column(
        String(36), 
        ForeignKey("admissions.id", ondelete="CASCADE"),
        nullable=True,  # Some procedures may be outpatient
        index=True
    )
    
    # Procedure identification
    procedure_name = Column(
        String(200), 
        nullable=False,
        index=True,
        comment="Name of surgical or interventional procedure"
    )
    
    cpt_code = Column(
        String(20), 
        nullable=False,
        index=True,
        comment="Current Procedural Terminology code for billing"
    )
    
    icd_pcs_code = Column(
        String(20), 
        nullable=True,
        comment="ICD-10 Procedure Coding System code"
    )
    
    # Timing
    scheduled_date = Column(
        DateTime, 
        nullable=False,
        comment="Scheduled procedure date and time"
    )
    
    actual_start_time = Column(
        DateTime, 
        nullable=True,
        comment="Actual start time of procedure"
    )
    
    actual_end_time = Column(
        DateTime, 
        nullable=True,
        comment="Actual end time of procedure"
    )
    
    # Location and resources
    operating_room = Column(
        String(50), 
        nullable=True,
        comment="OR room number"
    )
    
    primary_surgeon = Column(
        String(100), 
        nullable=False,
        index=True,
        comment="Lead surgeon performing procedure"
    )
    
    assistant_surgeon = Column(
        String(100), 
        nullable=True,
        comment="Assistant surgeon"
    )
    
    anesthesiologist = Column(
        String(100), 
        nullable=True,
        comment="Anesthesia provider"
    )
    
    scrub_nurse = Column(
        String(100), 
        nullable=True,
        comment="Scrub nurse or circulating nurse"
    )
    
    # Clinical details
    anesthesia_type = Column(
        String(50), 
        nullable=True,
        comment="General, Regional, Local, MAC"
    )
    
    procedure_details = Column(
        JSON, 
        nullable=True,
        comment="Detailed procedure notes and findings"
    )
    
    complications = Column(
        JSON, 
        nullable=True,
        comment="Intra-operative or post-operative complications"
    )
    
    # Outcome tracking
    status = Column(
        String(50), 
        nullable=False,
        default="scheduled",
        comment="Scheduled, In Progress, Completed, Cancelled"
    )
    
    cancellation_reason = Column(
        Text, 
        nullable=True,
        comment="Reason if procedure was cancelled"
    )
    
    # Quality metrics
    time_to_procedure_hours = Column(
        Float, 
        nullable=True,
        comment="Hours from admission to procedure start"
    )
    
    procedure_duration_minutes = Column(
        Integer, 
        nullable=True,
        comment="Actual duration in minutes"
    )
    
    # Metadata
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    def calculate_wait_time(self):
        """
        Calculate wait time from admission to procedure
        
        Returns:
            float: Wait time in hours, or None if not applicable
        """
        if not self.admission_id or not self.actual_start_time:
            return None
            
        # Would need to fetch admission record to get admission_date
        # This is simplified - in practice would query database
        return self.time_to_procedure_hours
    
    def calculate_duration(self):
        """
        Calculate procedure duration in minutes
        
        Returns:
            int: Duration in minutes, or None if timing incomplete
        """
        if self.actual_start_time and self.actual_end_time:
            duration = self.actual_end_time - self.actual_start_time
            return int(duration.total_seconds() / 60)
        return self.procedure_duration_minutes
    
    def to_dict(self):
        """
        Convert procedure to dictionary for API responses
        """
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "admission_id": self.admission_id,
            "procedure_name": self.procedure_name,
            "cpt_code": self.cpt_code,
            "scheduled_date": self.scheduled_date.isoformat() if self.scheduled_date else None,
            "actual_start_time": self.actual_start_time.isoformat() if self.actual_start_time else None,
            "actual_end_time": self.actual_end_time.isoformat() if self.actual_end_time else None,
            "operating_room": self.operating_room,
            "primary_surgeon": self.primary_surgeon,
            "anesthesia_type": self.anesthesia_type,
            "status": self.status,
            "complications": self.complications,
            "duration_minutes": self.calculate_duration(),
            "wait_time_hours": self.calculate_wait_time(),
            "created_at": self.created_at.isoformat() if self.created_at else None
        }