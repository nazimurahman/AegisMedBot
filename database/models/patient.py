"""
Patient Model Module for AegisMedBot

This module defines the Patient database model and related entities for storing
patient demographic information, medical history, and clinical data.
It serves as the core data structure for all patient-related operations in the system.
"""

from sqlalchemy import Column, String, Integer, DateTime, Date, JSON, Boolean, Float, Text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid
import json
from typing import List, Dict, Any, Optional

# Create base class for all database models
# This base class maintains the SQLAlchemy registry and provides
# common functionality for all model classes
Base = declarative_base()

class Patient(Base):
    """
    Patient Model - Core entity representing a patient in the hospital system
    
    This model stores comprehensive patient information including demographics,
    medical history, insurance details, and emergency contacts. It serves as
    the central reference for all patient-related data across the platform.
    
    Attributes:
        id: Unique identifier (UUID) for the patient record
        mrn: Medical Record Number - unique identifier used in hospital systems
        first_name: Patient's first name
        last_name: Patient's last name
        date_of_birth: Patient's birth date for age calculation
        gender: Patient's gender identity
        blood_type: Blood group for emergency transfusions
        allergies: JSON array of known allergies with severity levels
        chronic_conditions: JSON array of chronic medical conditions
        emergency_contact: JSON object with emergency contact information
        insurance_info: JSON object with insurance provider details
        created_at: Timestamp when record was created
        updated_at: Timestamp when record was last modified
        is_active: Soft delete flag - false means patient record is archived
    """
    
    # Table name in PostgreSQL database
    __tablename__ = "patients"
    
    # Primary key - using UUID string for distributed systems compatibility
    # UUID ensures unique identifiers across different database shards
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Medical Record Number - must be unique and indexed for fast lookup
    # This is the primary identifier used by hospital staff
    mrn = Column(String(50), unique=True, nullable=False, index=True)
    
    # Patient identification fields
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    
    # Date of birth - used for age calculation and demographic analysis
    date_of_birth = Column(Date, nullable=False)
    
    # Gender field - using string to accommodate diverse gender identities
    gender = Column(String(20))
    
    # Clinical information fields
    blood_type = Column(String(5))  # A+, A-, B+, B-, AB+, AB-, O+, O-
    
    # JSON fields for flexible data storage without schema changes
    # JSON allows storing complex nested data structures
    allergies = Column(JSON, default=list)  # Stores list of allergy objects
    chronic_conditions = Column(JSON, default=list)  # Stores list of condition objects
    
    # Administrative information
    emergency_contact = Column(JSON, default=dict)  # Stores contact person details
    insurance_info = Column(JSON, default=dict)  # Stores insurance policy details
    
    # Metadata fields for tracking and auditing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships to other tables
    # These define foreign key relationships for joined queries
    admissions = relationship("Admission", back_populates="patient", cascade="all, delete-orphan")
    vitals = relationship("VitalSign", back_populates="patient", cascade="all, delete-orphan")
    lab_results = relationship("LabResult", back_populates="patient", cascade="all, delete-orphan")
    medications = relationship("Medication", back_populates="patient", cascade="all, delete-orphan")
    clinical_notes = relationship("ClinicalNote", back_populates="patient", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert patient object to dictionary for JSON serialization
        
        This method transforms the SQLAlchemy model into a Python dictionary
        suitable for API responses. It handles date and JSON serialization properly.
        
        Returns:
            Dictionary containing all patient fields with proper formatting
        """
        return {
            "id": self.id,
            "mrn": self.mrn,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": f"{self.first_name} {self.last_name}",  # Convenience field
            "date_of_birth": self.date_of_birth.isoformat() if self.date_of_birth else None,
            "age": self._calculate_age(),  # Dynamically calculated age
            "gender": self.gender,
            "blood_type": self.blood_type,
            "allergies": self.allergies or [],
            "chronic_conditions": self.chronic_conditions or [],
            "emergency_contact": self.emergency_contact or {},
            "insurance_info": self.insurance_info or {},
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    def _calculate_age(self) -> Optional[int]:
        """
        Calculate patient's age from date of birth
        
        This internal method computes the current age in years based on
        the stored date of birth and current date.
        
        Returns:
            Age in years as integer, or None if date of birth is missing
        """
        if self.date_of_birth:
            today = datetime.utcnow().date()
            age = today.year - self.date_of_birth.year
            # Adjust age if birthday hasn't occurred yet this year
            if today.month < self.date_of_birth.month or \
               (today.month == self.date_of_birth.month and today.day < self.date_of_birth.day):
                age -= 1
            return age
        return None
    
    def add_allergy(self, allergen: str, severity: str, reaction: str) -> None:
        """
        Add a new allergy to patient's record
        
        This method safely appends a new allergy to the JSON allergies array.
        
        Args:
            allergen: Name of the allergen (e.g., "Penicillin")
            severity: Severity level ("Mild", "Moderate", "Severe", "Life-threatening")
            reaction: Description of allergic reaction
        """
        if self.allergies is None:
            self.allergies = []
        
        allergy_entry = {
            "allergen": allergen,
            "severity": severity,
            "reaction": reaction,
            "recorded_date": datetime.utcnow().isoformat()
        }
        self.allergies.append(allergy_entry)
        self.updated_at = datetime.utcnow()
    
    def add_chronic_condition(self, condition: str, diagnosed_date: str, status: str) -> None:
        """
        Add a chronic condition to patient's medical history
        
        Args:
            condition: Medical condition name (e.g., "Type 2 Diabetes")
            diagnosed_date: Date of diagnosis in ISO format
            status: Current status ("Active", "Managed", "In Remission")
        """
        if self.chronic_conditions is None:
            self.chronic_conditions = []
        
        condition_entry = {
            "condition": condition,
            "diagnosed_date": diagnosed_date,
            "status": status,
            "last_updated": datetime.utcnow().isoformat()
        }
        self.chronic_conditions.append(condition_entry)
        self.updated_at = datetime.utcnow()


class Admission(Base):
    """
    Admission Model - Records patient hospital admissions and stays
    
    This model tracks each patient's admission episodes including admission dates,
    discharge information, assigned departments, and attending physicians.
    It is crucial for bed management and patient flow analytics.
    
    Attributes:
        id: Unique identifier for the admission record
        patient_id: Foreign key reference to Patient table
        admission_date: Date and time of admission
        discharge_date: Date and time of discharge (null if still admitted)
        admission_type: Type of admission (Emergency, Elective, Urgent, Transfer)
        department: Hospital department where patient is admitted
        room_number: Physical room location
        bed_number: Specific bed within the room
        diagnosis: JSON array of primary and secondary diagnoses with ICD codes
        admitting_physician: Doctor who admitted the patient
        attending_physician: Primary doctor responsible for care
        status: Current admission status
        discharge_disposition: Where patient went after discharge
    """
    
    __tablename__ = "admissions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Foreign key relationship to Patient table
    # Indexed for faster join queries
    patient_id = Column(String(36), nullable=False, index=True)
    
    # Admission timeline fields
    admission_date = Column(DateTime, nullable=False)
    discharge_date = Column(DateTime)
    
    # Classification fields
    admission_type = Column(String(50), nullable=False)  # Emergency, Elective, Urgent, Transfer
    department = Column(String(100), nullable=False)
    room_number = Column(String(20))
    bed_number = Column(String(20))
    
    # Clinical information
    # JSON allows storing complex diagnosis structures with ICD codes
    diagnosis = Column(JSON, default=dict)  # Primary and secondary diagnoses
    admitting_physician = Column(String(100))
    attending_physician = Column(String(100))
    
    # Status tracking
    status = Column(String(50), default="Active", nullable=False)  # Active, Discharged, Transferred
    discharge_disposition = Column(String(100))  # Home, Skilled Nursing, Transfer, Expired
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    patient = relationship("Patient", back_populates="admissions")
    vitals = relationship("VitalSign", back_populates="admission", cascade="all, delete-orphan")
    lab_results = relationship("LabResult", back_populates="admission", cascade="all, delete-orphan")
    medications = relationship("Medication", back_populates="admission", cascade="all, delete-orphan")
    clinical_notes = relationship("ClinicalNote", back_populates="admission", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert admission record to dictionary for API responses
        
        Returns:
            Dictionary containing admission information with computed fields
        """
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "admission_date": self.admission_date.isoformat() if self.admission_date else None,
            "discharge_date": self.discharge_date.isoformat() if self.discharge_date else None,
            "length_of_stay_days": self._calculate_los(),
            "admission_type": self.admission_type,
            "department": self.department,
            "room_number": self.room_number,
            "bed_number": self.bed_number,
            "primary_diagnosis": self.diagnosis.get("primary", "") if self.diagnosis else "",
            "secondary_diagnoses": self.diagnosis.get("secondary", []) if self.diagnosis else [],
            "admitting_physician": self.admitting_physician,
            "attending_physician": self.attending_physician,
            "status": self.status,
            "discharge_disposition": self.discharge_disposition,
            "is_currently_admitted": self.status == "Active",
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    def _calculate_los(self) -> Optional[float]:
        """
        Calculate Length of Stay in days
        
        For active admissions, calculates from admission to current time.
        For discharged, calculates from admission to discharge.
        
        Returns:
            Length of stay in days as float, or None if admission_date missing
        """
        if not self.admission_date:
            return None
        
        end_date = self.discharge_date if self.discharge_date else datetime.utcnow()
        delta = end_date - self.admission_date
        return round(delta.total_seconds() / 86400, 1)  # Convert seconds to days
    
    def discharge(self, discharge_date: DateTime, disposition: str) -> None:
        """
        Mark admission as discharged
        
        This method updates the admission record with discharge information.
        
        Args:
            discharge_date: Date and time of discharge
            disposition: Where patient is going after discharge
        """
        self.discharge_date = discharge_date
        self.discharge_disposition = disposition
        self.status = "Discharged"
        self.updated_at = datetime.utcnow()


class VitalSign(Base):
    """
    Vital Signs Model - Tracks patient physiological measurements
    
    This model stores time-series vital sign data for patient monitoring,
    trend analysis, and early warning score calculation.
    
    Attributes:
        id: Unique identifier for the vital sign reading
        patient_id: Foreign key reference to Patient
        admission_id: Foreign key reference to Admission (optional)
        timestamp: When the vital signs were measured
        heart_rate: Beats per minute (normal: 60-100)
        blood_pressure_systolic: Top number - pressure during heart contraction
        blood_pressure_diastolic: Bottom number - pressure between beats
        respiratory_rate: Breaths per minute (normal: 12-20)
        temperature: Body temperature in Celsius (normal: 36.1-37.2)
        oxygen_saturation: SpO2 percentage (normal: 95-100)
        pain_level: Patient reported pain on 0-10 scale
        recorded_by: Staff member who recorded the vitals
    """
    
    __tablename__ = "vital_signs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Foreign keys - indexed for efficient time-series queries
    patient_id = Column(String(36), nullable=False, index=True)
    admission_id = Column(String(36), index=True)
    
    # Measurement timestamp - critical for time-series analysis
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Vital signs measurements
    heart_rate = Column(Integer)  # Beats per minute
    blood_pressure_systolic = Column(Integer)  # mmHg
    blood_pressure_diastolic = Column(Integer)  # mmHg
    respiratory_rate = Column(Integer)  # Breaths per minute
    temperature = Column(Float)  # Celsius
    oxygen_saturation = Column(Integer)  # Percentage
    pain_level = Column(Integer)  # 0-10 scale
    
    # Metadata
    recorded_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    patient = relationship("Patient", back_populates="vitals")
    admission = relationship("Admission", back_populates="vitals")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert vital signs to dictionary for API responses
        
        Returns:
            Dictionary with vital signs and calculated metrics
        """
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "admission_id": self.admission_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "heart_rate": self.heart_rate,
            "blood_pressure_systolic": self.blood_pressure_systolic,
            "blood_pressure_diastolic": self.blood_pressure_diastolic,
            "blood_pressure": f"{self.blood_pressure_systolic}/{self.blood_pressure_diastolic}" if self.blood_pressure_systolic and self.blood_pressure_diastolic else None,
            "respiratory_rate": self.respiratory_rate,
            "temperature": self.temperature,
            "temperature_fahrenheit": round(self.temperature * 9/5 + 32, 1) if self.temperature else None,
            "oxygen_saturation": self.oxygen_saturation,
            "pain_level": self.pain_level,
            "recorded_by": self.recorded_by,
            "ews_score": self._calculate_early_warning_score(),
            "is_critical": self._is_critical(),
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    def _calculate_early_warning_score(self) -> int:
        """
        Calculate National Early Warning Score (NEWS)
        
        NEWS is a standardized scoring system for detecting clinical deterioration.
        Higher scores indicate greater risk of adverse outcomes.
        
        Returns:
            NEWS score from 0-20 where 0-4 is low risk, 5-6 is medium,
            7-8 is high, and 9+ is very high risk
        """
        score = 0
        
        # Heart rate scoring (bpm)
        if self.heart_rate:
            if self.heart_rate <= 40:
                score += 3
            elif 41 <= self.heart_rate <= 50:
                score += 2
            elif 51 <= self.heart_rate <= 90:
                score += 0
            elif 91 <= self.heart_rate <= 110:
                score += 1
            elif 111 <= self.heart_rate <= 130:
                score += 2
            elif self.heart_rate >= 131:
                score += 3
        
        # Respiratory rate scoring
        if self.respiratory_rate:
            if self.respiratory_rate <= 8:
                score += 3
            elif 9 <= self.respiratory_rate <= 11:
                score += 1
            elif 12 <= self.respiratory_rate <= 20:
                score += 0
            elif 21 <= self.respiratory_rate <= 24:
                score += 2
            elif self.respiratory_rate >= 25:
                score += 3
        
        # Oxygen saturation scoring
        if self.oxygen_saturation:
            if self.oxygen_saturation <= 91:
                score += 3
            elif 92 <= self.oxygen_saturation <= 93:
                score += 2
            elif 94 <= self.oxygen_saturation <= 95:
                score += 1
            elif self.oxygen_saturation >= 96:
                score += 0
        
        # Temperature scoring (Celsius)
        if self.temperature:
            if self.temperature <= 35.0:
                score += 3
            elif 35.1 <= self.temperature <= 36.0:
                score += 1
            elif 36.1 <= self.temperature <= 38.0:
                score += 0
            elif 38.1 <= self.temperature <= 39.0:
                score += 1
            elif self.temperature >= 39.1:
                score += 2
        
        return score
    
    def _is_critical(self) -> bool:
        """
        Determine if vital signs indicate critical condition
        
        Checks for values that require immediate medical attention
        
        Returns:
            Boolean indicating critical condition
        """
        if self.heart_rate and (self.heart_rate < 40 or self.heart_rate > 140):
            return True
        if self.oxygen_saturation and self.oxygen_saturation < 85:
            return True
        if self.respiratory_rate and (self.respiratory_rate < 8 or self.respiratory_rate > 30):
            return True
        if self.temperature and (self.temperature < 35.0 or self.temperature > 39.5):
            return True
        if self.blood_pressure_systolic and self.blood_pressure_systolic < 80:
            return True
        return False


class LabResult(Base):
    """
    Lab Results Model - Stores laboratory test results for patients
    
    This model tracks all laboratory tests ordered for patients including
    results, reference ranges, and interpretations.
    
    Attributes:
        id: Unique identifier for the lab result
        patient_id: Foreign key reference to Patient
        admission_id: Foreign key reference to Admission (optional)
        test_name: Name of the laboratory test
        test_code: LOINC code for standardized identification
        result_value: Numeric result if applicable
        result_text: Textual result for qualitative tests
        unit: Measurement unit
        reference_range: Normal reference range
        interpretation: Clinical interpretation (Normal, Abnormal, Critical)
        collected_at: When specimen was collected
        resulted_at: When results were available
    """
    
    __tablename__ = "lab_results"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Foreign keys
    patient_id = Column(String(36), nullable=False, index=True)
    admission_id = Column(String(36), index=True)
    
    # Test identification
    test_name = Column(String(200), nullable=False)
    test_code = Column(String(50))  # LOINC standard code
    test_category = Column(String(100))  # Hematology, Chemistry, Microbiology, etc.
    
    # Results
    result_value = Column(Float)  # Numeric result
    result_text = Column(Text)  # Text result for cultures, qualitative tests
    unit = Column(String(20))
    reference_range = Column(String(100))
    
    # Clinical interpretation
    interpretation = Column(String(50))  # Normal, Abnormal Low, Abnormal High, Critical
    flag = Column(String(10))  # H (High), L (Low), N (Normal), C (Critical)
    
    # Timing
    ordered_at = Column(DateTime)
    collected_at = Column(DateTime)
    resulted_at = Column(DateTime)
    
    # Metadata
    ordered_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    patient = relationship("Patient", back_populates="lab_results")
    admission = relationship("Admission", back_populates="lab_results")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert lab result to dictionary for API responses
        
        Returns:
            Dictionary with lab result data and clinical interpretation
        """
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "admission_id": self.admission_id,
            "test_name": self.test_name,
            "test_code": self.test_code,
            "test_category": self.test_category,
            "result": f"{self.result_value} {self.unit}" if self.result_value else self.result_text,
            "result_value": self.result_value,
            "result_text": self.result_text,
            "unit": self.unit,
            "reference_range": self.reference_range,
            "interpretation": self.interpretation,
            "flag": self.flag,
            "is_abnormal": self.interpretation != "Normal",
            "is_critical": self.interpretation == "Critical",
            "ordered_at": self.ordered_at.isoformat() if self.ordered_at else None,
            "collected_at": self.collected_at.isoformat() if self.collected_at else None,
            "resulted_at": self.resulted_at.isoformat() if self.resulted_at else None,
            "ordered_by": self.ordered_by,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    def set_interpretation(self) -> None:
        """
        Automatically set interpretation based on result value and reference range
        
        This method parses the reference range and determines if the result
        is normal, abnormal, or critical based on medical guidelines.
        """
        if not self.result_value or not self.reference_range:
            return
        
        # Parse reference range (e.g., "4.0-11.0" or "<5.0" or ">100")
        import re
        range_parts = re.findall(r'(\d+\.?\d*)', self.reference_range)
        
        if len(range_parts) == 2:  # Normal range with lower and upper bounds
            lower = float(range_parts[0])
            upper = float(range_parts[1])
            
            if self.result_value < lower:
                self.interpretation = "Abnormal Low"
                self.flag = "L"
            elif self.result_value > upper:
                self.interpretation = "Abnormal High"
                self.flag = "H"
            else:
                self.interpretation = "Normal"
                self.flag = "N"
        
        # Critical value check (simplified - in production use medical thresholds)
        if self.test_category == "Hematology":
            if self.test_name == "Hemoglobin" and self.result_value < 7:
                self.interpretation = "Critical"
                self.flag = "C"
            elif self.test_name == "Platelets" and self.result_value < 20000:
                self.interpretation = "Critical"
                self.flag = "C"


class Medication(Base):
    """
    Medication Model - Tracks patient medications and prescriptions
    
    This model stores medication orders, administration records, and
    prescription details for patient care management.
    
    Attributes:
        id: Unique identifier for medication record
        patient_id: Foreign key reference to Patient
        admission_id: Foreign key reference to Admission
        medication_name: Generic or brand name of medication
        route: Administration route (Oral, IV, IM, Subcutaneous, etc.)
        dosage: Amount of medication
        dosage_unit: Unit of measurement (mg, mcg, mL, etc.)
        frequency: How often to administer (BID, TID, QD, PRN, etc.)
        start_date: When medication was started
        end_date: When medication was discontinued (null if active)
        status: Active, Completed, Discontinued, On Hold
        prescribed_by: Doctor who prescribed the medication
        pharmacy_notes: Additional instructions for pharmacy
    """
    
    __tablename__ = "medications"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Foreign keys
    patient_id = Column(String(36), nullable=False, index=True)
    admission_id = Column(String(36), index=True)
    
    # Medication details
    medication_name = Column(String(200), nullable=False)
    generic_name = Column(String(200))
    medication_class = Column(String(100))  # Antibiotic, Antihypertensive, etc.
    
    # Administration details
    route = Column(String(50))  # Oral, IV, IM, Subcutaneous, Topical, etc.
    dosage = Column(Float)
    dosage_unit = Column(String(20))  # mg, mcg, g, mL, etc.
    frequency = Column(String(50))  # BID, TID, QD, Q4H, PRN, etc.
    
    # Timing
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime)
    
    # Status tracking
    status = Column(String(50), default="Active", nullable=False)
    is_controlled_substance = Column(Boolean, default=False)
    
    # Clinical information
    indication = Column(Text)  # Why medication is prescribed
    instructions = Column(Text)  # Special administration instructions
    side_effects = Column(JSON, default=list)  # Known side effects to monitor
    
    # Provider information
    prescribed_by = Column(String(100))
    verified_by = Column(String(100))  # Pharmacist verification
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    patient = relationship("Patient", back_populates="medications")
    admission = relationship("Admission", back_populates="medications")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert medication record to dictionary for API responses
        
        Returns:
            Dictionary with medication details and administration information
        """
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "admission_id": self.admission_id,
            "medication_name": self.medication_name,
            "generic_name": self.generic_name,
            "medication_class": self.medication_class,
            "route": self.route,
            "dosage": f"{self.dosage} {self.dosage_unit}" if self.dosage and self.dosage_unit else None,
            "dosage_amount": self.dosage,
            "dosage_unit": self.dosage_unit,
            "frequency": self.frequency,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "status": self.status,
            "is_active": self.status == "Active" and (not self.end_date or self.end_date > datetime.utcnow()),
            "is_controlled_substance": self.is_controlled_substance,
            "indication": self.indication,
            "instructions": self.instructions,
            "side_effects": self.side_effects or [],
            "prescribed_by": self.prescribed_by,
            "verified_by": self.verified_by,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    def discontinue(self, reason: str) -> None:
        """
        Discontinue a medication order
        
        Args:
            reason: Clinical reason for discontinuation
        """
        self.end_date = datetime.utcnow()
        self.status = "Discontinued"
        self.instructions = f"{self.instructions or ''} Discontinued: {reason}"
        self.updated_at = datetime.utcnow()


class ClinicalNote(Base):
    """
    Clinical Notes Model - Stores doctor and nurse progress notes
    
    This model captures all clinical documentation including progress notes,
    consultation notes, discharge summaries, and other medical documentation.
    
    Attributes:
        id: Unique identifier for the clinical note
        patient_id: Foreign key reference to Patient
        admission_id: Foreign key reference to Admission
        note_type: Type of note (Progress, Admission, Discharge, Consult, etc.)
        title: Note title or subject
        content: Full text content of the clinical note
        author: Healthcare provider who wrote the note
        note_date: Date when the note was written (clinical date)
        is_signed: Whether note has been electronically signed
        signed_by: Provider who signed the note
        signed_at: When note was signed
    """
    
    __tablename__ = "clinical_notes"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Foreign keys
    patient_id = Column(String(36), nullable=False, index=True)
    admission_id = Column(String(36), index=True)
    
    # Note metadata
    note_type = Column(String(50), nullable=False)  # Progress, Admission, Discharge, Consult, Operative
    title = Column(String(200))
    content = Column(Text, nullable=False)
    
    # Author information
    author = Column(String(100), nullable=False)
    author_role = Column(String(50))  # Physician, Nurse, Resident, etc.
    note_date = Column(DateTime, nullable=False)  # Clinical date of the note
    
    # Signatures
    is_signed = Column(Boolean, default=False)
    signed_by = Column(String(100))
    signed_at = Column(DateTime)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    patient = relationship("Patient", back_populates="clinical_notes")
    admission = relationship("Admission", back_populates="clinical_notes")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert clinical note to dictionary for API responses
        
        Returns:
            Dictionary with note content and metadata
        """
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "admission_id": self.admission_id,
            "note_type": self.note_type,
            "title": self.title,
            "content": self.content,
            "content_preview": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "author": self.author,
            "author_role": self.author_role,
            "note_date": self.note_date.isoformat() if self.note_date else None,
            "is_signed": self.is_signed,
            "signed_by": self.signed_by,
            "signed_at": self.signed_at.isoformat() if self.signed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    def sign(self, signer: str) -> None:
        """
        Electronically sign the clinical note
        
        Args:
            signer: Name of the provider signing the note
        """
        self.is_signed = True
        self.signed_by = signer
        self.signed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()


class Staff(Base):
    """
    Staff Model - Manages hospital staff information and credentials
    
    This model stores information about all hospital staff members including
    physicians, nurses, administrators, and support staff.
    
    Attributes:
        id: Unique identifier for staff member
        employee_id: Hospital employee identification number
        first_name: Staff member's first name
        last_name: Staff member's last name
        role: Job role (Physician, Nurse, Administrator, etc.)
        department: Assigned department
        specialization: Medical specialty for clinical staff
        license_number: Professional license number
        npi_number: National Provider Identifier for US providers
        email: Work email address
        phone: Work phone number
        is_active: Whether staff member is currently employed
    """
    
    __tablename__ = "staff"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Identification
    employee_id = Column(String(50), unique=True, nullable=False, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    
    # Professional information
    role = Column(String(50), nullable=False)  # Physician, Nurse, Pharmacist, etc.
    department = Column(String(100))
    specialization = Column(String(100))  # Cardiology, Neurology, etc.
    
    # Credentials
    license_number = Column(String(50))
    npi_number = Column(String(20))  # National Provider Identifier
    dea_number = Column(String(20))  # DEA registration for controlled substances
    
    # Contact information
    email = Column(String(200), unique=True, index=True)
    phone = Column(String(20))
    pager_number = Column(String(20))
    
    # Schedule and availability
    on_call_status = Column(Boolean, default=False)
    current_shift = Column(String(50))  # Day, Evening, Night, Weekend
    
    # Metadata
    hire_date = Column(Date)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert staff record to dictionary for API responses
        
        Returns:
            Dictionary with staff information
        """
        return {
            "id": self.id,
            "employee_id": self.employee_id,
            "full_name": f"{self.first_name} {self.last_name}",
            "first_name": self.first_name,
            "last_name": self.last_name,
            "role": self.role,
            "department": self.department,
            "specialization": self.specialization,
            "license_number": self.license_number,
            "npi_number": self.npi_number,
            "email": self.email,
            "phone": self.phone,
            "pager_number": self.pager_number,
            "on_call_status": self.on_call_status,
            "current_shift": self.current_shift,
            "hire_date": self.hire_date.isoformat() if self.hire_date else None,
            "is_active": self.is_active,
            "years_of_service": self._calculate_years_of_service()
        }
    
    def _calculate_years_of_service(self) -> Optional[float]:
        """
        Calculate years of service from hire date
        
        Returns:
            Years of service as float, or None if hire date missing
        """
        if self.hire_date:
            today = datetime.utcnow().date()
            delta = today - self.hire_date
            return round(delta.days / 365.25, 1)
        return None


class Resource(Base):
    """
    Resource Model - Manages hospital equipment and facility resources
    
    This model tracks all hospital resources including beds, ventilators,
    infusion pumps, and other medical equipment for capacity management.
    
    Attributes:
        id: Unique identifier for the resource
        resource_type: Type of resource (Bed, Ventilator, Monitor, etc.)
        resource_id: Internal tracking identifier
        location: Physical location in hospital
        department: Assigned department
        status: Current operational status
        is_available: Whether resource is available for use
        patient_id: Patient currently using resource (if applicable)
        maintenance_due: Next scheduled maintenance date
    """
    
    __tablename__ = "resources"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Resource identification
    resource_type = Column(String(50), nullable=False)  # Bed, Ventilator, Monitor, Pump
    resource_id = Column(String(50), unique=True, nullable=False, index=True)
    model_number = Column(String(100))
    serial_number = Column(String(100))
    
    # Location tracking
    location = Column(String(200))  # Room number, department area
    department = Column(String(100))
    floor = Column(Integer)
    wing = Column(String(50))
    
    # Status management
    status = Column(String(50), default="Available", nullable=False)
    # Status options: Available, In Use, Maintenance, Out of Service, Reserved
    is_available = Column(Boolean, default=True)
    is_clean = Column(Boolean, default=True)
    
    # Usage tracking
    patient_id = Column(String(36), index=True)  # Current patient if in use
    assigned_to = Column(String(100))  # Staff member responsible
    
    # Maintenance
    last_maintenance = Column(Date)
    maintenance_due = Column(Date)
    vendor = Column(String(100))
    
    # Metadata
    acquired_date = Column(Date)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert resource record to dictionary for API responses
        
        Returns:
            Dictionary with resource information and availability status
        """
        return {
            "id": self.id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "model_number": self.model_number,
            "serial_number": self.serial_number,
            "location": self.location,
            "department": self.department,
            "floor": self.floor,
            "wing": self.wing,
            "status": self.status,
            "is_available": self.is_available,
            "is_clean": self.is_clean,
            "patient_id": self.patient_id,
            "assigned_to": self.assigned_to,
            "last_maintenance": self.last_maintenance.isoformat() if self.last_maintenance else None,
            "maintenance_due": self.maintenance_due.isoformat() if self.maintenance_due else None,
            "maintenance_overdue": self.maintenance_due and self.maintenance_due < datetime.utcnow().date(),
            "vendor": self.vendor,
            "acquired_date": self.acquired_date.isoformat() if self.acquired_date else None
        }
    
    def assign_to_patient(self, patient_id: str, staff_id: str) -> None:
        """
        Assign resource to a patient
        
        Args:
            patient_id: ID of patient using the resource
            staff_id: ID of staff member making assignment
        """
        self.patient_id = patient_id
        self.assigned_to = staff_id
        self.status = "In Use"
        self.is_available = False
        self.updated_at = datetime.utcnow()
    
    def release_resource(self) -> None:
        """
        Release resource from patient assignment
        """
        self.patient_id = None
        self.status = "Available"
        self.is_available = True
        self.is_clean = False  # Needs cleaning before next use
        self.updated_at = datetime.utcnow()
    
    def mark_clean(self) -> None:
        """
        Mark resource as clean and ready for use
        """
        self.is_clean = True
        if self.status == "Available":
            self.status = "Available"
        self.updated_at = datetime.utcnow()