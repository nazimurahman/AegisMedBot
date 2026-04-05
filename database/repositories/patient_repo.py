"""
Patient Repository Module for AegisMedBot

This module implements the repository pattern for database operations
related to patients. It provides a clean abstraction layer between
the business logic and the database, encapsulating all CRUD operations.
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, date, timedelta
import logging

from database.models.patient import Patient, Admission, VitalSign, LabResult, ClinicalNote

# Configure logging for repository operations
logger = logging.getLogger(__name__)

class PatientRepository:
    """
    Repository for Patient database operations
    
    This class handles all database interactions for the Patient model
    and its related entities. It follows the repository pattern to
    separate data access logic from business logic.
    
    Attributes:
        session: SQLAlchemy database session for executing queries
    """
    
    def __init__(self, session: Session):
        """
        Initialize repository with database session
        
        Args:
            session: SQLAlchemy Session object for database operations
        """
        self.session = session
    
    def create_patient(self, patient_data: Dict[str, Any]) -> Patient:
        """
        Create a new patient record in the database
        
        This method validates required fields, creates a Patient object,
        and persists it to the database.
        
        Args:
            patient_data: Dictionary containing patient information
            
        Returns:
            Created Patient object with assigned ID
            
        Raises:
            ValueError: If required fields are missing
        """
        # Validate required fields
        required_fields = ["mrn", "first_name", "last_name", "date_of_birth"]
        for field in required_fields:
            if field not in patient_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Create patient object
        patient = Patient(
            mrn=patient_data["mrn"],
            first_name=patient_data["first_name"],
            last_name=patient_data["last_name"],
            date_of_birth=datetime.strptime(patient_data["date_of_birth"], "%Y-%m-%d").date() 
                if isinstance(patient_data["date_of_birth"], str) else patient_data["date_of_birth"],
            gender=patient_data.get("gender"),
            blood_type=patient_data.get("blood_type"),
            allergies=patient_data.get("allergies", []),
            chronic_conditions=patient_data.get("chronic_conditions", []),
            emergency_contact=patient_data.get("emergency_contact", {}),
            insurance_info=patient_data.get("insurance_info", {})
        )
        
        # Add to session and commit
        self.session.add(patient)
        self.session.commit()
        self.session.refresh(patient)
        
        logger.info(f"Created new patient with MRN: {patient.mrn}, ID: {patient.id}")
        return patient
    
    def get_patient_by_id(self, patient_id: str) -> Optional[Patient]:
        """
        Retrieve a patient by their UUID
        
        Args:
            patient_id: UUID string of the patient
            
        Returns:
            Patient object if found, None otherwise
        """
        return self.session.query(Patient).filter(Patient.id == patient_id).first()
    
    def get_patient_by_mrn(self, mrn: str) -> Optional[Patient]:
        """
        Retrieve a patient by their Medical Record Number
        
        MRN is the primary identifier used in hospital systems,
        making this a common lookup method.
        
        Args:
            mrn: Medical Record Number string
            
        Returns:
            Patient object if found, None otherwise
        """
        return self.session.query(Patient).filter(Patient.mrn == mrn).first()
    
    def search_patients(self, search_term: str, limit: int = 20) -> List[Patient]:
        """
        Search for patients by name or MRN
        
        This method performs a flexible search across patient names
        and MRN for patient lookup functionality.
        
        Args:
            search_term: Text to search for (name or MRN)
            limit: Maximum number of results to return
            
        Returns:
            List of matching Patient objects
        """
        search_pattern = f"%{search_term}%"
        
        patients = self.session.query(Patient).filter(
            or_(
                Patient.first_name.ilike(search_pattern),
                Patient.last_name.ilike(search_pattern),
                Patient.mrn.ilike(search_pattern)
            )
        ).filter(Patient.is_active == True).limit(limit).all()
        
        logger.info(f"Found {len(patients)} patients matching '{search_term}'")
        return patients
    
    def update_patient(self, patient_id: str, update_data: Dict[str, Any]) -> Optional[Patient]:
        """
        Update patient information
        
        This method updates specified fields of a patient record
        while preserving existing data for unmodified fields.
        
        Args:
            patient_id: UUID of patient to update
            update_data: Dictionary of fields to update
            
        Returns:
            Updated Patient object if found, None otherwise
        """
        patient = self.get_patient_by_id(patient_id)
        if not patient:
            logger.warning(f"Patient not found for update: {patient_id}")
            return None
        
        # Update allowed fields
        allowed_fields = [
            "first_name", "last_name", "gender", "blood_type",
            "allergies", "chronic_conditions", "emergency_contact",
            "insurance_info"
        ]
        
        for field in allowed_fields:
            if field in update_data:
                setattr(patient, field, update_data[field])
        
        # Handle date of birth separately
        if "date_of_birth" in update_data:
            dob = update_data["date_of_birth"]
            if isinstance(dob, str):
                patient.date_of_birth = datetime.strptime(dob, "%Y-%m-%d").date()
            else:
                patient.date_of_birth = dob
        
        # Update timestamp and commit
        patient.updated_at = datetime.utcnow()
        self.session.commit()
        self.session.refresh(patient)
        
        logger.info(f"Updated patient: {patient_id}")
        return patient
    
    def deactivate_patient(self, patient_id: str) -> bool:
        """
        Soft delete a patient record
        
        Instead of deleting data, we mark it as inactive to preserve
        historical records for compliance and analytics.
        
        Args:
            patient_id: UUID of patient to deactivate
            
        Returns:
            True if successful, False if patient not found
        """
        patient = self.get_patient_by_id(patient_id)
        if not patient:
            return False
        
        patient.is_active = False
        patient.updated_at = datetime.utcnow()
        self.session.commit()
        
        logger.info(f"Deactivated patient: {patient_id}")
        return True
    
    def add_admission(self, patient_id: str, admission_data: Dict[str, Any]) -> Optional[Admission]:
        """
        Add a new admission record for a patient
        
        Args:
            patient_id: UUID of patient being admitted
            admission_data: Dictionary with admission information
            
        Returns:
            Created Admission object if patient exists, None otherwise
        """
        patient = self.get_patient_by_id(patient_id)
        if not patient:
            logger.warning(f"Cannot add admission - patient not found: {patient_id}")
            return None
        
        # Parse admission date
        admission_date = admission_data.get("admission_date")
        if isinstance(admission_date, str):
            admission_date = datetime.fromisoformat(admission_date)
        
        # Create admission record
        admission = Admission(
            patient_id=patient_id,
            admission_date=admission_date or datetime.utcnow(),
            admission_type=admission_data.get("admission_type", "Elective"),
            department=admission_data.get("department"),
            room_number=admission_data.get("room_number"),
            bed_number=admission_data.get("bed_number"),
            diagnosis=admission_data.get("diagnosis", {}),
            admitting_physician=admission_data.get("admitting_physician"),
            attending_physician=admission_data.get("attending_physician"),
            status="Active"
        )
        
        self.session.add(admission)
        self.session.commit()
        self.session.refresh(admission)
        
        logger.info(f"Created admission {admission.id} for patient {patient_id}")
        return admission
    
    def get_active_admission(self, patient_id: str) -> Optional[Admission]:
        """
        Get the current active admission for a patient
        
        Returns the most recent admission that hasn't been discharged.
        
        Args:
            patient_id: UUID of patient
            
        Returns:
            Active Admission object if exists, None otherwise
        """
        admission = self.session.query(Admission).filter(
            and_(
                Admission.patient_id == patient_id,
                Admission.status == "Active"
            )
        ).order_by(desc(Admission.admission_date)).first()
        
        return admission
    
    def get_patient_admissions(self, patient_id: str, limit: int = 10) -> List[Admission]:
        """
        Get all admissions for a patient ordered by date
        
        Args:
            patient_id: UUID of patient
            limit: Maximum number of admissions to return
            
        Returns:
            List of Admission objects in reverse chronological order
        """
        admissions = self.session.query(Admission).filter(
            Admission.patient_id == patient_id
        ).order_by(desc(Admission.admission_date)).limit(limit).all()
        
        return admissions
    
    def add_vital_signs(self, patient_id: str, vital_data: Dict[str, Any]) -> Optional[VitalSign]:
        """
        Record vital signs for a patient
        
        This method associates vital signs with the patient's
        current admission if available.
        
        Args:
            patient_id: UUID of patient
            vital_data: Dictionary with vital sign measurements
            
        Returns:
            Created VitalSign object if successful, None otherwise
        """
        patient = self.get_patient_by_id(patient_id)
        if not patient:
            logger.warning(f"Cannot add vitals - patient not found: {patient_id}")
            return None
        
        # Get current admission for context
        current_admission = self.get_active_admission(patient_id)
        
        # Parse timestamp
        timestamp = vital_data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # Create vital signs record
        vitals = VitalSign(
            patient_id=patient_id,
            admission_id=current_admission.id if current_admission else None,
            timestamp=timestamp or datetime.utcnow(),
            heart_rate=vital_data.get("heart_rate"),
            blood_pressure_systolic=vital_data.get("blood_pressure_systolic"),
            blood_pressure_diastolic=vital_data.get("blood_pressure_diastolic"),
            respiratory_rate=vital_data.get("respiratory_rate"),
            temperature=vital_data.get("temperature"),
            oxygen_saturation=vital_data.get("oxygen_saturation"),
            pain_level=vital_data.get("pain_level"),
            recorded_by=vital_data.get("recorded_by")
        )
        
        self.session.add(vitals)
        self.session.commit()
        self.session.refresh(vitals)
        
        # Check if vitals indicate critical condition
        if vitals._is_critical():
            logger.warning(f"Critical vitals recorded for patient {patient_id}: EWS={vitals._calculate_early_warning_score()}")
        
        logger.info(f"Added vital signs for patient {patient_id}")
        return vitals
    
    def get_recent_vitals(self, patient_id: str, hours: int = 24) -> List[VitalSign]:
        """
        Get recent vital signs for a patient
        
        Args:
            patient_id: UUID of patient
            hours: Number of hours to look back
            
        Returns:
            List of VitalSign objects within time window
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        vitals = self.session.query(VitalSign).filter(
            and_(
                VitalSign.patient_id == patient_id,
                VitalSign.timestamp >= cutoff_time
            )
        ).order_by(desc(VitalSign.timestamp)).all()
        
        return vitals
    
    def add_lab_result(self, patient_id: str, lab_data: Dict[str, Any]) -> Optional[LabResult]:
        """
        Add a laboratory result for a patient
        
        Args:
            patient_id: UUID of patient
            lab_data: Dictionary with lab result information
            
        Returns:
            Created LabResult object if successful, None otherwise
        """
        patient = self.get_patient_by_id(patient_id)
        if not patient:
            logger.warning(f"Cannot add lab result - patient not found: {patient_id}")
            return None
        
        # Get current admission for context
        current_admission = self.get_active_admission(patient_id)
        
        # Parse dates
        collected_at = lab_data.get("collected_at")
        if isinstance(collected_at, str):
            collected_at = datetime.fromisoformat(collected_at)
        
        resulted_at = lab_data.get("resulted_at")
        if isinstance(resulted_at, str):
            resulted_at = datetime.fromisoformat(resulted_at)
        
        # Create lab result
        lab_result = LabResult(
            patient_id=patient_id,
            admission_id=current_admission.id if current_admission else None,
            test_name=lab_data.get("test_name"),
            test_code=lab_data.get("test_code"),
            test_category=lab_data.get("test_category"),
            result_value=lab_data.get("result_value"),
            result_text=lab_data.get("result_text"),
            unit=lab_data.get("unit"),
            reference_range=lab_data.get("reference_range"),
            collected_at=collected_at,
            resulted_at=resulted_at or datetime.utcnow(),
            ordered_by=lab_data.get("ordered_by")
        )
        
        # Set interpretation based on result
        lab_result.set_interpretation()
        
        self.session.add(lab_result)
        self.session.commit()
        self.session.refresh(lab_result)
        
        # Log critical results
        if lab_result.interpretation == "Critical":
            logger.warning(f"Critical lab result for patient {patient_id}: {lab_result.test_name}={lab_result.result_value}")
        
        logger.info(f"Added lab result for patient {patient_id}: {lab_result.test_name}")
        return lab_result
    
    def get_lab_results(self, patient_id: str, limit: int = 50) -> List[LabResult]:
        """
        Get recent lab results for a patient
        
        Args:
            patient_id: UUID of patient
            limit: Maximum number of results to return
            
        Returns:
            List of LabResult objects ordered by resulted date
        """
        results = self.session.query(LabResult).filter(
            LabResult.patient_id == patient_id
        ).order_by(desc(LabResult.resulted_at)).limit(limit).all()
        
        return results
    
    def add_clinical_note(self, patient_id: str, note_data: Dict[str, Any]) -> Optional[ClinicalNote]:
        """
        Add a clinical note for a patient
        
        Args:
            patient_id: UUID of patient
            note_data: Dictionary with clinical note information
            
        Returns:
            Created ClinicalNote object if successful, None otherwise
        """
        patient = self.get_patient_by_id(patient_id)
        if not patient:
            logger.warning(f"Cannot add clinical note - patient not found: {patient_id}")
            return None
        
        # Get current admission for context
        current_admission = self.get_active_admission(patient_id)
        
        # Parse note date
        note_date = note_data.get("note_date")
        if isinstance(note_date, str):
            note_date = datetime.fromisoformat(note_date)
        
        # Create clinical note
        note = ClinicalNote(
            patient_id=patient_id,
            admission_id=current_admission.id if current_admission else None,
            note_type=note_data.get("note_type", "Progress"),
            title=note_data.get("title"),
            content=note_data.get("content"),
            author=note_data.get("author"),
            author_role=note_data.get("author_role"),
            note_date=note_date or datetime.utcnow(),
            is_signed=note_data.get("is_signed", False)
        )
        
        self.session.add(note)
        self.session.commit()
        self.session.refresh(note)
        
        logger.info(f"Added clinical note for patient {patient_id}: {note.note_type}")
        return note
    
    def get_clinical_notes(self, patient_id: str, limit: int = 50) -> List[ClinicalNote]:
        """
        Get clinical notes for a patient
        
        Args:
            patient_id: UUID of patient
            limit: Maximum number of notes to return
            
        Returns:
            List of ClinicalNote objects ordered by note date
        """
        notes = self.session.query(ClinicalNote).filter(
            ClinicalNote.patient_id == patient_id
        ).order_by(desc(ClinicalNote.note_date)).limit(limit).all()
        
        return notes
    
    def get_patient_summary(self, patient_id: str) -> Dict[str, Any]:
        """
        Get comprehensive patient summary including recent data
        
        This method aggregates multiple data sources to provide a
        complete clinical picture of the patient.
        
        Args:
            patient_id: UUID of patient
            
        Returns:
            Dictionary with comprehensive patient information
        """
        patient = self.get_patient_by_id(patient_id)
        if not patient:
            return {}
        
        current_admission = self.get_active_admission(patient_id)
        recent_vitals = self.get_recent_vitals(patient_id, hours=24)
        recent_labs = self.get_lab_results(patient_id, limit=20)
        recent_notes = self.get_clinical_notes(patient_id, limit=10)
        admissions = self.get_patient_admissions(patient_id, limit=5)
        
        # Calculate summary statistics
        latest_vitals = recent_vitals[0] if recent_vitals else None
        
        summary = {
            "patient": patient.to_dict(),
            "current_admission": current_admission.to_dict() if current_admission else None,
            "latest_vitals": latest_vitals.to_dict() if latest_vitals else None,
            "recent_vitals": [v.to_dict() for v in recent_vitals[:10]],
            "recent_labs": [l.to_dict() for l in recent_labs],
            "recent_notes": [n.to_dict() for n in recent_notes],
            "admission_history": [a.to_dict() for a in admissions],
            "statistics": {
                "total_admissions": len(admissions),
                "total_lab_results": len(recent_labs),
                "last_visit": admissions[0].admission_date.isoformat() if admissions else None,
                "critical_alerts": self._count_critical_alerts(recent_labs, recent_vitals)
            }
        }
        
        return summary
    
    def _count_critical_alerts(self, labs: List[LabResult], vitals: List[VitalSign]) -> int:
        """
        Count critical alerts from labs and vitals
        
        Args:
            labs: List of lab results
            vitals: List of vital signs
            
        Returns:
            Number of critical findings
        """
        count = 0
        
        for lab in labs:
            if lab.interpretation == "Critical":
                count += 1
        
        for vital in vitals:
            if vital._is_critical():
                count += 1
        
        return count
    
    def get_patient_statistics(self) -> Dict[str, Any]:
        """
        Get aggregate statistics across all patients
        
        This method provides population-level statistics for
        hospital analytics and reporting.
        
        Returns:
            Dictionary with patient population statistics
        """
        total_patients = self.session.query(func.count(Patient.id)).filter(Patient.is_active == True).scalar()
        
        active_admissions = self.session.query(func.count(Admission.id)).filter(Admission.status == "Active").scalar()
        
        # Gender distribution
        male_count = self.session.query(func.count(Patient.id)).filter(Patient.gender == "M", Patient.is_active == True).scalar()
        female_count = self.session.query(func.count(Patient.id)).filter(Patient.gender == "F", Patient.is_active == True).scalar()
        
        # Age distribution (simplified)
        today = date.today()
        age_groups = {
            "0-18": 0,
            "19-35": 0,
            "36-50": 0,
            "51-65": 0,
            "65+": 0
        }
        
        patients = self.session.query(Patient).filter(Patient.is_active == True).all()
        for patient in patients:
            age = patient._calculate_age()
            if age:
                if age <= 18:
                    age_groups["0-18"] += 1
                elif age <= 35:
                    age_groups["19-35"] += 1
                elif age <= 50:
                    age_groups["36-50"] += 1
                elif age <= 65:
                    age_groups["51-65"] += 1
                else:
                    age_groups["65+"] += 1
        
        return {
            "total_patients": total_patients,
            "active_admissions": active_admissions,
            "gender_distribution": {
                "male": male_count,
                "female": female_count,
                "other": total_patients - male_count - female_count
            },
            "age_distribution": age_groups,
            "occupancy_rate": round((active_admissions / total_patients) * 100, 1) if total_patients > 0 else 0
        }