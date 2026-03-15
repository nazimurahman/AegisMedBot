"""
Patients routes module for managing patient data and interactions.
This module provides endpoints for patient information, risk assessment, and care coordination.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime, date
import logging
import uuid

from ...core.config import settings
from ...core.security import get_current_user, require_role
from ...core.database import get_db
from ...services.audit_service import AuditService
from ...models.schemas.response import ResponseModel, ErrorResponseModel
from ...models.schemas.patient import PatientCreate, PatientUpdate, PatientResponse
from ...models.enums import PatientStatus, AdmissionType, RiskLevel

# Import database repositories
from database.repositories.patient_repo import PatientRepository
from database.repositories.admission_repo import AdmissionRepository

# Import agent for risk assessment
from agents.risk_agent.risk_predictor import RiskPredictor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/patients", tags=["Patients"])

# Pydantic models for request/response

class PatientSearchParams(BaseModel):
    """
    Search parameters for patient queries.
    """
    
    query: Optional[str] = Field(None, description="Search query (name, MRN, etc.)")
    status: Optional[PatientStatus] = Field(None, description="Filter by status")
    department: Optional[str] = Field(None, description="Filter by department")
    attending_physician: Optional[str] = Field(None, description="Filter by physician")
    admit_date_from: Optional[date] = Field(None, description="Admission date start")
    admit_date_to: Optional[date] = Field(None, description="Admission date end")
    risk_level: Optional[RiskLevel] = Field(None, description="Filter by risk level")
    limit: int = Field(50, ge=1, le=200, description="Maximum results")
    offset: int = Field(0, ge=0, description="Pagination offset")

class VitalSignsCreate(BaseModel):
    """
    Model for recording vital signs.
    """
    
    patient_id: str = Field(..., description="Patient ID")
    heart_rate: Optional[int] = Field(None, ge=0, le=300)
    blood_pressure_systolic: Optional[int] = Field(None, ge=0, le=300)
    blood_pressure_diastolic: Optional[int] = Field(None, ge=0, le=200)
    respiratory_rate: Optional[int] = Field(None, ge=0, le=100)
    temperature: Optional[float] = Field(None, ge=30.0, le=45.0)
    oxygen_saturation: Optional[int] = Field(None, ge=0, le=100)
    pain_level: Optional[int] = Field(None, ge=0, le=10)
    
    @validator('blood_pressure_systolic')
    def validate_blood_pressure(cls, v, values):
        """Validate blood pressure relationship."""
        if v and 'blood_pressure_diastolic' in values and values['blood_pressure_diastolic']:
            if v <= values['blood_pressure_diastolic']:
                raise ValueError('Systolic pressure must be greater than diastolic')
        return v

class RiskAssessmentRequest(BaseModel):
    """
    Request model for patient risk assessment.
    """
    
    patient_id: str = Field(..., description="Patient ID")
    assessment_type: str = Field(..., description="Type: readmission, complication, mortality, icu")
    horizon_hours: int = Field(24, ge=1, le=720, description="Prediction horizon in hours")

class RiskAssessmentResponse(BaseModel):
    """
    Response model for risk assessment.
    """
    
    patient_id: str = Field(..., description="Patient ID")
    assessment_id: str = Field(..., description="Unique assessment ID")
    risk_level: RiskLevel = Field(..., description="Overall risk level")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score 0-1")
    probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability")
    contributing_factors: List[Dict[str, Any]] = Field(..., description="Risk factors")
    recommendations: List[str] = Field(..., description="Clinical recommendations")
    assessed_at: datetime = Field(..., description="Assessment timestamp")
    valid_until: datetime = Field(..., description="When assessment expires")

# Dependency to get repositories
async def get_patient_repository(db=Depends(get_db)) -> PatientRepository:
    return PatientRepository(db)

async def get_admission_repository(db=Depends(get_db)) -> AdmissionRepository:
    return AdmissionRepository(db)

@router.post("/", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
async def create_patient(
    patient: PatientCreate,
    patient_repo: PatientRepository = Depends(get_patient_repository),
    current_user: Dict[str, Any] = Depends(require_role(["admin", "doctor", "nurse"])),
    audit_service: AuditService = Depends()
):
    """
    Create a new patient record.
    Requires clinical role.
    """
    
    logger.info(f"Creating new patient by user {current_user['id']}")
    
    try:
        # Check if patient with same MRN already exists
        existing = await patient_repo.get_by_mrn(patient.mrn)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Patient with MRN {patient.mrn} already exists"
            )
        
        # Create patient
        created_patient = await patient_repo.create(patient)
        
        # Log audit
        await audit_service.log_patient_creation(
            user_id=current_user["id"],
            patient_id=created_patient.id,
            timestamp=datetime.now()
        )
        
        return PatientResponse.from_orm(created_patient)
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error creating patient: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating patient record"
        )

@router.get("/search", response_model=List[PatientResponse])
async def search_patients(
    params: PatientSearchParams = Depends(),
    patient_repo: PatientRepository = Depends(get_patient_repository),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Search for patients with filtering.
    Available to all authenticated users.
    """
    
    logger.info(f"Patient search by user {current_user['id']}: {params.query}")
    
    try:
        # Build search filters from params
        filters = params.dict(exclude_none=True, exclude={"limit", "offset"})
        
        # Search patients
        patients = await patient_repo.search(
            filters=filters,
            limit=params.limit,
            offset=params.offset
        )
        
        return [PatientResponse.from_orm(p) for p in patients]
        
    except Exception as e:
        logger.error(f"Error searching patients: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error searching patients")

@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: str,
    patient_repo: PatientRepository = Depends(get_patient_repository),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get detailed patient information by ID.
    """
    
    logger.info(f"Retrieving patient {patient_id} for user {current_user['id']}")
    
    try:
        patient = await patient_repo.get_by_id(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        return PatientResponse.from_orm(patient)
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error retrieving patient: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving patient")

@router.patch("/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: str,
    updates: PatientUpdate,
    patient_repo: PatientRepository = Depends(get_patient_repository),
    current_user: Dict[str, Any] = Depends(require_role(["admin", "doctor", "nurse"])),
    audit_service: AuditService = Depends()
):
    """
    Update patient information.
    Requires clinical role.
    """
    
    logger.info(f"Updating patient {patient_id} by user {current_user['id']}")
    
    try:
        # Check if patient exists
        existing = await patient_repo.get_by_id(patient_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Apply updates
        update_dict = updates.dict(exclude_unset=True)
        updated_patient = await patient_repo.update(patient_id, update_dict)
        
        # Log audit
        await audit_service.log_patient_update(
            user_id=current_user["id"],
            patient_id=patient_id,
            changes=update_dict,
            timestamp=datetime.now()
        )
        
        return PatientResponse.from_orm(updated_patient)
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error updating patient: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error updating patient")

@router.post("/{patient_id}/vitals")
async def record_vitals(
    patient_id: str,
    vitals: VitalSignsCreate,
    patient_repo: PatientRepository = Depends(get_patient_repository),
    current_user: Dict[str, Any] = Depends(require_role(["doctor", "nurse"])),
    audit_service: AuditService = Depends()
):
    """
    Record vital signs for a patient.
    Requires clinical role.
    """
    
    logger.info(f"Recording vitals for patient {patient_id} by user {current_user['id']}")
    
    try:
        # Verify patient exists
        patient = await patient_repo.get_by_id(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Ensure patient ID matches
        vitals.patient_id = patient_id
        
        # Record vitals
        vital_record = await patient_repo.add_vitals(vitals)
        
        # Check for abnormal vitals
        alerts = self._check_abnormal_vitals(vitals)
        if alerts:
            # Send alerts to relevant staff
            notification_service = NotificationService()
            await notification_service.send_vital_alerts(
                patient_id=patient_id,
                alerts=alerts,
                recorded_by=current_user["id"]
            )
        
        # Log audit
        await audit_service.log_vitals_recorded(
            user_id=current_user["id"],
            patient_id=patient_id,
            vital_id=vital_record.id,
            timestamp=datetime.now()
        )
        
        return ResponseModel(
            status="success",
            message="Vital signs recorded",
            data={
                "vital_id": vital_record.id,
                "alerts": alerts
            }
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error recording vitals: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error recording vital signs")

@router.post("/{patient_id}/risk-assessment", response_model=RiskAssessmentResponse)
async def assess_patient_risk(
    patient_id: str,
    request: RiskAssessmentRequest,
    patient_repo: PatientRepository = Depends(get_patient_repository),
    admission_repo: AdmissionRepository = Depends(get_admission_repository),
    current_user: Dict[str, Any] = Depends(require_role(["doctor", "risk_manager"])),
    audit_service: AuditService = Depends()
):
    """
    Perform AI-based risk assessment for a patient.
    Requires appropriate clinical role.
    """
    
    logger.info(f"Risk assessment for patient {patient_id} by user {current_user['id']}")
    
    try:
        # Verify patient exists
        patient = await patient_repo.get_by_id(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Get patient's current admission if any
        current_admission = await admission_repo.get_current_admission(patient_id)
        
        # Get recent vitals and labs
        recent_vitals = await patient_repo.get_recent_vitals(patient_id, hours=24)
        recent_labs = await patient_repo.get_recent_labs(patient_id, hours=72)
        
        # Prepare features for risk model
        features = {
            "demographics": {
                "age": patient.age,
                "gender": patient.gender,
                "has_chronic_conditions": len(patient.chronic_conditions) > 0
            },
            "vitals": [v.dict() for v in recent_vitals],
            "labs": [l.dict() for l in recent_labs],
            "admission": current_admission.dict() if current_admission else None,
            "assessment_type": request.assessment_type,
            "horizon_hours": request.horizon_hours
        }
        
        # Call risk prediction agent
        risk_predictor = RiskPredictor()
        assessment = await risk_predictor.predict_risk(features)
        
        # Store assessment result
        assessment_id = str(uuid.uuid4())
        await patient_repo.store_risk_assessment(
            patient_id=patient_id,
            assessment_id=assessment_id,
            assessment_data=assessment
        )
        
        # Log audit
        await audit_service.log_risk_assessment(
            user_id=current_user["id"],
            patient_id=patient_id,
            assessment_id=assessment_id,
            risk_level=assessment["risk_level"],
            timestamp=datetime.now()
        )
        
        # If high risk, notify care team
        if assessment["risk_level"] in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            notification_service = NotificationService()
            await notification_service.send_risk_alert(
                patient_id=patient_id,
                risk_level=assessment["risk_level"],
                risk_score=assessment["risk_score"],
                recommendations=assessment["recommendations"]
            )
        
        return RiskAssessmentResponse(
            patient_id=patient_id,
            assessment_id=assessment_id,
            risk_level=RiskLevel(assessment["risk_level"]),
            risk_score=assessment["risk_score"],
            probability=assessment["probability"],
            contributing_factors=assessment["contributing_factors"],
            recommendations=assessment["recommendations"],
            assessed_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=request.horizon_hours)
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error in risk assessment: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error performing risk assessment")

@router.get("/{patient_id}/timeline")
async def get_patient_timeline(
    patient_id: str,
    days: int = Query(30, ge=1, le=365),
    patient_repo: PatientRepository = Depends(get_patient_repository),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get comprehensive patient timeline including admissions, vitals, labs, and events.
    """
    
    logger.info(f"Getting timeline for patient {patient_id}")
    
    try:
        # Verify patient exists
        patient = await patient_repo.get_by_id(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Get all data for timeline
        since = datetime.now() - timedelta(days=days)
        
        admissions = await patient_repo.get_admissions(patient_id, since=since)
        vitals = await patient_repo.get_vitals(patient_id, since=since)
        labs = await patient_repo.get_labs(patient_id, since=since)
        medications = await patient_repo.get_medications(patient_id, since=since)
        notes = await patient_repo.get_notes(patient_id, since=since)
        
        # Combine and sort all events
        timeline = []
        
        for admission in admissions:
            timeline.append({
                "type": "admission",
                "date": admission.admission_date,
                "data": admission.dict(),
                "icon": "🏥"
            })
        
        for vital in vitals:
            timeline.append({
                "type": "vitals",
                "date": vital.timestamp,
                "data": vital.dict(),
                "icon": "❤️"
            })
        
        for lab in labs:
            timeline.append({
                "type": "lab",
                "date": lab.resulted_at,
                "data": lab.dict(),
                "icon": "🔬"
            })
        
        # Sort by date descending
        timeline.sort(key=lambda x: x["date"], reverse=True)
        
        return {
            "patient_id": patient_id,
            "patient_name": f"{patient.first_name} {patient.last_name}",
            "timeline": timeline,
            "summary": {
                "total_admissions": len(admissions),
                "total_vitals": len(vitals),
                "total_labs": len(labs),
                "period_days": days
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting patient timeline: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving patient timeline")

def _check_abnormal_vitals(self, vitals: VitalSignsCreate) -> List[Dict[str, Any]]:
    """
    Check for abnormal vital signs based on clinical thresholds.
    
    Args:
        vitals: Vital signs to check
        
    Returns:
        List of alerts for abnormal values
    """
    alerts = []
    
    # Define thresholds (simplified - would use evidence-based guidelines)
    thresholds = {
        "heart_rate": {"low": 60, "high": 100, "critical_low": 40, "critical_high": 140},
        "blood_pressure_systolic": {"low": 90, "high": 180, "critical_low": 70, "critical_high": 220},
        "respiratory_rate": {"low": 12, "high": 20, "critical_low": 8, "critical_high": 30},
        "temperature": {"low": 36.0, "high": 38.0, "critical_low": 35.0, "critical_high": 39.5},
        "oxygen_saturation": {"low": 95, "critical_low": 90}
    }
    
    # Check each vital sign
    if vitals.heart_rate:
        hr = vitals.heart_rate
        if hr < thresholds["heart_rate"]["critical_low"]:
            alerts.append({"vital": "heart_rate", "value": hr, "severity": "critical", "message": "Critical bradycardia"})
        elif hr < thresholds["heart_rate"]["low"]:
            alerts.append({"vital": "heart_rate", "value": hr, "severity": "warning", "message": "Bradycardia"})
        elif hr > thresholds["heart_rate"]["critical_high"]:
            alerts.append({"vital": "heart_rate", "value": hr, "severity": "critical", "message": "Critical tachycardia"})
        elif hr > thresholds["heart_rate"]["high"]:
            alerts.append({"vital": "heart_rate", "value": hr, "severity": "warning", "message": "Tachycardia"})
    
    # Check blood pressure
    if vitals.blood_pressure_systolic:
        sbp = vitals.blood_pressure_systolic
        if sbp < thresholds["blood_pressure_systolic"]["critical_low"]:
            alerts.append({"vital": "blood_pressure", "value": sbp, "severity": "critical", "message": "Critical hypotension"})
        elif sbp < thresholds["blood_pressure_systolic"]["low"]:
            alerts.append({"vital": "blood_pressure", "value": sbp, "severity": "warning", "message": "Hypotension"})
        elif sbp > thresholds["blood_pressure_systolic"]["critical_high"]:
            alerts.append({"vital": "blood_pressure", "value": sbp, "severity": "critical", "message": "Hypertensive crisis"})
        elif sbp > thresholds["blood_pressure_systolic"]["high"]:
            alerts.append({"vital": "blood_pressure", "value": sbp, "severity": "warning", "message": "Hypertension"})
    
    return alerts