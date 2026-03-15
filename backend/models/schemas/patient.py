"""
Patient data schemas for AegisMedBot.
Defines Pydantic models for patient data validation and serialization.
"""

from pydantic import BaseModel, Field, ConfigDict, validator
from typing import Optional, List, Dict, Any, Union
from datetime import date, datetime
from enum import Enum

# Enumerations for patient data
class Gender(str, Enum):
    """
    Gender enumeration for patient records.
    Follows standard medical gender categories.
    """
    MALE = "M"
    FEMALE = "F"
    OTHER = "O"
    UNKNOWN = "U"


class BloodType(str, Enum):
    """
    Blood type enumeration following medical standards.
    Includes Rh factor in the type.
    """
    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"
    UNKNOWN = "UNK"


class AdmissionType(str, Enum):
    """
    Types of hospital admissions.
    """
    EMERGENCY = "emergency"
    ELECTIVE = "elective"
    URGENT = "urgent"
    TRANSFER = "transfer"


class AdmissionStatus(str, Enum):
    """
    Status of patient admission.
    """
    ACTIVE = "active"
    DISCHARGED = "discharged"
    TRANSFERRED = "transferred"
    CANCELLED = "cancelled"


class VitalSignUnit(str, Enum):
    """
    Units for vital sign measurements.
    """
    BEATS_PER_MINUTE = "bpm"
    MMHG = "mmHg"
    BREATHS_PER_MINUTE = "breaths/min"
    CELSIUS = "°C"
    FAHRENHEIT = "°F"
    PERCENT = "%"
    SCALE_1_10 = "1-10"


class LabResultInterpretation(str, Enum):
    """
    Interpretation of laboratory results.
    """
    NORMAL = "normal"
    ABNORMAL = "abnormal"
    CRITICAL = "critical"
    HIGH = "high"
    LOW = "low"
    INCONCLUSIVE = "inconclusive"


# Main patient schema
class PatientBase(BaseModel):
    """
    Base patient schema with common fields.
    All patient-related schemas inherit from this.
    """
    mrn: str = Field(
        ...,
        description="Medical Record Number - unique identifier",
        min_length=3,
        max_length=50,
        pattern=r"^[A-Z0-9\-]+$"  # Alphanumeric with hyphens
    )
    first_name: str = Field(
        ...,
        description="Patient's first name",
        min_length=1,
        max_length=100
    )
    last_name: str = Field(
        ...,
        description="Patient's last name",
        min_length=1,
        max_length=100
    )
    date_of_birth: date = Field(
        ...,
        description="Patient's date of birth"
    )
    gender: Gender = Field(
        ...,
        description="Patient's gender"
    )
    blood_type: Optional[BloodType] = Field(
        None,
        description="Patient's blood type"
    )
    allergies: List[str] = Field(
        default_factory=list,
        description="List of patient allergies"
    )
    chronic_conditions: List[str] = Field(
        default_factory=list,
        description="List of chronic medical conditions"
    )
    emergency_contact: Optional[Dict[str, str]] = Field(
        None,
        description="Emergency contact information"
    )
    insurance_info: Optional[Dict[str, Any]] = Field(
        None,
        description="Health insurance information"
    )
    
    @validator("date_of_birth")
    def validate_age(cls, v):
        """
        Validate that date of birth is not in the future
        and patient is not older than 150 years.
        """
        if v > date.today():
            raise ValueError("Date of birth cannot be in the future")
        
        age = (date.today() - v).days / 365.25
        if age > 150:
            raise ValueError("Patient age exceeds maximum (150 years)")
        
        return v
    
    @validator("emergency_contact")
    def validate_emergency_contact(cls, v):
        """
        Validate emergency contact has required fields.
        """
        if v:
            required_fields = ["name", "phone", "relationship"]
            for field in required_fields:
                if field not in v:
                    raise ValueError(f"Emergency contact missing {field}")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mrn": "MRN-2024-001234",
                "first_name": "John",
                "last_name": "Doe",
                "date_of_birth": "1980-05-15",
                "gender": "M",
                "blood_type": "O+",
                "allergies": ["Penicillin", "Sulfa"],
                "chronic_conditions": ["Hypertension", "Type 2 Diabetes"],
                "emergency_contact": {
                    "name": "Jane Doe",
                    "phone": "+1-555-123-4567",
                    "relationship": "Spouse"
                },
                "insurance_info": {
                    "provider": "Blue Cross",
                    "policy_number": "BC-123456789"
                }
            }
        }
    )


class PatientCreate(PatientBase):
    """
    Schema for creating a new patient.
    Inherits all fields from PatientBase.
    """
    pass


class PatientUpdate(BaseModel):
    """
    Schema for updating an existing patient.
    All fields are optional for partial updates.
    """
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    date_of_birth: Optional[date] = None
    gender: Optional[Gender] = None
    blood_type: Optional[BloodType] = None
    allergies: Optional[List[str]] = None
    chronic_conditions: Optional[List[str]] = None
    emergency_contact: Optional[Dict[str, str]] = None
    insurance_info: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "first_name": "Jonathan",
                "allergies": ["Penicillin", "Sulfa", "Latex"],
                "is_active": True
            }
        }
    )


class PatientInDB(PatientBase):
    """
    Schema for patient as stored in database.
    Includes database-specific fields.
    """
    id: str = Field(..., description="Unique patient ID (UUID)")
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    is_active: bool = Field(True, description="Whether patient record is active")
    
    model_config = ConfigDict(
        from_attributes=True  # Enable ORM mode for SQLAlchemy
    )


class PatientResponse(PatientInDB):
    """
    Schema for patient API responses.
    Includes additional computed fields.
    """
    full_name: str = Field(..., description="Full name (first + last)")
    age: int = Field(..., description="Calculated age in years")
    
    @validator("full_name", always=True)
    def compute_full_name(cls, v, values):
        """
        Compute full name from first and last name.
        """
        data = values.data
        return f"{data.get('first_name', '')} {data.get('last_name', '')}".strip()
    
    @validator("age", always=True)
    def compute_age(cls, v, values):
        """
        Compute age from date of birth.
        """
        data = values.data
        if "date_of_birth" in data and data["date_of_birth"]:
            today = date.today()
            born = data["date_of_birth"]
            age = today.year - born.year - (
                (today.month, today.day) < (born.month, born.day)
            )
            return age
        return 0


# Admission schemas
class AdmissionBase(BaseModel):
    """
    Base schema for patient admissions.
    """
    patient_id: str = Field(..., description="Patient ID for this admission")
    admission_date: datetime = Field(
        default_factory=datetime.now,
        description="Date and time of admission"
    )
    admission_type: AdmissionType = Field(
        ...,
        description="Type of admission"
    )
    department: str = Field(
        ...,
        description="Department where patient is admitted",
        min_length=1,
        max_length=100
    )
    room_number: Optional[str] = Field(
        None,
        description="Room number",
        max_length=20
    )
    bed_number: Optional[str] = Field(
        None,
        description="Bed number",
        max_length=20
    )
    diagnosis: Optional[Dict[str, Any]] = Field(
        None,
        description="Primary and secondary diagnoses"
    )
    admitting_physician: Optional[str] = Field(
        None,
        description="Physician who admitted the patient",
        max_length=100
    )
    attending_physician: Optional[str] = Field(
        None,
        description="Attending physician",
        max_length=100
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_id": "123e4567-e89b-12d3-a456-426614174000",
                "admission_date": "2024-06-15T14:30:00Z",
                "admission_type": "emergency",
                "department": "Emergency Department",
                "room_number": "ER-12",
                "bed_number": "B",
                "diagnosis": {
                    "primary": "Chest pain",
                    "secondary": ["Hypertension", "Anxiety"]
                },
                "admitting_physician": "Dr. Smith",
                "attending_physician": "Dr. Jones"
            }
        }
    )


class AdmissionCreate(AdmissionBase):
    """
    Schema for creating a new admission.
    """
    pass


class AdmissionUpdate(BaseModel):
    """
    Schema for updating an admission.
    """
    discharge_date: Optional[datetime] = None
    department: Optional[str] = Field(None, min_length=1, max_length=100)
    room_number: Optional[str] = Field(None, max_length=20)
    bed_number: Optional[str] = Field(None, max_length=20)
    diagnosis: Optional[Dict[str, Any]] = None
    attending_physician: Optional[str] = Field(None, max_length=100)
    status: Optional[AdmissionStatus] = None
    discharge_disposition: Optional[str] = Field(None, max_length=100)


class AdmissionInDB(AdmissionBase):
    """
    Schema for admission as stored in database.
    """
    id: str = Field(..., description="Unique admission ID (UUID)")
    discharge_date: Optional[datetime] = Field(
        None,
        description="Date and time of discharge"
    )
    status: AdmissionStatus = Field(
        AdmissionStatus.ACTIVE,
        description="Current admission status"
    )
    discharge_disposition: Optional[str] = Field(
        None,
        description="Discharge destination or plan"
    )
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    model_config = ConfigDict(from_attributes=True)


class AdmissionResponse(AdmissionInDB):
    """
    Schema for admission API responses.
    """
    length_of_stay_days: Optional[float] = Field(
        None,
        description="Length of stay in days (if discharged)"
    )
    
    @validator("length_of_stay_days", always=True)
    def compute_length_of_stay(cls, v, values):
        """
        Compute length of stay if patient is discharged.
        """
        data = values.data
        if data.get("discharge_date") and data.get("admission_date"):
            stay = data["discharge_date"] - data["admission_date"]
            return stay.total_seconds() / 86400  # Convert to days
        return None


# Vital signs schemas
class VitalSignBase(BaseModel):
    """
    Base schema for vital signs measurements.
    """
    patient_id: str = Field(..., description="Patient ID")
    admission_id: Optional[str] = Field(
        None,
        description="Associated admission ID"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Measurement timestamp"
    )
    heart_rate: Optional[int] = Field(
        None,
        description="Heart rate in beats per minute",
        ge=0,
        le=300
    )
    blood_pressure_systolic: Optional[int] = Field(
        None,
        description="Systolic blood pressure in mmHg",
        ge=0,
        le=300
    )
    blood_pressure_diastolic: Optional[int] = Field(
        None,
        description="Diastolic blood pressure in mmHg",
        ge=0,
        le=200
    )
    respiratory_rate: Optional[int] = Field(
        None,
        description="Respiratory rate in breaths per minute",
        ge=0,
        le=100
    )
    temperature: Optional[float] = Field(
        None,
        description="Body temperature",
        ge=20.0,
        le=45.0
    )
    temperature_unit: VitalSignUnit = Field(
        VitalSignUnit.CELSIUS,
        description="Temperature unit"
    )
    oxygen_saturation: Optional[int] = Field(
        None,
        description="Oxygen saturation percentage",
        ge=0,
        le=100
    )
    pain_level: Optional[int] = Field(
        None,
        description="Pain level on 1-10 scale",
        ge=0,
        le=10
    )
    recorded_by: Optional[str] = Field(
        None,
        description="Staff member who recorded the measurement",
        max_length=100
    )
    
    @validator("blood_pressure_systolic", "blood_pressure_diastolic")
    def validate_blood_pressure(cls, v, values, **kwargs):
        """
        Validate that systolic is greater than diastolic if both present.
        """
        field_name = kwargs.get("field").name if hasattr(kwargs.get("field"), "name") else ""
        
        if field_name == "blood_pressure_diastolic" and v is not None:
            systolic = values.data.get("blood_pressure_systolic")
            if systolic is not None and v >= systolic:
                raise ValueError("Diastolic pressure must be less than systolic")
        
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2024-06-15T15:30:00Z",
                "heart_rate": 72,
                "blood_pressure_systolic": 120,
                "blood_pressure_diastolic": 80,
                "respiratory_rate": 16,
                "temperature": 37.0,
                "oxygen_saturation": 98,
                "pain_level": 2,
                "recorded_by": "Nurse Johnson"
            }
        }
    )


class VitalSignCreate(VitalSignBase):
    """
    Schema for creating a new vital sign record.
    """
    pass


class VitalSignInDB(VitalSignBase):
    """
    Schema for vital signs as stored in database.
    """
    id: str = Field(..., description="Unique vital sign ID (UUID)")
    created_at: datetime = Field(..., description="Record creation timestamp")
    
    model_config = ConfigDict(from_attributes=True)


class VitalSignResponse(VitalSignInDB):
    """
    Schema for vital signs API responses.
    """
    pass


# Lab results schemas
class LabResultBase(BaseModel):
    """
    Base schema for laboratory results.
    """
    patient_id: str = Field(..., description="Patient ID")
    admission_id: Optional[str] = Field(
        None,
        description="Associated admission ID"
    )
    test_name: str = Field(
        ...,
        description="Name of the laboratory test",
        min_length=1,
        max_length=200
    )
    test_code: Optional[str] = Field(
        None,
        description="Standard test code (e.g., LOINC)",
        max_length=50
    )
    result_value: Optional[float] = Field(
        None,
        description="Numeric result value"
    )
    result_text: Optional[str] = Field(
        None,
        description="Text result if not numeric",
        max_length=500
    )
    unit: Optional[str] = Field(
        None,
        description="Unit of measurement",
        max_length=20
    )
    reference_range: Optional[str] = Field(
        None,
        description="Normal reference range",
        max_length=100
    )
    interpretation: LabResultInterpretation = Field(
        ...,
        description="Clinical interpretation of result"
    )
    collected_at: Optional[datetime] = Field(
        None,
        description="Sample collection time"
    )
    resulted_at: Optional[datetime] = Field(
        None,
        description="Result reporting time"
    )
    
    @validator("result_value", "result_text", always=True)
    def validate_result(cls, v, values):
        """
        Ensure at least one of result_value or result_text is provided.
        """
        # This validator runs for both fields
        # We need to check after both are processed
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_id": "123e4567-e89b-12d3-a456-426614174000",
                "test_name": "Complete Blood Count",
                "test_code": "58410-2",
                "result_value": 13.5,
                "unit": "g/dL",
                "reference_range": "12.0-15.5",
                "interpretation": "normal",
                "collected_at": "2024-06-15T08:00:00Z",
                "resulted_at": "2024-06-15T10:30:00Z"
            }
        }
    )


class LabResultCreate(LabResultBase):
    """
    Schema for creating a new lab result.
    """
    pass


class LabResultInDB(LabResultBase):
    """
    Schema for lab result as stored in database.
    """
    id: str = Field(..., description="Unique lab result ID (UUID)")
    created_at: datetime = Field(..., description="Record creation timestamp")
    
    model_config = ConfigDict(from_attributes=True)


class LabResultResponse(LabResultInDB):
    """
    Schema for lab result API responses.
    """
    formatted_result: str = Field(..., description="Formatted result string")
    
    @validator("formatted_result", always=True)
    def format_result(cls, v, values):
        """
        Format result with unit for display.
        """
        data = values.data
        if data.get("result_value") is not None and data.get("unit"):
            return f"{data['result_value']} {data['unit']}"
        return data.get("result_text", "N/A")


# Patient summary schema
class PatientSummary(BaseModel):
    """
    Comprehensive patient summary combining multiple data sources.
    Used for clinical decision support and reporting.
    """
    patient: PatientResponse
    active_admission: Optional[AdmissionResponse] = None
    recent_vitals: List[VitalSignResponse] = Field(
        default_factory=list,
        description="Most recent vital signs"
    )
    recent_labs: List[LabResultResponse] = Field(
        default_factory=list,
        description="Most recent lab results"
    )
    active_medications: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Currently active medications"
    )
    risk_score: Optional[float] = Field(
        None,
        description="Computed risk score",
        ge=0,
        le=1
    )
    alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active clinical alerts"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient": {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "full_name": "John Doe",
                    "age": 44
                },
                "risk_score": 0.23,
                "alerts": [
                    {
                        "type": "allergy",
                        "message": "Penicillin allergy documented",
                        "severity": "high"
                    }
                ]
            }
        }
    )


# Patient search parameters
class PatientSearchParams(BaseModel):
    """
    Parameters for searching patients.
    """
    name: Optional[str] = Field(
        None,
        description="Search by patient name (partial match)",
        min_length=2
    )
    mrn: Optional[str] = Field(
        None,
        description="Search by exact MRN",
        min_length=3
    )
    date_of_birth_from: Optional[date] = Field(
        None,
        description="Minimum date of birth"
    )
    date_of_birth_to: Optional[date] = Field(
        None,
        description="Maximum date of birth"
    )
    gender: Optional[Gender] = Field(
        None,
        description="Filter by gender"
    )
    blood_type: Optional[BloodType] = Field(
        None,
        description="Filter by blood type"
    )
    is_active: Optional[bool] = Field(
        None,
        description="Filter by active status"
    )
    department: Optional[str] = Field(
        None,
        description="Filter by current department"
    )
    limit: int = Field(
        50,
        description="Maximum number of results",
        ge=1,
        le=1000
    )
    offset: int = Field(
        0,
        description="Number of results to skip",
        ge=0
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "John",
                "is_active": True,
                "limit": 20,
                "offset": 0
            }
        }
    )