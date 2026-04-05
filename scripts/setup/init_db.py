"""
Database initialization script for AegisMedBot.

This script creates all database tables, indexes, and relationships required
for the hospital intelligence platform. It handles:
- Table creation for all entities
- Foreign key relationships
- Index optimization for query performance
- Enum type creation
- Initial schema validation

Usage:
    python scripts/setup/init_db.py --force-recreate
    python scripts/setup/init_db.py --check-only
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import database dependencies
from sqlalchemy import (
    create_engine, 
    MetaData, 
    Table, 
    Column, 
    String, 
    Integer, 
    DateTime, 
    Float, 
    Boolean,
    ForeignKey,
    Text,
    JSON,
    Enum,
    Index,
    inspect,
    text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create declarative base for ORM models
Base = declarative_base()

# Define database models with relationships
class Patient(Base):
    """
    Patient model representing individuals receiving medical care.
    
    This table stores core patient demographic and medical information.
    It serves as the central reference for all patient-related data.
    """
    
    __tablename__ = 'patients'
    
    # Primary key using UUID for distributed system compatibility
    id = Column(String(36), primary_key=True, nullable=False)
    # Medical Record Number - unique identifier for healthcare systems
    mrn = Column(String(50), unique=True, nullable=False, index=True)
    # Personal identifiers
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(DateTime, nullable=False)
    # Demographic information
    gender = Column(String(20))
    blood_type = Column(String(5))
    # Medical history stored as JSON for flexibility
    allergies = Column(JSON)  # Array of allergy objects
    chronic_conditions = Column(JSON)  # Array of condition objects
    # Contact and emergency information
    phone = Column(String(20))
    email = Column(String(100))
    address = Column(Text)
    emergency_contact = Column(JSON)
    # Insurance and financial information
    insurance_info = Column(JSON)
    primary_care_physician = Column(String(100))
    # Audit and status fields
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    is_active = Column(Boolean, default=True)
    # Additional metadata
    metadata = Column(JSON, default={})
    
    # Create indexes for common query patterns
    __table_args__ = (
        Index('idx_patient_mrn', 'mrn'),
        Index('idx_patient_name', 'last_name', 'first_name'),
        Index('idx_patient_dob', 'date_of_birth'),
        Index('idx_patient_active', 'is_active'),
    )

class Admission(Base):
    """
    Admission model tracking patient hospital stays.
    
    This table records each hospital admission episode including
    dates, departments, physicians, and outcomes.
    """
    
    __tablename__ = 'admissions'
    
    id = Column(String(36), primary_key=True, nullable=False)
    patient_id = Column(String(36), ForeignKey('patients.id'), nullable=False, index=True)
    
    # Admission details
    admission_date = Column(DateTime, nullable=False)
    discharge_date = Column(DateTime)
    admission_type = Column(String(50))  # Emergency, Elective, Urgent
    admission_source = Column(String(100))  # Home, Transfer, Other hospital
    
    # Location tracking
    department = Column(String(100))
    unit = Column(String(100))
    room_number = Column(String(20))
    bed_number = Column(String(20))
    
    # Clinical information
    primary_diagnosis = Column(Text)
    secondary_diagnoses = Column(JSON)  # Array of diagnoses
    procedures = Column(JSON)  # Array of procedures performed
    
    # Physician information
    admitting_physician = Column(String(100))
    attending_physician = Column(String(100))
    consulting_physicians = Column(JSON)  # Array of consultant names
    
    # Status tracking
    status = Column(String(50), default='active')  # Active, Discharged, Transferred
    discharge_disposition = Column(String(100))  # Home, SNF, Rehab, Expired
    
    # Risk scores and predictions
    admission_risk_score = Column(Float)
    mortality_risk = Column(Float)
    readmission_risk = Column(Float)
    
    # Audit fields
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    __table_args__ = (
        Index('idx_admission_patient', 'patient_id'),
        Index('idx_admission_dates', 'admission_date', 'discharge_date'),
        Index('idx_admission_status', 'status'),
        Index('idx_admission_department', 'department'),
    )

class VitalSign(Base):
    """
    Vital signs model for patient monitoring data.
    
    Stores time-series vital sign measurements including
    heart rate, blood pressure, temperature, and respiratory metrics.
    """
    
    __tablename__ = 'vital_signs'
    
    id = Column(String(36), primary_key=True, nullable=False)
    patient_id = Column(String(36), ForeignKey('patients.id'), nullable=False, index=True)
    admission_id = Column(String(36), ForeignKey('admissions.id'), index=True)
    
    # Measurement timestamp
    recorded_at = Column(DateTime, nullable=False, index=True)
    
    # Cardiovascular metrics
    heart_rate = Column(Integer)  # Beats per minute
    blood_pressure_systolic = Column(Integer)  # mmHg
    blood_pressure_diastolic = Column(Integer)  # mmHg
    mean_arterial_pressure = Column(Float)  # Calculated MAP
    
    # Respiratory metrics
    respiratory_rate = Column(Integer)  # Breaths per minute
    oxygen_saturation = Column(Float)  # SpO2 percentage
    oxygen_flow_rate = Column(Float)  # Liters per minute
    oxygen_device = Column(String(50))  # Nasal cannula, mask, ventilator
    
    # Temperature and metabolic
    temperature = Column(Float)  # Celsius
    temperature_site = Column(String(50))  # Oral, Axillary, Tympanic
    glucose = Column(Float)  # Blood glucose mg/dL
    
    # Neurological and pain assessment
    consciousness_level = Column(String(50))  # Alert, Verbal, Pain, Unresponsive
    pain_score = Column(Integer)  # 0-10 scale
    pain_location = Column(String(100))
    
    # Additional metrics
    weight = Column(Float)  # kg
    height = Column(Float)  # cm
    bmi = Column(Float)  # Body Mass Index
    
    # Clinical flags
    is_abnormal = Column(Boolean, default=False)
    is_critical = Column(Boolean, default=False)
    
    # Who recorded the measurement
    recorded_by = Column(String(100))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        Index('idx_vitals_patient_time', 'patient_id', 'recorded_at'),
        Index('idx_vitals_admission', 'admission_id'),
        Index('idx_vitals_critical', 'is_critical'),
    )

class LabResult(Base):
    """
    Laboratory results model for patient test results.
    
    Stores laboratory test results including blood work,
    microbiology, pathology, and other diagnostic tests.
    """
    
    __tablename__ = 'lab_results'
    
    id = Column(String(36), primary_key=True, nullable=False)
    patient_id = Column(String(36), ForeignKey('patients.id'), nullable=False, index=True)
    admission_id = Column(String(36), ForeignKey('admissions.id'), index=True)
    
    # Test identification
    test_name = Column(String(200), nullable=False)
    test_code = Column(String(50))  # LOINC code for standardization
    test_category = Column(String(100))  # Chemistry, Hematology, Microbiology
    
    # Results
    result_value = Column(Float)
    result_text = Column(Text)
    unit = Column(String(50))
    reference_range_low = Column(Float)
    reference_range_high = Column(Float)
    
    # Interpretation
    interpretation = Column(String(50))  # Normal, Abnormal, Critical
    flag = Column(String(10))  # H, L, HH, LL
    
    # Timing
    ordered_at = Column(DateTime)
    collected_at = Column(DateTime)
    resulted_at = Column(DateTime, index=True)
    
    # Ordering information
    ordering_physician = Column(String(100))
    performing_lab = Column(String(100))
    
    # Clinical significance
    is_critical_value = Column(Boolean, default=False)
    requires_followup = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    __table_args__ = (
        Index('idx_lab_patient', 'patient_id'),
        Index('idx_lab_test', 'test_name'),
        Index('idx_lab_resulted', 'resulted_at'),
        Index('idx_lab_critical', 'is_critical_value'),
    )

class MedicationOrder(Base):
    """
    Medication orders model for patient prescriptions.
    
    Tracks all medication orders including dosing, administration,
    and monitoring requirements.
    """
    
    __tablename__ = 'medication_orders'
    
    id = Column(String(36), primary_key=True, nullable=False)
    patient_id = Column(String(36), ForeignKey('patients.id'), nullable=False, index=True)
    admission_id = Column(String(36), ForeignKey('admissions.id'), index=True)
    
    # Medication identification
    medication_name = Column(String(200), nullable=False)
    medication_code = Column(String(50))  # RxNorm code
    generic_name = Column(String(200))
    drug_class = Column(String(100))
    
    # Dosing information
    dose = Column(Float)
    dose_unit = Column(String(20))
    route = Column(String(50))  # Oral, IV, IM, SubQ
    frequency = Column(String(100))  # QD, BID, TID, QID, PRN
    duration_days = Column(Integer)
    
    # Scheduling
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime)
    last_administered = Column(DateTime)
    
    # Order details
    ordering_physician = Column(String(100))
    order_type = Column(String(50))  # New, Renew, Change, Discontinue
    order_status = Column(String(50), default='active')  # Active, Completed, Discontinued
    
    # Clinical instructions
    instructions = Column(Text)
    special_instructions = Column(Text)
    
    # Safety checks
    requires_renal_adjustment = Column(Boolean, default=False)
    requires_hepatic_adjustment = Column(Boolean, default=False)
    therapeutic_duplicate = Column(Boolean, default=False)
    
    # Monitoring requirements
    monitoring_required = Column(JSON)  # Lab tests, vital signs needed
    
    # Audit fields
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    __table_args__ = (
        Index('idx_medication_patient', 'patient_id'),
        Index('idx_medication_active', 'order_status'),
        Index('idx_medication_dates', 'start_date', 'end_date'),
    )

class ClinicalNote(Base):
    """
    Clinical notes model for documentation.
    
    Stores all clinical documentation including progress notes,
    discharge summaries, and consultation notes.
    """
    
    __tablename__ = 'clinical_notes'
    
    id = Column(String(36), primary_key=True, nullable=False)
    patient_id = Column(String(36), ForeignKey('patients.id'), nullable=False, index=True)
    admission_id = Column(String(36), ForeignKey('admissions.id'), index=True)
    
    # Note identification
    note_type = Column(String(100))  # Progress Note, Discharge Summary, Consult
    note_title = Column(String(200))
    note_content = Column(Text, nullable=False)
    
    # Authorship
    author = Column(String(100), nullable=False)
    author_role = Column(String(100))  # Physician, Nurse, Resident
    
    # Timing
    note_date = Column(DateTime, nullable=False, index=True)
    signed_date = Column(DateTime)
    
    # Sections (stored as JSON for flexibility)
    sections = Column(JSON)  # Subjective, Objective, Assessment, Plan
    structured_data = Column(JSON)  # Extracted clinical data
    
    # Status
    status = Column(String(50), default='draft')  # Draft, Final, Amended
    is_verified = Column(Boolean, default=False)
    
    # Clinical coding
    icd10_codes = Column(JSON)  # Diagnosis codes
    cpt_codes = Column(JSON)  # Procedure codes
    
    # Audit fields
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    __table_args__ = (
        Index('idx_notes_patient', 'patient_id'),
        Index('idx_notes_date', 'note_date'),
        Index('idx_notes_type', 'note_type'),
    )

class Staff(Base):
    """
    Staff model for hospital personnel.
    
    Tracks all hospital staff including physicians, nurses,
    administrators, and support staff.
    """
    
    __tablename__ = 'staff'
    
    id = Column(String(36), primary_key=True, nullable=False)
    employee_id = Column(String(50), unique=True, nullable=False, index=True)
    
    # Personal information
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    phone = Column(String(20))
    
    # Professional information
    role = Column(String(100), nullable=False)  # Physician, Nurse, Admin
    specialty = Column(String(100))  # Cardiology, Neurology, etc.
    department = Column(String(100))
    
    # Credentials
    license_number = Column(String(50))
    npi_number = Column(String(20))  # National Provider Identifier
    dea_number = Column(String(20))  # DEA registration
    
    # Access control
    permissions = Column(JSON)  # JSON array of permissions
    access_level = Column(String(50), default='standard')
    
    # Schedule information
    default_shift = Column(String(50))  # Day, Evening, Night
    on_call_schedule = Column(JSON)  # On-call rotation data
    
    # Status
    is_active = Column(Boolean, default=True)
    is_on_call = Column(Boolean, default=False)
    
    # Audit fields
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    __table_args__ = (
        Index('idx_staff_employee', 'employee_id'),
        Index('idx_staff_role', 'role'),
        Index('idx_staff_department', 'department'),
    )

class HospitalResource(Base):
    """
    Hospital resources model for equipment and facilities.
    
    Tracks all hospital resources including beds, equipment,
    operating rooms, and other critical assets.
    """
    
    __tablename__ = 'hospital_resources'
    
    id = Column(String(36), primary_key=True, nullable=False)
    resource_id = Column(String(50), unique=True, nullable=False, index=True)
    
    # Resource identification
    resource_type = Column(String(100), nullable=False)  # Bed, Ventilator, OR
    resource_name = Column(String(200))
    department = Column(String(100))
    location = Column(String(200))
    
    # Status tracking
    status = Column(String(50), default='available')  # Available, InUse, Maintenance
    is_available = Column(Boolean, default=True)
    is_emergency_only = Column(Boolean, default=False)
    
    # Capacity and specifications
    capacity = Column(Integer)  # For beds: patient capacity
    specifications = Column(JSON)  # Technical specifications
    
    # Maintenance
    last_maintenance = Column(DateTime)
    next_maintenance = Column(DateTime)
    maintenance_notes = Column(Text)
    
    # Utilization
    current_occupant_id = Column(String(36))  # Patient or staff ID
    occupied_since = Column(DateTime)
    utilization_rate = Column(Float)  # Percentage
    
    # Audit fields
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    __table_args__ = (
        Index('idx_resource_type', 'resource_type'),
        Index('idx_resource_status', 'status'),
        Index('idx_resource_department', 'department'),
    )

class AuditLog(Base):
    """
    Audit log model for compliance tracking.
    
    Records all system access, data modifications, and
    sensitive operations for compliance and security.
    """
    
    __tablename__ = 'audit_logs'
    
    id = Column(String(36), primary_key=True, nullable=False)
    
    # Action details
    user_id = Column(String(36), nullable=False, index=True)
    user_role = Column(String(100))
    action = Column(String(100), nullable=False)  # CREATE, READ, UPDATE, DELETE
    resource_type = Column(String(100))  # Patient, Admission, Medication
    resource_id = Column(String(36))
    
    # Context
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(String(500))
    session_id = Column(String(100))
    
    # Data tracking
    old_values = Column(JSON)  # Before modification
    new_values = Column(JSON)  # After modification
    query_params = Column(JSON)  # API query parameters
    
    # Compliance
    reason = Column(Text)  # Why this action was performed
    requires_review = Column(Boolean, default=False)
    reviewed_by = Column(String(100))
    reviewed_at = Column(DateTime)
    
    # Outcome
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.now, nullable=False, index=True)
    
    __table_args__ = (
        Index('idx_audit_user', 'user_id'),
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_action', 'action'),
        Index('idx_audit_timestamp', 'created_at'),
    )

class Conversation(Base):
    """
    Conversation model for AI assistant interactions.
    
    Stores all chat conversations between users and AI agents
    for context preservation and training data collection.
    """
    
    __tablename__ = 'conversations'
    
    id = Column(String(36), primary_key=True, nullable=False)
    
    # Participants
    user_id = Column(String(36), nullable=False, index=True)
    user_role = Column(String(100))
    
    # Context
    patient_id = Column(String(36), index=True)
    admission_id = Column(String(36))
    
    # Conversation metadata
    title = Column(String(200))
    messages = Column(JSON)  # Array of message objects
    message_count = Column(Integer, default=0)
    
    # Agent interaction
    agents_involved = Column(JSON)  # Which agents participated
    final_agent = Column(String(100))
    resolved = Column(Boolean, default=False)
    
    # Quality metrics
    user_satisfaction = Column(Integer)  # 1-5 rating
    response_time_ms = Column(Integer)
    confidence_scores = Column(JSON)
    
    # Feedback
    user_feedback = Column(Text)
    clinical_accuracy_score = Column(Float)  # For training evaluation
    
    # Timing
    started_at = Column(DateTime, default=datetime.now, nullable=False)
    ended_at = Column(DateTime)
    last_message_at = Column(DateTime)
    
    __table_args__ = (
        Index('idx_conversation_user', 'user_id'),
        Index('idx_conversation_patient', 'patient_id'),
        Index('idx_conversation_started', 'started_at'),
    )

def create_database_engine():
    """
    Create SQLAlchemy database engine with connection pooling.
    
    This function establishes connection to PostgreSQL database
    with optimized connection pooling for production workloads.
    
    Returns:
        SQLAlchemy engine object configured for production use
    """
    # Build database URL from environment variables
    db_user = os.getenv('POSTGRES_USER', 'medintel')
    db_password = os.getenv('POSTGRES_PASSWORD', 'medintel123')
    db_host = os.getenv('POSTGRES_SERVER', 'localhost')
    db_port = os.getenv('POSTGRES_PORT', '5432')
    db_name = os.getenv('POSTGRES_DB', 'medintel')
    
    database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    # Create engine with connection pooling for performance
    engine = create_engine(
        database_url,
        # Connection pool configuration
        pool_size=20,  # Maximum connections to keep in pool
        max_overflow=10,  # Extra connections beyond pool_size
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600,  # Recycle connections every hour
        # Performance optimizations
        echo=False,  # Disable SQL logging in production
        future=True,  # Use SQLAlchemy 2.0 style
        # Statement execution options
        execution_options={
            "isolation_level": "READ_COMMITTED"
        }
    )
    
    return engine

def create_tables(engine, drop_existing=False):
    """
    Create all database tables defined in the models.
    
    Args:
        engine: SQLAlchemy database engine
        drop_existing: If True, drop existing tables before creating
    """
    try:
        if drop_existing:
            logger.warning("Dropping all existing tables...")
            Base.metadata.drop_all(bind=engine)
            logger.info("All existing tables dropped successfully")
        
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("All tables created successfully")
        
        return True
        
    except SQLAlchemyError as e:
        logger.error(f"Error creating tables: {str(e)}")
        return False

def create_indexes(engine):
    """
    Create additional indexes for performance optimization.
    
    This function creates composite indexes and partial indexes
    that aren't automatically created by SQLAlchemy declarative models.
    """
    try:
        with engine.connect() as conn:
            # Create partial index for active patients
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_active_patients 
                ON patients(id, mrn, last_name, first_name) 
                WHERE is_active = true
            """))
            
            # Create composite index for vital signs queries
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_vitals_patient_critical 
                ON vital_signs(patient_id, recorded_at, is_critical) 
                WHERE is_critical = true
            """))
            
            # Create index for lab result trending
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_lab_trending 
                ON lab_results(patient_id, test_name, resulted_at) 
                WHERE is_critical_value = false
            """))
            
            # Create index for active medication orders
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_active_medications 
                ON medication_orders(patient_id, start_date, end_date) 
                WHERE order_status = 'active'
            """))
            
            # Create index for recent audit logs
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_recent_audit_logs 
                ON audit_logs(created_at, user_id, action) 
                WHERE created_at > NOW() - INTERVAL '30 days'
            """))
            
            conn.commit()
            logger.info("Additional indexes created successfully")
            
    except SQLAlchemyError as e:
        logger.error(f"Error creating indexes: {str(e)}")

def create_enums(engine):
    """
    Create PostgreSQL enum types for standardized values.
    
    Enums ensure data consistency and provide better query performance
    for columns with predefined value sets.
    """
    try:
        with engine.connect() as conn:
            # Admission status enum
            conn.execute(text("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'admission_status') THEN
                        CREATE TYPE admission_status AS ENUM (
                            'active', 'discharged', 'transferred', 'expired'
                        );
                    END IF;
                END
                $$;
            """))
            
            # Order status enum
            conn.execute(text("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'order_status') THEN
                        CREATE TYPE order_status AS ENUM (
                            'active', 'completed', 'discontinued', 'on_hold'
                        );
                    END IF;
                END
                $$;
            """))
            
            # Resource status enum
            conn.execute(text("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'resource_status') THEN
                        CREATE TYPE resource_status AS ENUM (
                            'available', 'in_use', 'maintenance', 'reserved'
                        );
                    END IF;
                END
                $$;
            """))
            
            conn.commit()
            logger.info("Database enums created successfully")
            
    except SQLAlchemyError as e:
        logger.error(f"Error creating enums: {str(e)}")

def validate_schema(engine):
    """
    Validate that all required tables and columns exist.
    
    This function performs a comprehensive check of the database schema
    to ensure everything was created correctly.
    
    Returns:
        Dictionary with validation results
    """
    inspector = inspect(engine)
    validation_results = {
        'tables_found': [],
        'tables_missing': [],
        'columns_validated': {},
        'is_valid': True
    }
    
    # List of required tables
    required_tables = [
        'patients', 'admissions', 'vital_signs', 'lab_results',
        'medication_orders', 'clinical_notes', 'staff', 
        'hospital_resources', 'audit_logs', 'conversations'
    ]
    
    # Get existing tables
    existing_tables = inspector.get_table_names()
    
    # Check each required table
    for table in required_tables:
        if table in existing_tables:
            validation_results['tables_found'].append(table)
            
            # Validate columns for critical tables
            columns = inspector.get_columns(table)
            column_names = [col['name'] for col in columns]
            validation_results['columns_validated'][table] = column_names
            
        else:
            validation_results['tables_missing'].append(table)
            validation_results['is_valid'] = False
    
    # Log validation results
    if validation_results['is_valid']:
        logger.info("Schema validation passed: All required tables exist")
        logger.info(f"Tables found: {len(validation_results['tables_found'])}")
    else:
        logger.error(f"Schema validation failed: Missing tables {validation_results['tables_missing']}")
    
    return validation_results

def setup_database_permissions(engine):
    """
    Set up database user permissions for role-based access.
    
    This function creates database roles and grants appropriate
    permissions for different application components.
    """
    try:
        with engine.connect() as conn:
            # Create readonly role for reporting and analytics
            conn.execute(text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'medintel_readonly') THEN
                        CREATE ROLE medintel_readonly;
                    END IF;
                END
                $$;
            """))
            
            # Grant read permissions on all tables
            conn.execute(text("""
                GRANT SELECT ON ALL TABLES IN SCHEMA public TO medintel_readonly;
            """))
            
            # Create readwrite role for application
            conn.execute(text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'medintel_readwrite') THEN
                        CREATE ROLE medintel_readwrite;
                    END IF;
                END
                $$;
            """))
            
            # Grant all DML permissions
            conn.execute(text("""
                GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO medintel_readwrite;
            """))
            
            conn.commit()
            logger.info("Database permissions configured successfully")
            
    except SQLAlchemyError as e:
        logger.error(f"Error setting up permissions: {str(e)}")

def main():
    """
    Main execution function for database initialization.
    
    Parses command line arguments and orchestrates the
    database initialization process.
    """
    parser = argparse.ArgumentParser(
        description='Initialize AegisMedBot database with all required tables and indexes'
    )
    parser.add_argument(
        '--force-recreate',
        action='store_true',
        help='Drop existing tables before creating new ones'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check if database is properly configured without making changes'
    )
    parser.add_argument(
        '--skip-indexes',
        action='store_true',
        help='Skip creation of additional indexes'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting AegisMedBot database initialization")
    logger.info(f"Force recreate: {args.force_recreate}")
    logger.info(f"Check only: {args.check_only}")
    
    # Create database engine
    engine = create_database_engine()
    
    if args.check_only:
        logger.info("Running in check-only mode")
        validation = validate_schema(engine)
        
        if validation['is_valid']:
            logger.info("Database schema is valid and ready for use")
            sys.exit(0)
        else:
            logger.error("Database schema validation failed")
            logger.error(f"Missing tables: {validation['tables_missing']}")
            sys.exit(1)
    
    # Create tables
    if not create_tables(engine, drop_existing=args.force_recreate):
        logger.error("Failed to create tables")
        sys.exit(1)
    
    # Create enum types
    create_enums(engine)
    
    # Create additional indexes if not skipped
    if not args.skip_indexes:
        create_indexes(engine)
    
    # Set up permissions
    setup_database_permissions(engine)
    
    # Validate final schema
    validation = validate_schema(engine)
    
    if validation['is_valid']:
        logger.info("=" * 60)
        logger.info("Database initialization completed successfully!")
        logger.info(f"Created {len(validation['tables_found'])} tables")
        logger.info("Database is ready for use with AegisMedBot")
        logger.info("=" * 60)
    else:
        logger.error("Database initialization completed with errors")
        sys.exit(1)

if __name__ == "__main__":
    main()