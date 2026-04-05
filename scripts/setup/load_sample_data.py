"""
Sample data loader for AegisMedBot.

This script populates the database with realistic sample data
for testing, development, and demonstration purposes.

Data includes:
- Sample patients with demographics and medical history
- Admission records with diagnoses and procedures
- Vital signs time series for multiple patients
- Lab results with normal and abnormal values
- Medication orders with dosing schedules
- Clinical notes and documentation
- Staff profiles with roles and permissions
- Hospital resources and equipment tracking
- Conversation history with AI agents

Usage:
    python scripts/setup/load_sample_data.py --size small|medium|large
    python scripts/setup/load_sample_data.py --clear-existing
    python scripts/setup/load_sample_data.py --patient-count 100
"""

import sys
import os
import argparse
import logging
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import database models and utilities
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Import database models
from scripts.setup.init_db import (
    Base, Patient, Admission, VitalSign, LabResult,
    MedicationOrder, ClinicalNote, Staff, HospitalResource,
    AuditLog, Conversation
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SampleDataGenerator:
    """
    Generates realistic sample healthcare data for testing.
    
    This class creates synthetic patient data that mimics real
    hospital data while maintaining patient privacy through
    synthetic generation rather than using real patient data.
    """
    
    def __init__(self, session: Session, data_size: str = 'medium'):
        """
        Initialize the data generator.
        
        Args:
            session: SQLAlchemy database session
            data_size: Size of dataset to generate (small, medium, large)
        """
        self.session = session
        self.data_size = data_size
        
        # Set size multipliers based on data_size parameter
        size_multipliers = {
            'small': {'patients': 10, 'admissions_per_patient': 1, 'vitals_per_admission': 10},
            'medium': {'patients': 50, 'admissions_per_patient': 2, 'vitals_per_admission': 50},
            'large': {'patients': 200, 'admissions_per_patient': 3, 'vitals_per_admission': 100}
        }
        
        self.multipliers = size_multipliers.get(data_size, size_multipliers['medium'])
        
        # Medical terminology and data pools for realistic generation
        self.first_names = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 
                           'Michael', 'Linda', 'William', 'Elizabeth', 'David', 'Susan',
                           'Richard', 'Jessica', 'Joseph', 'Sarah', 'Thomas', 'Karen',
                           'Charles', 'Nancy', 'Christopher', 'Lisa', 'Daniel', 'Betty']
        
        self.last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia',
                          'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez',
                          'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore',
                          'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson', 'White']
        
        self.departments = ['Emergency', 'Cardiology', 'Neurology', 'Orthopedics',
                           'Oncology', 'Pediatrics', 'ICU', 'General Medicine',
                           'Surgery', 'Psychiatry', 'Radiology', 'Laboratory']
        
        self.diagnoses = [
            'Acute Myocardial Infarction', 'Community Acquired Pneumonia',
            'Type 2 Diabetes Mellitus', 'Hypertensive Crisis', 'Cerebrovascular Accident',
            'Femur Fracture', 'Appendicitis', 'Sepsis', 'Congestive Heart Failure',
            'Chronic Obstructive Pulmonary Disease', 'Urinary Tract Infection',
            'Atrial Fibrillation', 'Pulmonary Embolism', 'Gastrointestinal Bleed'
        ]
        
        self.medications = [
            'Lisinopril', 'Metformin', 'Atorvastatin', 'Amiodarone', 'Warfarin',
            'Furosemide', 'Pantoprazole', 'Albuterol', 'Ceftriaxone', 'Vancomycin',
            'Morphine', 'Acetaminophen', 'Ibuprofen', 'Ondansetron', 'Heparin'
        ]
        
        self.lab_tests = [
            {'name': 'Complete Blood Count', 'code': 'CBC', 'unit': 'cells/mcL',
             'normal_range': {'low': 4.5, 'high': 11.0}},
            {'name': 'Comprehensive Metabolic Panel', 'code': 'CMP', 'unit': 'mg/dL',
             'normal_range': {'low': 70, 'high': 100}},
            {'name': 'Troponin I', 'code': 'TROP', 'unit': 'ng/mL',
             'normal_range': {'low': 0, 'high': 0.04}},
            {'name': 'International Normalized Ratio', 'code': 'INR', 'unit': 'ratio',
             'normal_range': {'low': 0.8, 'high': 1.2}},
            {'name': 'C-Reactive Protein', 'code': 'CRP', 'unit': 'mg/L',
             'normal_range': {'low': 0, 'high': 10.0}}
        ]
        
        # Pre-generate patient IDs to track created records
        self.generated_ids = {
            'patients': [],
            'admissions': [],
            'staff': []
        }
    
    def generate_patient(self) -> Patient:
        """
        Generate a single patient record with realistic demographics.
        
        Returns:
            Patient object with populated attributes
        """
        # Generate unique identifiers
        patient_id = str(uuid.uuid4())
        mrn = f"MRN{random.randint(100000, 999999)}"
        
        # Randomly select name components
        first_name = random.choice(self.first_names)
        last_name = random.choice(self.last_names)
        
        # Generate realistic date of birth (ages 0-100 years)
        age_years = random.randint(0, 100)
        date_of_birth = datetime.now() - timedelta(days=age_years * 365)
        
        # Random demographics
        gender = random.choice(['Male', 'Female', 'Other'])
        blood_type = random.choice(['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-'])
        
        # Generate realistic allergies (20% chance of having allergies)
        allergies = []
        if random.random() < 0.2:
            possible_allergies = ['Penicillin', 'Sulfa Drugs', 'Latex', 'Shellfish', 
                                 'Peanuts', 'Codeine', 'Aspirin', 'Iodine']
            allergy_count = random.randint(1, 3)
            allergies = random.sample(possible_allergies, allergy_count)
        
        # Generate chronic conditions (30% chance of having chronic conditions)
        chronic_conditions = []
        if random.random() < 0.3:
            possible_conditions = ['Hypertension', 'Diabetes Type 2', 'Hyperlipidemia',
                                  'Asthma', 'COPD', 'Coronary Artery Disease',
                                  'Chronic Kidney Disease', 'Depression']
            condition_count = random.randint(1, 2)
            chronic_conditions = random.sample(possible_conditions, condition_count)
        
        # Generate contact information
        phone = f"{random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}"
        email = f"{first_name.lower()}.{last_name.lower()}@example.com"
        
        address = f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Maple', 'Cedar'])} St, {random.choice(['Springfield', 'Riverside', 'Lakewood'])}"
        
        # Emergency contact (90% chance of having one)
        emergency_contact = None
        if random.random() < 0.9:
            emergency_contact = {
                'name': f"{random.choice(self.first_names)} {random.choice(self.last_names)}",
                'relationship': random.choice(['Spouse', 'Parent', 'Child', 'Sibling']),
                'phone': phone
            }
        
        # Insurance information
        insurance_info = {
            'provider': random.choice(['Blue Cross', 'Aetna', 'Cigna', 'UnitedHealth', 'Medicare']),
            'policy_number': f"POL{random.randint(100000, 999999)}",
            'group_number': f"GRP{random.randint(1000, 9999)}"
        }
        
        # Create patient object
        patient = Patient(
            id=patient_id,
            mrn=mrn,
            first_name=first_name,
            last_name=last_name,
            date_of_birth=date_of_birth,
            gender=gender,
            blood_type=blood_type,
            allergies=allergies,
            chronic_conditions=chronic_conditions,
            phone=phone,
            email=email,
            address=address,
            emergency_contact=emergency_contact,
            insurance_info=insurance_info,
            primary_care_physician=f"Dr. {random.choice(self.last_names)}",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            is_active=True,
            metadata={}
        )
        
        return patient
    
    def generate_admission(self, patient: Patient) -> Admission:
        """
        Generate an admission record for a patient.
        
        Args:
            patient: Patient object to associate with admission
            
        Returns:
            Admission object with populated attributes
        """
        admission_id = str(uuid.uuid4())
        
        # Generate admission date (within last 90 days)
        days_ago = random.randint(0, 90)
        admission_date = datetime.now() - timedelta(days=days_ago)
        
        # Determine if patient has been discharged (70% chance if admission > 7 days ago)
        is_discharged = False
        discharge_date = None
        if days_ago > 7 and random.random() < 0.7:
            is_discharged = True
            discharge_date = admission_date + timedelta(days=random.randint(1, days_ago))
        
        # Admission type and source
        admission_type = random.choice(['Emergency', 'Elective', 'Urgent'])
        admission_source = random.choice(['Home', 'Physician Referral', 'Transfer', 'ER'])
        
        # Department assignment
        department = random.choice(self.departments)
        unit = f"{department} Unit"
        room_number = f"{random.randint(100, 500)}{random.choice(['A', 'B', 'C'])}"
        bed_number = str(random.randint(1, 4))
        
        # Diagnoses (primary and secondary)
        primary_diagnosis = random.choice(self.diagnoses)
        secondary_count = random.randint(0, 2)
        secondary_diagnoses = random.sample(self.diagnoses, secondary_count) if secondary_count > 0 else []
        
        # Procedures performed (30% chance)
        procedures = []
        if random.random() < 0.3:
            possible_procedures = ['X-ray', 'CT Scan', 'MRI', 'Ultrasound', 
                                  'Echocardiogram', 'Cardiac Catheterization']
            procedure_count = random.randint(1, 2)
            procedures = random.sample(possible_procedures, procedure_count)
        
        # Physician assignment
        physicians_list = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia']
        admitting_physician = f"Dr. {random.choice(physicians_list)}"
        attending_physician = f"Dr. {random.choice(physicians_list)}"
        
        # Consulting physicians (30% chance)
        consulting_physicians = []
        if random.random() < 0.3:
            consulting_count = random.randint(1, 2)
            consulting_physicians = [f"Dr. {random.choice(physicians_list)}" for _ in range(consulting_count)]
        
        # Status determination
        status = 'discharged' if is_discharged else 'active'
        discharge_disposition = None
        if is_discharged:
            discharge_disposition = random.choice(['Home', 'Skilled Nursing Facility', 
                                                  'Rehabilitation', 'Home Health Care'])
        
        # Risk scores (for AI prediction demonstration)
        admission_risk_score = random.uniform(0, 1)
        mortality_risk = random.uniform(0, 0.3)  # Base mortality rate 0-30%
        readmission_risk = random.uniform(0, 0.5)  # Base readmission rate 0-50%
        
        # Create admission object
        admission = Admission(
            id=admission_id,
            patient_id=patient.id,
            admission_date=admission_date,
            discharge_date=discharge_date,
            admission_type=admission_type,
            admission_source=admission_source,
            department=department,
            unit=unit,
            room_number=room_number,
            bed_number=bed_number,
            primary_diagnosis=primary_diagnosis,
            secondary_diagnoses=secondary_diagnoses,
            procedures=procedures,
            admitting_physician=admitting_physician,
            attending_physician=attending_physician,
            consulting_physicians=consulting_physicians,
            status=status,
            discharge_disposition=discharge_disposition,
            admission_risk_score=admission_risk_score,
            mortality_risk=mortality_risk,
            readmission_risk=readmission_risk,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        return admission
    
    def generate_vital_signs(self, patient: Patient, admission: Admission, count: int) -> List[VitalSign]:
        """
        Generate time-series vital signs for a patient admission.
        
        Args:
            patient: Patient object
            admission: Admission object
            count: Number of vital sign measurements to generate
            
        Returns:
            List of VitalSign objects
        """
        vital_signs = []
        
        # Base vital sign values (normal ranges)
        base_heart_rate = random.randint(60, 100)
        base_systolic = random.randint(90, 140)
        base_diastolic = random.randint(60, 90)
        base_respiratory = random.randint(12, 20)
        base_temperature = random.uniform(36.5, 37.2)
        base_oxygen = random.uniform(95, 100)
        
        # Determine if patient has abnormal vitals (30% chance)
        is_abnormal_patient = random.random() < 0.3
        
        for i in range(count):
            # Calculate timestamp (spread over admission duration)
            if admission.discharge_date:
                max_time = admission.discharge_date
            else:
                max_time = datetime.now()
            
            time_offset = (max_time - admission.admission_date) * (i / count)
            recorded_at = admission.admission_date + time_offset
            
            # Add some random variation to vital signs
            variation_factor = random.uniform(-0.1, 0.1)
            
            # Adjust vitals if patient has abnormal condition
            if is_abnormal_patient:
                abnormality_type = random.choice(['high_bp', 'low_oxygen', 'fever', 'tachycardia'])
                
                if abnormality_type == 'high_bp':
                    systolic = base_systolic + random.randint(20, 50)
                    diastolic = base_diastolic + random.randint(10, 30)
                elif abnormality_type == 'low_oxygen':
                    oxygen = base_oxygen - random.uniform(5, 15)
                elif abnormality_type == 'fever':
                    temperature = base_temperature + random.uniform(1, 2)
                elif abnormality_type == 'tachycardia':
                    heart_rate = base_heart_rate + random.randint(20, 50)
            else:
                systolic = base_systolic * (1 + variation_factor)
                diastolic = base_diastolic * (1 + variation_factor)
                heart_rate = base_heart_rate * (1 + variation_factor)
                respiratory = base_respiratory * (1 + variation_factor)
                temperature = base_temperature + random.uniform(-0.5, 0.5)
                oxygen = base_oxygen + random.uniform(-2, 0)
            
            # Ensure values are within realistic bounds
            systolic = max(60, min(220, systolic))
            diastolic = max(40, min(130, diastolic))
            heart_rate = max(40, min(150, heart_rate))
            respiratory = max(8, min(40, respiratory))
            temperature = max(35.0, min(40.0, temperature))
            oxygen = max(70, min(100, oxygen))
            
            # Calculate derived metrics
            mean_arterial_pressure = diastolic + (systolic - diastolic) / 3
            
            # Determine if values are abnormal or critical
            is_abnormal = (systolic > 140 or systolic < 90 or 
                          heart_rate > 100 or heart_rate < 60 or
                          oxygen < 90 or temperature > 38.0)
            
            is_critical = (systolic > 180 or systolic < 70 or
                          heart_rate > 140 or heart_rate < 40 or
                          oxygen < 85 or temperature > 39.5)
            
            # Create vital sign record
            vital = VitalSign(
                id=str(uuid.uuid4()),
                patient_id=patient.id,
                admission_id=admission.id,
                recorded_at=recorded_at,
                heart_rate=int(heart_rate),
                blood_pressure_systolic=int(systolic),
                blood_pressure_diastolic=int(diastolic),
                mean_arterial_pressure=round(mean_arterial_pressure, 1),
                respiratory_rate=int(respiratory),
                oxygen_saturation=round(oxygen, 1),
                oxygen_flow_rate=random.uniform(0, 4) if oxygen < 92 else 0,
                oxygen_device='Nasal Cannula' if oxygen < 92 else None,
                temperature=round(temperature, 1),
                temperature_site=random.choice(['Oral', 'Tympanic', 'Axillary']),
                glucose=random.uniform(70, 180) if random.random() < 0.3 else None,
                consciousness_level=random.choice(['Alert', 'Verbal', 'Pain', 'Unresponsive']),
                pain_score=random.randint(0, 10) if random.random() < 0.5 else None,
                is_abnormal=is_abnormal,
                is_critical=is_critical,
                recorded_by=f"Nurse {random.choice(self.last_names)}",
                created_at=datetime.now()
            )
            
            vital_signs.append(vital)
        
        return vital_signs
    
    def generate_lab_results(self, patient: Patient, admission: Admission) -> List[LabResult]:
        """
        Generate laboratory results for a patient admission.
        
        Args:
            patient: Patient object
            admission: Admission object
            
        Returns:
            List of LabResult objects
        """
        lab_results = []
        
        # Number of lab tests (2-5 per admission)
        num_tests = random.randint(2, 5)
        
        for _ in range(num_tests):
            # Select random lab test
            lab_test = random.choice(self.lab_tests)
            
            # Determine if result is abnormal (30% chance)
            is_abnormal = random.random() < 0.3
            
            if is_abnormal:
                # Generate abnormal value (above or below normal range)
                if random.random() < 0.5:
                    # High value
                    result_value = lab_test['normal_range']['high'] * random.uniform(1.1, 2.0)
                    interpretation = 'High'
                else:
                    # Low value
                    result_value = lab_test['normal_range']['low'] * random.uniform(0.1, 0.9)
                    interpretation = 'Low'
            else:
                # Generate normal value
                result_value = random.uniform(
                    lab_test['normal_range']['low'],
                    lab_test['normal_range']['high']
                )
                interpretation = 'Normal'
            
            # Determine if critical (10% of abnormal results)
            is_critical = is_abnormal and random.random() < 0.3
            
            # Generate timestamps
            collected_at = admission.admission_date + timedelta(hours=random.randint(1, 48))
            resulted_at = collected_at + timedelta(hours=random.randint(2, 12))
            
            # Create lab result record
            lab_result = LabResult(
                id=str(uuid.uuid4()),
                patient_id=patient.id,
                admission_id=admission.id,
                test_name=lab_test['name'],
                test_code=lab_test['code'],
                test_category=lab_test['name'].split()[0],
                result_value=round(result_value, 2),
                unit=lab_test['unit'],
                reference_range_low=lab_test['normal_range']['low'],
                reference_range_high=lab_test['normal_range']['high'],
                interpretation=interpretation,
                flag='H' if interpretation == 'High' else 'L' if interpretation == 'Low' else 'N',
                ordered_at=collected_at - timedelta(hours=2),
                collected_at=collected_at,
                resulted_at=resulted_at,
                ordering_physician=f"Dr. {random.choice(self.last_names)}",
                performing_lab=random.choice(['Main Lab', 'Stat Lab', 'Reference Lab']),
                is_critical_value=is_critical,
                requires_followup=is_critical,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            lab_results.append(lab_result)
        
        return lab_results
    
    def generate_medication_orders(self, patient: Patient, admission: Admission) -> List[MedicationOrder]:
        """
        Generate medication orders for a patient admission.
        
        Args:
            patient: Patient object
            admission: Admission object
            
        Returns:
            List of MedicationOrder objects
        """
        medication_orders = []
        
        # Number of medications (1-4 per admission)
        num_medications = random.randint(1, 4)
        
        for _ in range(num_medications):
            medication_name = random.choice(self.medications)
            
            # Generate dosing information
            dose = random.choice([5, 10, 20, 25, 50, 100])
            dose_unit = random.choice(['mg', 'mcg', 'g', 'mL'])
            route = random.choice(['Oral', 'IV', 'IM', 'SubQ'])
            frequency = random.choice(['QD', 'BID', 'TID', 'QID', 'Q6H', 'Q8H', 'PRN'])
            
            # Duration
            duration_days = random.randint(3, 30)
            
            # Start and end dates
            start_date = admission.admission_date + timedelta(hours=random.randint(1, 24))
            
            # 80% of medications are still active if admission is recent
            is_active = random.random() < 0.8 if (datetime.now() - start_date).days < duration_days else False
            
            end_date = start_date + timedelta(days=duration_days) if not is_active else None
            
            # Determine if medication requires adjustments
            requires_renal = random.random() < 0.1
            requires_hepatic = random.random() < 0.1
            
            # Create medication order
            medication = MedicationOrder(
                id=str(uuid.uuid4()),
                patient_id=patient.id,
                admission_id=admission.id,
                medication_name=medication_name,
                medication_code=f"RX{random.randint(10000, 99999)}",
                generic_name=medication_name.lower(),
                drug_class=random.choice(['Antihypertensive', 'Antidiabetic', 'Antibiotic', 
                                         'Anticoagulant', 'Analgesic']),
                dose=float(dose),
                dose_unit=dose_unit,
                route=route,
                frequency=frequency,
                duration_days=duration_days,
                start_date=start_date,
                end_date=end_date,
                ordering_physician=f"Dr. {random.choice(self.last_names)}",
                order_type=random.choice(['New', 'Renew']),
                order_status='active' if is_active else 'completed',
                instructions=f"Take {dose}{dose_unit} {frequency} as directed",
                special_instructions="Take with food" if random.random() < 0.3 else None,
                requires_renal_adjustment=requires_renal,
                requires_hepatic_adjustment=requires_hepatic,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            medication_orders.append(medication)
        
        return medication_orders
    
    def generate_clinical_notes(self, patient: Patient, admission: Admission) -> List[ClinicalNote]:
        """
        Generate clinical notes for documentation.
        
        Args:
            patient: Patient object
            admission: Admission object
            
        Returns:
            List of ClinicalNote objects
        """
        clinical_notes = []
        
        # Note types and their typical content templates
        note_templates = {
            'Progress Note': """
                SUBJECTIVE: Patient reports {symptom_description}.
                OBJECTIVE: Vitals show {vital_status}. On exam, {exam_findings}.
                ASSESSMENT: {assessment_text}.
                PLAN: {plan_text}.
            """,
            'Discharge Summary': """
                Admission Date: {admission_date}
                Discharge Date: {discharge_date}
                Attending Physician: {physician}
                Primary Diagnosis: {diagnosis}
                Hospital Course: {course_description}
                Discharge Disposition: {disposition}
                Follow-up Instructions: {followup}
            """,
            'Consultation Note': """
                Reason for Consultation: {reason}
                History: {history}
                Physical Exam: {exam}
                Assessment: {assessment}
                Recommendations: {recommendations}
            """
        }
        
        # Generate 1-3 notes per admission
        num_notes = random.randint(1, 3)
        
        for _ in range(num_notes):
            note_type = random.choice(list(note_templates.keys()))
            
            # Generate realistic content based on note type
            if note_type == 'Progress Note':
                symptom_options = ['chest pain', 'shortness of breath', 'fatigue', 
                                  'headache', 'nausea', 'fever']
                symptom = random.choice(symptom_options)
                
                symptom_description = f"mild {symptom}" if random.random() < 0.7 else f"severe {symptom}"
                vital_status = "stable" if random.random() < 0.8 else "unstable"
                exam_findings = "unremarkable" if random.random() < 0.7 else "notable for abnormalities"
                assessment_text = f"Patient is improving with current treatment" if random.random() < 0.8 else "Condition requires further evaluation"
                plan_text = "Continue current management" if random.random() < 0.7 else "Adjust medications and order additional tests"
                
                content = note_templates[note_type].format(
                    symptom_description=symptom_description,
                    vital_status=vital_status,
                    exam_findings=exam_findings,
                    assessment_text=assessment_text,
                    plan_text=plan_text
                )
                
            elif note_type == 'Discharge Summary':
                content = note_templates[note_type].format(
                    admission_date=admission.admission_date.strftime('%Y-%m-%d'),
                    discharge_date=admission.discharge_date.strftime('%Y-%m-%d') if admission.discharge_date else 'Pending',
                    physician=admission.attending_physician,
                    diagnosis=admission.primary_diagnosis,
                    course_description="Patient admitted for management of condition. Responded well to treatment.",
                    disposition=admission.discharge_disposition or 'Home',
                    followup="Follow up with primary care physician in 1 week"
                )
            else:
                content = note_templates[note_type].format(
                    reason="Evaluation of cardiac symptoms",
                    history="Patient has history of hypertension",
                    exam="Vital signs within normal limits",
                    assessment="Stable condition",
                    recommendations="Continue current medications"
                )
            
            # Determine note date (within admission period)
            if admission.discharge_date and note_type == 'Discharge Summary':
                note_date = admission.discharge_date
            else:
                note_date = admission.admission_date + timedelta(days=random.randint(1, 7))
            
            # Determine if note is signed
            is_signed = random.random() < 0.9
            signed_date = note_date + timedelta(hours=random.randint(1, 48)) if is_signed else None
            
            # Create clinical note
            note = ClinicalNote(
                id=str(uuid.uuid4()),
                patient_id=patient.id,
                admission_id=admission.id,
                note_type=note_type,
                note_title=f"{note_type} - {note_date.strftime('%Y-%m-%d')}",
                note_content=content,
                author=admission.attending_physician,
                author_role='Physician',
                note_date=note_date,
                signed_date=signed_date,
                sections={
                    'subjective': 'Patient reported symptoms',
                    'objective': 'Physical exam findings',
                    'assessment': 'Clinical impression',
                    'plan': 'Treatment plan'
                },
                status='final' if is_signed else 'draft',
                is_verified=is_signed,
                icd10_codes=[f"I{random.randint(10, 99)}.{random.randint(0, 9)}"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            clinical_notes.append(note)
        
        return clinical_notes
    
    def generate_staff(self, count: int = 10) -> List[Staff]:
        """
        Generate hospital staff profiles.
        
        Args:
            count: Number of staff members to generate
            
        Returns:
            List of Staff objects
        """
        staff_members = []
        
        roles = [
            {'role': 'Physician', 'specialties': ['Cardiology', 'Neurology', 'Internal Medicine', 'Emergency']},
            {'role': 'Nurse', 'specialties': ['ICU', 'ER', 'Med-Surg', 'Pediatrics']},
            {'role': 'Administrator', 'specialties': ['Hospital Administration', 'Quality']}
        ]
        
        for i in range(count):
            role_info = random.choice(roles)
            
            first_name = random.choice(self.first_names)
            last_name = random.choice(self.last_names)
            
            staff = Staff(
                id=str(uuid.uuid4()),
                employee_id=f"EMP{random.randint(10000, 99999)}",
                first_name=first_name,
                last_name=last_name,
                email=f"{first_name.lower()}.{last_name.lower()}@hospital.org",
                phone=f"{random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}",
                role=role_info['role'],
                specialty=random.choice(role_info['specialties']) if role_info['specialties'] else None,
                department=random.choice(self.departments),
                license_number=f"LIC{random.randint(100000, 999999)}",
                npi_number=f"{random.randint(1000000000, 9999999999)}",
                permissions=['view_patients', 'create_notes'] if role_info['role'] == 'Nurse' else 
                           ['view_patients', 'create_notes', 'order_tests', 'prescribe'] if role_info['role'] == 'Physician' else
                           ['view_all', 'admin_access'],
                default_shift=random.choice(['Day', 'Evening', 'Night']),
                is_active=True,
                is_on_call=random.random() < 0.3,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            staff_members.append(staff)
        
        return staff_members
    
    def generate_hospital_resources(self) -> List[HospitalResource]:
        """
        Generate hospital resources and equipment.
        
        Returns:
            List of HospitalResource objects
        """
        resources = []
        
        resource_types = [
            {'type': 'Bed', 'departments': self.departments, 'count': 50},
            {'type': 'Ventilator', 'departments': ['ICU', 'Emergency'], 'count': 20},
            {'type': 'Operating Room', 'departments': ['Surgery'], 'count': 10},
            {'type': 'MRI Machine', 'departments': ['Radiology'], 'count': 3},
            {'type': 'CT Scanner', 'departments': ['Radiology'], 'count': 4},
            {'type': 'Ultrasound', 'departments': ['Radiology', 'Emergency'], 'count': 8}
        ]
        
        for resource_info in resource_types:
            for i in range(resource_info['count']):
                department = random.choice(resource_info['departments'])
                
                # Determine resource status (80% available, 15% in use, 5% maintenance)
                status_choice = random.random()
                if status_choice < 0.8:
                    status = 'available'
                    is_available = True
                elif status_choice < 0.95:
                    status = 'in_use'
                    is_available = False
                else:
                    status = 'maintenance'
                    is_available = False
                
                resource = HospitalResource(
                    id=str(uuid.uuid4()),
                    resource_id=f"{resource_info['type'][:3].upper()}{i+1:03d}",
                    resource_type=resource_info['type'],
                    resource_name=f"{resource_info['type']} {i+1}",
                    department=department,
                    location=f"{department} - {random.choice(['Wing A', 'Wing B', 'Main'])}",
                    status=status,
                    is_available=is_available,
                    is_emergency_only=random.random() < 0.1,
                    capacity=1 if resource_info['type'] == 'Bed' else None,
                    last_maintenance=datetime.now() - timedelta(days=random.randint(1, 90)),
                    next_maintenance=datetime.now() + timedelta(days=random.randint(1, 30)),
                    utilization_rate=random.uniform(0, 100) if status == 'in_use' else 0,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                resources.append(resource)
        
        return resources
    
    def load_all_data(self, clear_existing: bool = False):
        """
        Load all sample data into the database.
        
        Args:
            clear_existing: Whether to clear existing data before loading
        """
        try:
            if clear_existing:
                logger.warning("Clearing existing data...")
                # Delete in reverse order to maintain foreign key constraints
                self.session.query(Conversation).delete()
                self.session.query(ClinicalNote).delete()
                self.session.query(MedicationOrder).delete()
                self.session.query(LabResult).delete()
                self.session.query(VitalSign).delete()
                self.session.query(Admission).delete()
                self.session.query(Patient).delete()
                self.session.query(Staff).delete()
                self.session.query(HospitalResource).delete()
                self.session.commit()
                logger.info("Existing data cleared")
            
            # Generate and load staff
            logger.info("Generating staff profiles...")
            staff_members = self.generate_staff(count=15)
            self.session.add_all(staff_members)
            self.session.commit()
            logger.info(f"Added {len(staff_members)} staff members")
            
            # Generate and load hospital resources
            logger.info("Generating hospital resources...")
            resources = self.generate_hospital_resources()
            self.session.add_all(resources)
            self.session.commit()
            logger.info(f"Added {len(resources)} hospital resources")
            
            # Generate patients and related data
            logger.info(f"Generating {self.multipliers['patients']} patients...")
            
            for patient_idx in range(self.multipliers['patients']):
                # Generate patient
                patient = self.generate_patient()
                self.session.add(patient)
                self.session.flush()  # Get patient ID
                
                # Generate admissions for this patient
                num_admissions = random.randint(1, self.multipliers['admissions_per_patient'])
                
                for admission_idx in range(num_admissions):
                    admission = self.generate_admission(patient)
                    self.session.add(admission)
                    self.session.flush()  # Get admission ID
                    
                    # Generate vital signs
                    vitals_count = self.multipliers['vitals_per_admission']
                    vital_signs = self.generate_vital_signs(patient, admission, vitals_count)
                    self.session.add_all(vital_signs)
                    
                    # Generate lab results
                    lab_results = self.generate_lab_results(patient, admission)
                    self.session.add_all(lab_results)
                    
                    # Generate medication orders
                    medications = self.generate_medication_orders(patient, admission)
                    self.session.add_all(medications)
                    
                    # Generate clinical notes
                    notes = self.generate_clinical_notes(patient, admission)
                    self.session.add_all(notes)
                
                # Commit every 10 patients to avoid memory issues
                if (patient_idx + 1) % 10 == 0:
                    self.session.commit()
                    logger.info(f"Committed {patient_idx + 1} patients")
            
            # Final commit for remaining data
            self.session.commit()
            
            logger.info("=" * 60)
            logger.info("Sample data loading completed successfully!")
            logger.info(f"Data size: {self.data_size.upper()}")
            logger.info("Database is populated with realistic sample data")
            logger.info("=" * 60)
            
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error loading sample data: {str(e)}")
            raise

def main():
    """
    Main execution function for sample data loading.
    
    Parses command line arguments and orchestrates the
    sample data generation and loading process.
    """
    parser = argparse.ArgumentParser(
        description='Load sample data into AegisMedBot database'
    )
    parser.add_argument(
        '--size',
        choices=['small', 'medium', 'large'],
        default='medium',
        help='Size of dataset to generate'
    )
    parser.add_argument(
        '--clear-existing',
        action='store_true',
        help='Clear existing data before loading new samples'
    )
    parser.add_argument(
        '--patient-count',
        type=int,
        help='Override number of patients to generate (overrides size setting)'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting AegisMedBot sample data loading")
    logger.info(f"Dataset size: {args.size}")
    logger.info(f"Clear existing: {args.clear_existing}")
    
    # Build database URL
    db_user = os.getenv('POSTGRES_USER', 'medintel')
    db_password = os.getenv('POSTGRES_PASSWORD', 'medintel123')
    db_host = os.getenv('POSTGRES_SERVER', 'localhost')
    db_port = os.getenv('POSTGRES_PORT', '5432')
    db_name = os.getenv('POSTGRES_DB', 'medintel')
    
    database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    # Create engine and session
    engine = create_engine(database_url)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    try:
        # Create data generator
        generator = SampleDataGenerator(session, data_size=args.size)
        
        # Override patient count if specified
        if args.patient_count:
            generator.multipliers['patients'] = args.patient_count
        
        # Load all data
        generator.load_all_data(clear_existing=args.clear_existing)
        
    except Exception as e:
        logger.error(f"Failed to load sample data: {str(e)}")
        sys.exit(1)
    finally:
        session.close()

if __name__ == "__main__":
    main()