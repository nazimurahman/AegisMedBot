"""
Electronic Health Record Data Processor
Handles extraction, transformation, and loading of EHR data for machine learning models.
Includes handling of FHIR format, data validation, and feature engineering.
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import hashlib
import re

logger = logging.getLogger(__name__)

@dataclass
class PatientRecord:
    """
    Data class representing a complete patient record.
    Contains all relevant clinical information for a single patient.
    """
    patient_id: str
    mrn: str  # Medical Record Number
    demographics: Dict[str, Any]
    encounters: List[Dict[str, Any]]
    conditions: List[Dict[str, Any]]
    medications: List[Dict[str, Any]]
    observations: List[Dict[str, Any]]
    procedures: List[Dict[str, Any]]
    lab_results: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class FHIRParser:
    """
    Parse FHIR (Fast Healthcare Interoperability Resources) format data.
    Converts FHIR resources to internal data structures.
    """
    
    def __init__(self):
        """Initialize FHIR parser with resource type mappings."""
        self.resource_handlers = {
            'Patient': self._parse_patient,
            'Encounter': self._parse_encounter,
            'Condition': self._parse_condition,
            'MedicationRequest': self._parse_medication,
            'Observation': self._parse_observation,
            'Procedure': self._parse_procedure
        }
    
    def parse_resource(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a FHIR resource based on its resource type.
        
        Args:
            resource: FHIR resource dictionary
            
        Returns:
            Parsed resource in internal format
        """
        resource_type = resource.get('resourceType')
        handler = self.resource_handlers.get(resource_type)
        
        if handler:
            return handler(resource)
        else:
            logger.warning(f"No handler for resource type: {resource_type}")
            return {}
    
    def _parse_patient(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Parse FHIR Patient resource."""
        name = resource.get('name', [{}])[0]
        birth_date = resource.get('birthDate')
        gender = resource.get('gender')
        
        return {
            'id': resource.get('id'),
            'first_name': name.get('given', [''])[0],
            'last_name': name.get('family', ''),
            'date_of_birth': birth_date,
            'gender': gender,
            'identifier': resource.get('identifier', [])
        }
    
    def _parse_encounter(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Parse FHIR Encounter resource."""
        period = resource.get('period', {})
        return {
            'id': resource.get('id'),
            'status': resource.get('status'),
            'class': resource.get('class', {}).get('code'),
            'type': resource.get('type', [{}])[0].get('coding', [{}])[0].get('code'),
            'start_date': period.get('start'),
            'end_date': period.get('end'),
            'hospitalization': resource.get('hospitalization', {})
        }
    
    def _parse_condition(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Parse FHIR Condition resource."""
        code = resource.get('code', {})
        return {
            'id': resource.get('id'),
            'code': code.get('coding', [{}])[0].get('code'),
            'display': code.get('text', ''),
            'clinical_status': resource.get('clinicalStatus', {}).get('coding', [{}])[0].get('code'),
            'verification_status': resource.get('verificationStatus', {}).get('coding', [{}])[0].get('code'),
            'onset_date': resource.get('onsetDateTime'),
            'abatement_date': resource.get('abatementDateTime')
        }
    
    def _parse_medication(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Parse FHIR MedicationRequest resource."""
        medication = resource.get('medicationCodeableConcept', {})
        dosage = resource.get('dosageInstruction', [{}])[0]
        
        return {
            'id': resource.get('id'),
            'status': resource.get('status'),
            'intent': resource.get('intent'),
            'medication_code': medication.get('coding', [{}])[0].get('code'),
            'medication_name': medication.get('text', ''),
            'dosage_text': dosage.get('text', ''),
            'route': dosage.get('route', {}).get('text', ''),
            'authored_date': resource.get('authoredOn')
        }
    
    def _parse_observation(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Parse FHIR Observation resource."""
        code = resource.get('code', {})
        value = resource.get('valueQuantity', {})
        
        return {
            'id': resource.get('id'),
            'status': resource.get('status'),
            'code': code.get('coding', [{}])[0].get('code'),
            'display': code.get('text', ''),
            'value': value.get('value'),
            'unit': value.get('unit'),
            'effective_date': resource.get('effectiveDateTime'),
            'interpretation': resource.get('interpretation', [{}])[0].get('coding', [{}])[0].get('code')
        }
    
    def _parse_procedure(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Parse FHIR Procedure resource."""
        code = resource.get('code', {})
        performed = resource.get('performedDateTime')
        
        return {
            'id': resource.get('id'),
            'status': resource.get('status'),
            'code': code.get('coding', [{}])[0].get('code'),
            'display': code.get('text', ''),
            'performed_date': performed,
            'body_site': resource.get('bodySite', [{}])[0].get('text', '')
        }


class EHRDataValidator:
    """
    Validate EHR data for completeness and consistency.
    Ensures data quality before using for model training.
    """
    
    def __init__(self):
        """Initialize validator with validation rules."""
        self.required_fields = {
            'patient': ['patient_id', 'mrn', 'date_of_birth'],
            'encounter': ['patient_id', 'start_date', 'type'],
            'condition': ['patient_id', 'code', 'clinical_status'],
            'medication': ['patient_id', 'medication_code', 'status'],
            'observation': ['patient_id', 'code', 'value']
        }
        
        self.value_ranges = {
            'heart_rate': (30, 220),
            'systolic_bp': (70, 250),
            'diastolic_bp': (40, 150),
            'temperature': (35.0, 42.0),
            'oxygen_saturation': (70, 100),
            'respiratory_rate': (8, 40),
            'glucose': (40, 600),
            'creatinine': (0.2, 10.0),
            'hemoglobin': (5, 20)
        }
    
    def validate_patient(self, patient: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate patient demographic data.
        
        Args:
            patient: Patient data dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        for field in self.required_fields['patient']:
            if field not in patient or not patient[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate date of birth
        if 'date_of_birth' in patient and patient['date_of_birth']:
            try:
                dob = datetime.strptime(patient['date_of_birth'], '%Y-%m-%d')
                if dob > datetime.now():
                    errors.append("Date of birth cannot be in the future")
            except (ValueError, TypeError):
                errors.append("Invalid date of birth format")
        
        # Validate gender
        if 'gender' in patient and patient['gender']:
            if patient['gender'] not in ['M', 'F', 'O', 'U']:
                errors.append(f"Invalid gender value: {patient['gender']}")
        
        return len(errors) == 0, errors
    
    def validate_observation(
        self,
        observation: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate clinical observation data.
        
        Args:
            observation: Observation data dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        for field in self.required_fields['observation']:
            if field not in observation or observation[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate value ranges if applicable
        code = observation.get('code', '').lower()
        value = observation.get('value')
        
        if code in self.value_ranges and value is not None:
            low, high = self.value_ranges[code]
            if value < low or value > high:
                errors.append(
                    f"Value {value} for {code} is outside range [{low}, {high}]"
                )
        
        # Validate timestamp
        if 'effective_date' in observation and observation['effective_date']:
            try:
                datetime.fromisoformat(observation['effective_date'].replace('Z', '+00:00'))
            except (ValueError, TypeError):
                errors.append("Invalid effective_date format")
        
        return len(errors) == 0, errors
    
    def validate_encounter(self, encounter: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate encounter data.
        
        Args:
            encounter: Encounter data dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        for field in self.required_fields['encounter']:
            if field not in encounter or not encounter[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate dates
        if 'start_date' in encounter and encounter['start_date']:
            try:
                start = datetime.fromisoformat(encounter['start_date'].replace('Z', '+00:00'))
                if 'end_date' in encounter and encounter['end_date']:
                    end = datetime.fromisoformat(encounter['end_date'].replace('Z', '+00:00'))
                    if end < start:
                        errors.append("End date cannot be before start date")
            except (ValueError, TypeError):
                errors.append("Invalid date format")
        
        return len(errors) == 0, errors
    
    def validate_medication(self, medication: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate medication data.
        
        Args:
            medication: Medication data dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        for field in self.required_fields['medication']:
            if field not in medication or not medication[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate status
        valid_statuses = ['active', 'completed', 'stopped', 'entered-in-error']
        if 'status' in medication and medication['status']:
            if medication['status'].lower() not in valid_statuses:
                errors.append(f"Invalid medication status: {medication['status']}")
        
        return len(errors) == 0, errors


class EHRFeatureEngineer:
    """
    Feature engineering for EHR data.
    Creates derived features that capture clinical patterns and risk factors.
    """
    
    def __init__(self, time_window_days: int = 365):
        """
        Initialize feature engineer.
        
        Args:
            time_window_days: Window for aggregating historical features
        """
        self.time_window_days = time_window_days
        self.feature_cache = {}
    
    def create_demographic_features(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create features from demographic information.
        
        Args:
            patient: Patient data dictionary
            
        Returns:
            Dictionary of demographic features
        """
        features = {}
        
        # Age at admission
        if 'date_of_birth' in patient and patient['date_of_birth']:
            try:
                dob = datetime.strptime(patient['date_of_birth'], '%Y-%m-%d')
                age = (datetime.now() - dob).days / 365.25
                features['age'] = age
                features['age_group'] = self._categorize_age(age)
            except (ValueError, TypeError):
                features['age'] = None
                features['age_group'] = 'unknown'
        else:
            features['age'] = None
            features['age_group'] = 'unknown'
        
        # Gender encoding
        gender = patient.get('gender', 'U')
        features['gender_male'] = 1 if gender == 'M' else 0
        features['gender_female'] = 1 if gender == 'F' else 0
        features['gender_other'] = 1 if gender not in ['M', 'F'] else 0
        
        return features
    
    def _categorize_age(self, age: float) -> str:
        """Categorize age into groups for analysis."""
        if age < 18:
            return 'pediatric'
        elif age < 65:
            return 'adult'
        else:
            return 'geriatric'
    
    def create_comorbidity_features(
        self,
        conditions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create comorbidity-related features.
        
        Args:
            conditions: List of condition records
            
        Returns:
            Dictionary of comorbidity features
        """
        features = {
            'total_conditions': len(conditions),
            'active_conditions': 0,
            'chronic_conditions': 0,
            'acute_conditions': 0
        }
        
        chronic_codes = {
            'E11', 'I10', 'J44', 'I50', 'N18',  # Diabetes, Hypertension, COPD, HF, CKD
            'E10', 'E13', 'I11', 'J45', 'K70'    # Type 1 diabetes, etc.
        }
        
        for condition in conditions:
            status = condition.get('clinical_status', '')
            code = condition.get('code', '')
            
            if status == 'active':
                features['active_conditions'] += 1
            
            if code in chronic_codes:
                features['chronic_conditions'] += 1
            else:
                features['acute_conditions'] += 1
        
        # Calculate Charlson Comorbidity Index (simplified)
        features['charlson_score'] = self._calculate_charlson_score(conditions)
        
        return features
    
    def _calculate_charlson_score(self, conditions: List[Dict[str, Any]]) -> int:
        """
        Calculate simplified Charlson Comorbidity Index.
        
        Args:
            conditions: List of condition records
            
        Returns:
            Charlson score
        """
        score_map = {
            'I21': 1,  # Myocardial infarction
            'I50': 1,  # Congestive heart failure
            'I70': 1,  # Peripheral vascular disease
            'I60': 1,  # Cerebrovascular disease
            'F00': 1,  # Dementia
            'J44': 1,  # COPD
            'M05': 1,  # Rheumatologic disease
            'K70': 1,  # Liver disease
            'E10': 1,  # Diabetes
            'N18': 2,  # Renal disease
            'C00': 2,  # Cancer
            'K72': 3,  # Severe liver disease
            'C80': 6   # Metastatic cancer
        }
        
        score = 0
        for condition in conditions:
            code = condition.get('code', '')
            for pattern, value in score_map.items():
                if code.startswith(pattern):
                    score += value
                    break
        
        return min(score, 10)  # Cap at 10
    
    def create_temporal_features(
        self,
        observations: List[Dict[str, Any]],
        feature_name: str
    ) -> Dict[str, Any]:
        """
        Create temporal features from observation time series.
        
        Args:
            observations: List of observation records
            feature_name: Name of the observation to analyze
            
        Returns:
            Dictionary of temporal features
        """
        # Filter observations by feature name
        relevant_obs = [
            obs for obs in observations
            if obs.get('code', '').lower() == feature_name.lower()
        ]
        
        if not relevant_obs:
            return {}
        
        # Extract values and timestamps
        values = []
        timestamps = []
        
        for obs in relevant_obs:
            value = obs.get('value')
            timestamp = obs.get('effective_date')
            
            if value is not None and timestamp:
                values.append(value)
                try:
                    timestamps.append(
                        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    )
                except (ValueError, TypeError):
                    continue
        
        if not values:
            return {}
        
        # Calculate statistical features
        features = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'count': len(values)
        }
        
        # Calculate trend if enough points
        if len(values) >= 2:
            # Simple linear trend
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            features['trend'] = slope
        
        # Calculate recent trend (last 3 measurements)
        if len(values) >= 3:
            recent_slope = (values[-1] - values[-3]) / 2
            features['recent_trend'] = recent_slope
        
        # Calculate variability (coefficient of variation)
        if features['mean'] != 0:
            features['cv'] = features['std'] / features['mean']
        
        return features
    
    def create_lab_trend_features(
        self,
        lab_results: List[Dict[str, Any]],
        time_window_days: int = 90
    ) -> Dict[str, Any]:
        """
        Create features from lab results with temporal trends.
        
        Args:
            lab_results: List of lab result records
            time_window_days: Time window for aggregation
            
        Returns:
            Dictionary of lab trend features
        """
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        features = {}
        
        # Group lab results by test name
        lab_groups = {}
        for lab in lab_results:
            test_name = lab.get('display', lab.get('code', 'unknown'))
            if test_name not in lab_groups:
                lab_groups[test_name] = []
            
            timestamp = lab.get('effective_date')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    if dt >= cutoff_date:
                        lab_groups[test_name].append({
                            'value': lab.get('value'),
                            'timestamp': dt,
                            'interpretation': lab.get('interpretation')
                        })
                except (ValueError, TypeError):
                    continue
        
        # Create features for each lab test
        for test_name, measurements in lab_groups.items():
            if not measurements:
                continue
            
            values = [m['value'] for m in measurements if m['value'] is not None]
            if not values:
                continue
            
            prefix = f"lab_{test_name.lower().replace(' ', '_')}"
            features[f"{prefix}_latest"] = values[-1]
            features[f"{prefix}_mean"] = np.mean(values)
            features[f"{prefix}_std"] = np.std(values) if len(values) > 1 else 0
            
            # Abnormality flags
            abnormal_count = sum(
                1 for m in measurements
                if m.get('interpretation') in ['H', 'L', 'HH', 'LL']
            )
            features[f"{prefix}_abnormal_count"] = abnormal_count
            features[f"{prefix}_abnormal_ratio"] = abnormal_count / len(measurements)
            
            # Trend
            if len(values) >= 2:
                x = np.arange(len(values))
                slope, _ = np.polyfit(x, values, 1)
                features[f"{prefix}_trend"] = slope
        
        return features


class EHRDataLoader:
    """
    Load and manage EHR data from various sources.
    Provides unified interface for accessing patient data.
    """
    
    def __init__(self, data_directory: str = None):
        """
        Initialize EHR data loader.
        
        Args:
            data_directory: Directory containing EHR data files
        """
        self.data_directory = data_directory
        self.patients = {}
        self.fhir_parser = FHIRParser()
        self.validator = EHRDataValidator()
        
    def load_from_json(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load EHR data from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of patient records
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'patients' in data:
            return data['patients']
        else:
            return [data]
    
    def load_from_fhir_bundle(self, bundle: Dict[str, Any]) -> List[PatientRecord]:
        """
        Load patient records from FHIR bundle.
        
        Args:
            bundle: FHIR bundle dictionary
            
        Returns:
            List of parsed patient records
        """
        patients = {}
        entries = bundle.get('entry', [])
        
        # First pass: collect all resources
        for entry in entries:
            resource = entry.get('resource', {})
            resource_type = resource.get('resourceType')
            patient_id = None
            
            # Get patient reference
            if resource_type == 'Patient':
                patient_id = resource.get('id')
            else:
                # Extract patient reference from subject or patient field
                subject = resource.get('subject', {})
                reference = subject.get('reference', '')
                if reference.startswith('Patient/'):
                    patient_id = reference.split('/')[1]
            
            if not patient_id:
                continue
            
            # Initialize patient record if not exists
            if patient_id not in patients:
                patients[patient_id] = {
                    'patient_id': patient_id,
                    'mrn': '',
                    'demographics': {},
                    'encounters': [],
                    'conditions': [],
                    'medications': [],
                    'observations': [],
                    'procedures': []
                }
            
            # Parse and add resource
            parsed = self.fhir_parser.parse_resource(resource)
            if parsed:
                if resource_type == 'Patient':
                    patients[patient_id]['demographics'] = parsed
                    patients[patient_id]['mrn'] = parsed.get('identifier', [{}])[0].get('value', '')
                elif resource_type == 'Encounter':
                    patients[patient_id]['encounters'].append(parsed)
                elif resource_type == 'Condition':
                    patients[patient_id]['conditions'].append(parsed)
                elif resource_type == 'MedicationRequest':
                    patients[patient_id]['medications'].append(parsed)
                elif resource_type == 'Observation':
                    patients[patient_id]['observations'].append(parsed)
                elif resource_type == 'Procedure':
                    patients[patient_id]['procedures'].append(parsed)
        
        # Convert to PatientRecord objects
        patient_records = []
        for patient_id, data in patients.items():
            # Validate patient data
            is_valid, errors = self.validator.validate_patient(data['demographics'])
            if not is_valid:
                logger.warning(f"Invalid patient {patient_id}: {errors}")
            
            patient_records.append(PatientRecord(**data))
        
        return patient_records
    
    def get_patient_by_id(self, patient_id: str) -> Optional[PatientRecord]:
        """
        Retrieve patient by ID.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            PatientRecord if found, None otherwise
        """
        return self.patients.get(patient_id)
    
    def create_cohort(
        self,
        condition_codes: List[str],
        start_date: str = None,
        end_date: str = None
    ) -> List[str]:
        """
        Create a cohort of patients with specific conditions.
        
        Args:
            condition_codes: List of condition codes to include
            start_date: Earliest date for inclusion
            end_date: Latest date for inclusion
            
        Returns:
            List of patient IDs in the cohort
        """
        cohort = []
        
        for patient_id, patient in self.patients.items():
            # Check conditions
            has_condition = False
            for condition in patient.conditions:
                if condition.get('code') in condition_codes:
                    # Check dates if provided
                    if start_date or end_date:
                        onset = condition.get('onset_date')
                        if onset:
                            onset_dt = datetime.fromisoformat(onset.replace('Z', '+00:00'))
                            if start_date:
                                start_dt = datetime.fromisoformat(start_date)
                                if onset_dt < start_dt:
                                    continue
                            if end_date:
                                end_dt = datetime.fromisoformat(end_date)
                                if onset_dt > end_dt:
                                    continue
                    has_condition = True
                    break
            
            if has_condition:
                cohort.append(patient_id)
        
        return cohort
    
    def get_patient_features(
        self,
        patient_id: str,
        feature_engineer: EHRFeatureEngineer
    ) -> Dict[str, Any]:
        """
        Extract all features for a patient.
        
        Args:
            patient_id: Patient identifier
            feature_engineer: Feature engineer instance
            
        Returns:
            Dictionary of extracted features
        """
        patient = self.get_patient_by_id(patient_id)
        if not patient:
            return {}
        
        features = {}
        
        # Demographic features
        features.update(
            feature_engineer.create_demographic_features(patient.demographics)
        )
        
        # Comorbidity features
        features.update(
            feature_engineer.create_comorbidity_features(patient.conditions)
        )
        
        # Temporal features for key observations
        key_observations = ['heart_rate', 'systolic_bp', 'temperature', 'oxygen_saturation']
        for obs in key_observations:
            temporal = feature_engineer.create_temporal_features(
                patient.observations, obs
            )
            for key, value in temporal.items():
                features[f"{obs}_{key}"] = value
        
        # Lab trend features
        lab_features = feature_engineer.create_lab_trend_features(
            patient.lab_results
        )
        features.update(lab_features)
        
        return features
    
    def get_batch_features(
        self,
        patient_ids: List[str],
        feature_engineer: EHRFeatureEngineer
    ) -> pd.DataFrame:
        """
        Extract features for a batch of patients.
        
        Args:
            patient_ids: List of patient IDs
            feature_engineer: Feature engineer instance
            
        Returns:
            DataFrame with features for all patients
        """
        all_features = []
        
        for patient_id in patient_ids:
            features = self.get_patient_features(patient_id, feature_engineer)
            features['patient_id'] = patient_id
            all_features.append(features)
        
        return pd.DataFrame(all_features)


# Example usage
if __name__ == "__main__":
    # Test FHIR parser
    fhir_parser = FHIRParser()
    sample_patient = {
        "resourceType": "Patient",
        "id": "example",
        "name": [{"family": "Smith", "given": ["John"]}],
        "gender": "male",
        "birthDate": "1974-12-25"
    }
    parsed = fhir_parser.parse_resource(sample_patient)
    print(f"Parsed patient: {parsed}")
    
    # Test validator
    validator = EHRDataValidator()
    is_valid, errors = validator.validate_patient(parsed)
    print(f"Valid: {is_valid}, Errors: {errors}")
    
    # Test feature engineer
    engineer = EHRFeatureEngineer()
    demo_features = engineer.create_demographic_features(parsed)
    print(f"Demographic features: {demo_features}")