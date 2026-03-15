"""
Privacy Guardian Agent Module
Responsible for enforcing privacy policies, data protection, and compliance
with healthcare regulations including HIPAA, GDPR, and institutional policies.
"""

import re
import hashlib
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict
import threading
from dataclasses import dataclass, field

# Configure logging for compliance tracking
logger = logging.getLogger(__name__)

class DataClassification(Enum):
    """
    Enumeration for data sensitivity classification levels
    Determines how data should be handled and protected
    """
    PUBLIC = 1          # Publicly available information
    INTERNAL = 2        # Internal hospital information
    SENSITIVE = 3       # Sensitive but not PHI
    PHI = 4            # Protected Health Information (highest protection)
    RESTRICTED = 5      # Highly restricted data (mental health, HIV status, etc.)

class AccessLevel(Enum):
    """
    User access levels for role-based access control
    Aligns with hospital role hierarchies
    """
    PATIENT = "patient"          # Can only access own data
    NURSE = "nurse"              # Access to assigned patients
    DOCTOR = "doctor"            # Full clinical access
    ADMIN = "admin"              # Administrative access
    COMPLIANCE = "compliance"    # Audit and compliance access
    RESEARCH = "research"        # De-identified data for research

@dataclass
class PrivacyPolicy:
    """
    Data class defining privacy policies and rules
    Encapsulates all privacy-related configuration
    """
    name: str
    description: str
    rules: Dict[str, Any]
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class AccessRequest:
    """
    Data class for tracking data access requests
    Used for audit logging and access monitoring
    """
    request_id: str
    user_id: str
    user_role: str
    resource_type: str
    resource_id: str
    access_type: str  # read, write, delete
    timestamp: datetime
    granted: bool
    reason: Optional[str] = None
    data_classification: Optional[DataClassification] = None

class PrivacyGuardian:
    """
    Main Privacy Guardian class responsible for:
    1. Enforcing privacy policies on all data access
    2. Implementing role-based access control (RBAC)
    3. Data minimization and purpose limitation
    4. Consent management for patient data
    5. Breach detection and prevention
    6. Privacy policy enforcement
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Privacy Guardian with configuration settings
        
        Args:
            config: Dictionary containing privacy configuration parameters
                   Includes policy settings, retention periods, etc.
        """
        self.config = config or {}
        
        # Initialize core components
        self.phi_detector = PHIDetector()  # Detect PHI in data
        self.audit_logger = AuditLogger()   # Log all privacy events
        
        # Privacy policy storage
        self.policies: Dict[str, PrivacyPolicy] = {}
        self._initialize_default_policies()
        
        # Access control lists
        self.access_grants: Dict[str, Set[str]] = defaultdict(set)
        self.consent_records: Dict[str, Dict] = {}
        
        # Data retention settings
        self.retention_periods = {
            DataClassification.PUBLIC: timedelta(days=365),
            DataClassification.INTERNAL: timedelta(days=180),
            DataClassification.SENSITIVE: timedelta(days=90),
            DataClassification.PHI: timedelta(days=30),
            DataClassification.RESTRICTED: timedelta(days=7)
        }
        
        # Breach detection thresholds
        self.breach_thresholds = {
            "max_failed_access": 5,           # Max failed access attempts
            "max_data_volume": 1000,           # Max records accessed per minute
            "suspicious_time_window": 60,       # Seconds for anomaly detection
            "max_sensitive_queries": 10         # Max sensitive queries per hour
        }
        
        # Initialize breach detection counters
        self.access_counter = defaultdict(int)
        self.failed_access_counter = defaultdict(int)
        self.sensitive_query_counter = defaultdict(int)
        
        # Thread safety for concurrent access
        self._lock = threading.RLock()
        
        logger.info("Privacy Guardian initialized with %d policies", len(self.policies))
    
    def _initialize_default_policies(self):
        """
        Initialize default privacy policies based on healthcare regulations
        Creates baseline policies that all healthcare systems must follow
        """
        default_policies = [
            PrivacyPolicy(
                name="minimum_necessary",
                description="Only access minimum necessary data for the task",
                rules={
                    "enforce_field_level": True,
                    "max_fields_per_query": 10,
                    "require_justification": True
                }
            ),
            PrivacyPolicy(
                name="purpose_limitation",
                description="Data can only be used for specified purposes",
                rules={
                    "purposes": ["treatment", "payment", "operations", "research"],
                    "require_consent_for_research": True
                }
            ),
            PrivacyPolicy(
                name="data_retention",
                description="Data must be deleted after retention period",
                rules={
                    "enforce_deletion": True,
                    "archive_before_deletion": True
                }
            ),
            PrivacyPolicy(
                name="breach_notification",
                description="Notify on potential data breaches",
                rules={
                    "notify_compliance_team": True,
                    "notify_affected_patients": True,
                    "notification_delay_hours": 72
                }
            ),
            PrivacyPolicy(
                name="access_audit",
                description="Log all data access for audit purposes",
                rules={
                    "log_all_access": True,
                    "retain_logs_days": 365,
                    "encrypt_logs": True
                }
            ),
            PrivacyPolicy(
                name="consent_management",
                description="Respect patient consent preferences",
                rules={
                    "check_consent_before_access": True,
                    "allow_consent_withdrawal": True,
                    "honor_opt_outs": True
                }
            )
        ]
        
        for policy in default_policies:
            self.policies[policy.name] = policy
    
    async def enforce_privacy(
        self,
        data: Dict[str, Any],
        user_context: Dict[str, Any],
        purpose: str,
        conversation_id: Optional[str] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Main method to enforce privacy on all data access
        Applies all privacy policies and returns sanitized data with audit trail
        
        Args:
            data: The data to be accessed (can contain PHI)
            user_context: Information about the user requesting access
            purpose: Why the data is being accessed (treatment, research, etc.)
            conversation_id: Optional ID for tracking conversation context
        
        Returns:
            Tuple containing:
                - Sanitized data with PHI removed/redacted
                - List of privacy actions taken for audit
        """
        privacy_actions = []
        sanitized_data = data.copy() if isinstance(data, dict) else {"data": data}
        
        try:
            # Step 1: Validate purpose is allowed
            purpose_valid, purpose_action = await self._validate_purpose(purpose, user_context)
            privacy_actions.append(purpose_action)
            
            if not purpose_valid:
                logger.warning(f"Invalid purpose '{purpose}' for user {user_context.get('user_id')}")
                return {"error": "Access purpose not authorized"}, privacy_actions
            
            # Step 2: Apply minimum necessary rule
            sanitized_data, min_necessary_action = await self._apply_minimum_necessary(
                sanitized_data, user_context, purpose
            )
            privacy_actions.append(min_necessary_action)
            
            # Step 3: Check user permissions
            permission_check, permission_action = await self._check_permissions(
                user_context, sanitized_data
            )
            privacy_actions.append(permission_action)
            
            if not permission_check["granted"]:
                return {"error": permission_check["reason"]}, privacy_actions
            
            # Step 4: Apply data classification and handling
            classified_data, class_action = await self._classify_and_protect_data(
                sanitized_data, user_context
            )
            privacy_actions.append(class_action)
            
            # Step 5: Check patient consent if applicable
            if "patient_id" in data:
                consent_check, consent_action = await self._check_patient_consent(
                    data["patient_id"], purpose, user_context
                )
                privacy_actions.append(consent_action)
                
                if not consent_check["allowed"]:
                    return {"error": "Patient consent not granted"}, privacy_actions
            
            # Step 6: Apply data retention check
            retention_action = await self._check_retention_policy(data)
            privacy_actions.append(retention_action)
            
            # Step 7: Log the access for audit
            access_record = await self._create_access_record(
                user_context, data, purpose, "granted"
            )
            await self.audit_logger.log_privacy_event(access_record)
            privacy_actions.append({
                "action": "access_logged",
                "timestamp": datetime.now().isoformat(),
                "record_id": access_record.get("request_id")
            })
            
            # Step 8: Update breach detection counters
            await self._update_breach_counters(user_context, True)
            
            logger.info(f"Privacy enforcement completed for user {user_context.get('user_id')}")
            return classified_data, privacy_actions
            
        except Exception as e:
            logger.error(f"Privacy enforcement failed: {str(e)}", exc_info=True)
            
            # Log the error for audit
            error_record = {
                "error": str(e),
                "user_context": user_context,
                "timestamp": datetime.now().isoformat()
            }
            await self.audit_logger.log_error(error_record)
            
            return {"error": "Privacy enforcement failed"}, privacy_actions
    
    async def _validate_purpose(
        self,
        purpose: str,
        user_context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that the access purpose is permitted
        
        Args:
            purpose: Purpose of data access
            user_context: User information
        
        Returns:
            Tuple of (is_valid, action_record)
        """
        allowed_purposes = self.policies["purpose_limitation"].rules["purposes"]
        
        is_valid = purpose in allowed_purposes
        
        action = {
            "action": "purpose_validation",
            "purpose": purpose,
            "is_valid": is_valid,
            "timestamp": datetime.now().isoformat(),
            "user_role": user_context.get("role")
        }
        
        if not is_valid:
            action["reason"] = f"Purpose '{purpose}' not in allowed list: {allowed_purposes}"
        
        return is_valid, action
    
    async def _apply_minimum_necessary(
        self,
        data: Dict[str, Any],
        user_context: Dict[str, Any],
        purpose: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Apply minimum necessary rule - only return fields user needs
        
        Implements field-level access control based on:
        - User role
        - Access purpose
        - Data sensitivity
        """
        policy = self.policies["minimum_necessary"]
        
        # Define field access by role and purpose
        field_access_rules = {
            "doctor": {
                "treatment": ["all"],  # Doctors can see everything for treatment
                "research": ["demographics", "diagnosis_codes", "lab_results_anon"],
                "operations": ["admission_date", "department", "length_of_stay"]
            },
            "nurse": {
                "treatment": ["vitals", "medications", "allergies", "care_plan"],
                "operations": ["bed_number", "shift_notes"],
                "research": []  # Nurses don't access research data
            },
            "admin": {
                "operations": ["admission_date", "discharge_date", "insurance", "billing"],
                "payment": ["insurance", "billing_codes"]
            },
            "research": {
                "research": ["demographics_anon", "diagnosis_codes", "outcomes"]
            }
        }
        
        # Get allowed fields for this role and purpose
        role_rules = field_access_rules.get(
            user_context.get("role", "patient"),
            {}
        )
        allowed_fields = role_rules.get(purpose, [])
        
        # Apply field filtering
        filtered_data = {}
        removed_fields = []
        
        if "all" in allowed_fields:
            filtered_data = data.copy()
        else:
            for field, value in data.items():
                if field in allowed_fields:
                    filtered_data[field] = value
                else:
                    removed_fields.append(field)
        
        # If policy requires field-level logging
        if policy.rules.get("enforce_field_level"):
            for field in removed_fields:
                logger.debug(f"Field '{field}' removed for minimum necessary compliance")
        
        action = {
            "action": "minimum_necessary_applied",
            "original_fields": list(data.keys()),
            "returned_fields": list(filtered_data.keys()),
            "removed_fields": removed_fields,
            "timestamp": datetime.now().isoformat()
        }
        
        return filtered_data, action
    
    async def _check_permissions(
        self,
        user_context: Dict[str, Any],
        data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Check if user has permission to access this data
        Implements Role-Based Access Control (RBAC)
        
        Returns:
            Tuple of (permission_result, action_record)
        """
        user_role = user_context.get("role", "patient")
        user_id = user_context.get("user_id")
        
        # Define role permissions hierarchy
        role_hierarchy = {
            "patient": 1,
            "nurse": 2,
            "doctor": 3,
            "admin": 4,
            "compliance": 5
        }
        
        # Check if user has any special grants for this data
        has_special_grant = False
        if "patient_id" in data and data["patient_id"] in self.access_grants.get(user_id, set()):
            has_special_grant = True
        
        # Determine permission level
        user_level = role_hierarchy.get(user_role, 0)
        
        # Data sensitivity level
        data_sensitivity = self._determine_data_sensitivity(data)
        required_level = self._sensitivity_to_level(data_sensitivity)
        
        granted = user_level >= required_level or has_special_grant
        reason = None
        
        if not granted:
            reason = f"Insufficient permissions: {user_role} (level {user_level}) needs level {required_level}"
            # Update failed access counter
            self.failed_access_counter[user_id] += 1
            
            # Check if breach threshold exceeded
            if self.failed_access_counter[user_id] > self.breach_thresholds["max_failed_access"]:
                await self._trigger_breach_alert({
                    "type": "excessive_failed_access",
                    "user_id": user_id,
                    "count": self.failed_access_counter[user_id]
                })
        
        result = {
            "granted": granted,
            "reason": reason,
            "user_level": user_level,
            "required_level": required_level,
            "has_special_grant": has_special_grant
        }
        
        action = {
            "action": "permission_check",
            **result,
            "timestamp": datetime.now().isoformat()
        }
        
        return result, action
    
    def _determine_data_sensitivity(self, data: Dict[str, Any]) -> DataClassification:
        """
        Determine sensitivity level of data
        Uses PHI detector to identify sensitive information
        """
        # Convert data to string for PHI detection
        data_str = json.dumps(data)
        
        # Use PHI detector to find sensitive content
        phi_results = self.phi_detector.detect_phi(data_str)
        
        if phi_results["has_phi"]:
            # Check for specially restricted categories
            restricted_categories = ["mental_health", "hiv_status", "genetic_data"]
            for category in restricted_categories:
                if category in phi_results.get("categories", []):
                    return DataClassification.RESTRICTED
            return DataClassification.PHI
        elif any(keyword in data_str.lower() for keyword in ["salary", "ssn", "credit"]):
            return DataClassification.SENSITIVE
        elif "patient_id" in data or "mrn" in data:
            return DataClassification.INTERNAL
        else:
            return DataClassification.PUBLIC
    
    def _sensitivity_to_level(self, classification: DataClassification) -> int:
        """Convert data classification to required access level"""
        mapping = {
            DataClassification.PUBLIC: 1,
            DataClassification.INTERNAL: 2,
            DataClassification.SENSITIVE: 3,
            DataClassification.PHI: 4,
            DataClassification.RESTRICTED: 5
        }
        return mapping.get(classification, 1)
    
    async def _classify_and_protect_data(
        self,
        data: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Classify data and apply appropriate protection measures
        Handles PHI detection and redaction
        """
        protected_data = data.copy()
        protection_actions = []
        
        # Convert to string for PHI detection
        data_str = json.dumps(protected_data)
        
        # Detect PHI
        phi_results = self.phi_detector.detect_phi(data_str)
        
        if phi_results["has_phi"]:
            # For each PHI entity found, apply redaction or masking
            for entity in phi_results.get("entities", []):
                entity_text = entity["text"]
                entity_type = entity["type"]
                
                # Determine protection strategy based on user role and entity type
                protection_strategy = self._get_protection_strategy(
                    entity_type,
                    user_context.get("role")
                )
                
                if protection_strategy == "redact":
                    # Completely remove PHI
                    protected_data = self._redact_entity(protected_data, entity_text)
                    protection_actions.append({
                        "entity_type": entity_type,
                        "action": "redacted"
                    })
                elif protection_strategy == "mask":
                    # Partially mask PHI (e.g., show last 4 digits)
                    protected_data = self._mask_entity(protected_data, entity_text, entity_type)
                    protection_actions.append({
                        "entity_type": entity_type,
                        "action": "masked"
                    })
                elif protection_strategy == "hash":
                    # Replace with irreversible hash
                    protected_data = self._hash_entity(protected_data, entity_text)
                    protection_actions.append({
                        "entity_type": entity_type,
                        "action": "hashed"
                    })
        
        action = {
            "action": "data_classification",
            "classification": self._determine_data_sensitivity(data).name,
            "phi_detected": phi_results["has_phi"],
            "protection_actions": protection_actions,
            "timestamp": datetime.now().isoformat()
        }
        
        return protected_data, action
    
    def _get_protection_strategy(self, entity_type: str, user_role: str) -> str:
        """
        Determine protection strategy for PHI based on entity type and user role
        
        Strategies:
        - redact: Completely remove (for unauthorized users)
        - mask: Show partial (e.g., last 4 digits)
        - hash: Replace with hash (for research)
        - allow: Show full (for authorized clinicians)
        """
        # Default strategies by entity type
        default_strategies = {
            "name": "redact",
            "mrn": "mask",
            "ssn": "redact",
            "phone": "mask",
            "email": "mask",
            "address": "redact",
            "date": "mask",
            "insurance_id": "mask"
        }
        
        # Role-based overrides
        if user_role in ["doctor", "nurse"]:
            # Clinicians can see more
            clinician_allowed = ["name", "mrn", "date"]
            if entity_type in clinician_allowed:
                return "allow"
        
        if user_role == "research":
            # Research can use hashed identifiers
            if entity_type in ["mrn", "patient_id"]:
                return "hash"
        
        return default_strategies.get(entity_type, "redact")
    
    def _redact_entity(self, data: Dict, entity_text: str) -> Dict:
        """Completely remove PHI entity from data"""
        data_str = json.dumps(data)
        # Replace with [REDACTED]
        data_str = data_str.replace(entity_text, "[REDACTED]")
        return json.loads(data_str)
    
    def _mask_entity(self, data: Dict, entity_text: str, entity_type: str) -> Dict:
        """Partially mask PHI entity"""
        data_str = json.dumps(data)
        
        if entity_type in ["ssn", "mrn", "insurance_id"]:
            # Show last 4 digits only
            if len(entity_text) >= 4:
                masked = "***-**-" + entity_text[-4:]
                data_str = data_str.replace(entity_text, masked)
        elif entity_type == "phone":
            # Show last 4 digits
            if len(entity_text) >= 4:
                masked = "XXX-XXX-" + entity_text[-4:]
                data_str = data_str.replace(entity_text, masked)
        elif entity_type == "email":
            # Show first character and domain
            if "@" in entity_text:
                local, domain = entity_text.split("@")
                masked = local[0] + "***@" + domain
                data_str = data_str.replace(entity_text, masked)
        else:
            # Default masking
            if len(entity_text) > 3:
                masked = entity_text[0] + "*" * (len(entity_text) - 2) + entity_text[-1]
                data_str = data_str.replace(entity_text, masked)
        
        return json.loads(data_str)
    
    def _hash_entity(self, data: Dict, entity_text: str) -> Dict:
        """Replace entity with SHA-256 hash (for research use)"""
        data_str = json.dumps(data)
        hash_value = hashlib.sha256(entity_text.encode()).hexdigest()[:16]
        data_str = data_str.replace(entity_text, f"HASH_{hash_value}")
        return json.loads(data_str)
    
    async def _check_patient_consent(
        self,
        patient_id: str,
        purpose: str,
        user_context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Check if patient has given consent for this purpose
        
        Args:
            patient_id: Patient identifier
            purpose: Purpose of access
            user_context: User information
        
        Returns:
            Tuple of (consent_result, action_record)
        """
        # Get patient consent records
        consent = self.consent_records.get(patient_id, {})
        
        # Default consent (if no record, assume consent for treatment)
        if not consent:
            consent = {
                "treatment": True,
                "payment": True,
                "operations": True,
                "research": False,
                "opt_out_all": False,
                "consent_date": datetime.now().isoformat()
            }
        
        # Check if patient has opted out of everything
        if consent.get("opt_out_all", False):
            result = {
                "allowed": False,
                "reason": "Patient has opted out of all data access"
            }
        else:
            # Check purpose-specific consent
            allowed = consent.get(purpose, purpose in ["treatment", "payment", "operations"])
            result = {
                "allowed": allowed,
                "reason": None if allowed else f"Patient has not consented to {purpose}"
            }
        
        action = {
            "action": "consent_check",
            "patient_id": patient_id,
            "purpose": purpose,
            "consent_granted": result["allowed"],
            "timestamp": datetime.now().isoformat()
        }
        
        return result, action
    
    async def _check_retention_policy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if data has exceeded retention period
        """
        classification = self._determine_data_sensitivity(data)
        retention = self.retention_periods.get(classification, timedelta(days=30))
        
        # Check if data has timestamp
        data_timestamp = None
        if "created_at" in data:
            try:
                data_timestamp = datetime.fromisoformat(data["created_at"])
            except (ValueError, TypeError):
                data_timestamp = datetime.now()
        
        action = {
            "action": "retention_check",
            "classification": classification.name,
            "retention_period_days": retention.days,
            "data_timestamp": data_timestamp.isoformat() if data_timestamp else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # In production, would check if data should be deleted
        # Here we just log the check
        
        return action
    
    async def _create_access_record(
        self,
        user_context: Dict[str, Any],
        data: Dict[str, Any],
        purpose: str,
        status: str
    ) -> Dict[str, Any]:
        """
        Create comprehensive access record for audit
        """
        import uuid
        
        record = {
            "request_id": str(uuid.uuid4()),
            "user_id": user_context.get("user_id"),
            "user_role": user_context.get("role"),
            "user_department": user_context.get("department"),
            "resource_type": "patient_data" if "patient_id" in data else "other",
            "resource_id": data.get("patient_id") or data.get("conversation_id"),
            "access_type": "read",
            "purpose": purpose,
            "status": status,
            "data_classification": self._determine_data_sensitivity(data).name,
            "timestamp": datetime.now().isoformat(),
            "ip_address": user_context.get("ip_address"),
            "session_id": user_context.get("session_id")
        }
        
        return record
    
    async def _update_breach_counters(self, user_context: Dict[str, Any], success: bool):
        """
        Update breach detection counters
        """
        user_id = user_context.get("user_id")
        current_minute = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Update access counter
        self.access_counter[f"{user_id}:{current_minute}"] += 1
        
        # Check for anomalies
        if success:
            # Check for excessive data access
            if self.access_counter[f"{user_id}:{current_minute}"] > self.breach_thresholds["max_data_volume"]:
                await self._trigger_breach_alert({
                    "type": "excessive_data_access",
                    "user_id": user_id,
                    "count": self.access_counter[f"{user_id}:{current_minute}"],
                    "time_window": current_minute
                })
    
    async def _trigger_breach_alert(self, alert_data: Dict[str, Any]):
        """
        Trigger breach alert for compliance team
        In production, this would send emails, Slack messages, etc.
        """
        logger.warning(f"POTENTIAL BREACH DETECTED: {alert_data}")
        
        # Log to audit
        await self.audit_logger.log_breach_alert(alert_data)
        
        # In production, would notify compliance team
        # send_email, send_slack, etc.
    
    async def grant_special_access(
        self,
        user_id: str,
        resource_id: str,
        grantor_id: str,
        expiration: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Grant special access to a user for a specific resource
        Used for emergency access or temporary permissions
        """
        with self._lock:
            if resource_id not in self.access_grants[user_id]:
                self.access_grants[user_id].add(resource_id)
                
                grant_record = {
                    "grant_id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "resource_id": resource_id,
                    "grantor_id": grantor_id,
                    "granted_at": datetime.now().isoformat(),
                    "expires_at": expiration.isoformat() if expiration else None
                }
                
                # Log the grant
                await self.audit_logger.log_special_access(grant_record)
                
                return {"success": True, "grant": grant_record}
            
            return {"success": False, "reason": "Access already granted"}
    
    async def revoke_special_access(self, user_id: str, resource_id: str) -> bool:
        """Revoke special access"""
        with self._lock:
            if resource_id in self.access_grants.get(user_id, set()):
                self.access_grants[user_id].remove(resource_id)
                
                revoke_record = {
                    "user_id": user_id,
                    "resource_id": resource_id,
                    "revoked_at": datetime.now().isoformat()
                }
                
                await self.audit_logger.log_special_access_revocation(revoke_record)
                return True
            
            return False
    
    async def update_patient_consent(
        self,
        patient_id: str,
        consent_settings: Dict[str, bool],
        updated_by: str
    ) -> bool:
        """
        Update patient consent preferences
        """
        with self._lock:
            self.consent_records[patient_id] = {
                **consent_settings,
                "updated_at": datetime.now().isoformat(),
                "updated_by": updated_by
            }
            
            await self.audit_logger.log_consent_update({
                "patient_id": patient_id,
                "settings": consent_settings,
                "updated_by": updated_by,
                "timestamp": datetime.now().isoformat()
            })
            
            return True
    
    def get_privacy_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Generate comprehensive privacy report for compliance
        """
        report = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "policies_enforced": len(self.policies),
            "active_consents": len(self.consent_records),
            "special_grants": sum(len(grants) for grants in self.access_grants.values()),
            "breach_alerts": [],  # Would fetch from audit logger
            "access_summary": {
                "total_access": sum(self.access_counter.values()),
                "failed_access": sum(self.failed_access_counter.values())
            }
        }
        
        return report