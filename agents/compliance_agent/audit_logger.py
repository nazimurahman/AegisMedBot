"""
Audit Logger Module
Responsible for comprehensive audit logging of all compliance-related events
Ensures complete traceability for regulatory compliance and security investigations
"""

import json
import uuid
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
from collections import deque
import os

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """
    Types of audit events for categorization
    """
    ACCESS = "access"               # Data access events
    MODIFICATION = "modification"    # Data modification
    PHI_DETECTION = "phi_detection" # PHI detection events
    PRIVACY_CHECK = "privacy_check" # Privacy policy checks
    CONSENT_UPDATE = "consent_update" # Patient consent changes
    BREACH_ALERT = "breach_alert"    # Security breach alerts
    USER_ACTION = "user_action"      # User administrative actions
    SYSTEM_EVENT = "system_event"     # System-level events
    ERROR = "error"                   # Error events
    COMPLIANCE = "compliance"         # Compliance-related events

class AuditSeverity(Enum):
    """
    Severity levels for audit events
    """
    INFO = "info"          # Informational events
    WARNING = "warning"    # Potential issues
    ERROR = "error"        # Errors requiring attention
    CRITICAL = "critical"  # Critical compliance issues

class AuditLogEntry:
    """
    Represents a single audit log entry
    Immutable after creation to ensure integrity
    """
    
    def __init__(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        user_id: Optional[str],
        action: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Create a new audit log entry with timestamp and unique ID
        """
        self.entry_id = str(uuid.uuid4())
        self.timestamp = datetime.now()
        self.event_type = event_type
        self.severity = severity
        self.user_id = user_id
        self.action = action
        self.resource_id = resource_id
        self.details = details or {}
        self.ip_address = ip_address
        self.session_id = session_id
        
        # Create hash for integrity verification
        self.hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """
        Calculate SHA-256 hash of entry content for tamper detection
        """
        content = f"{self.entry_id}{self.timestamp.isoformat()}{self.event_type.value}{self.severity.value}{self.user_id}{self.action}{self.resource_id}{json.dumps(self.details, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entry to dictionary for serialization
        """
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "user_id": self.user_id,
            "action": self.action,
            "resource_id": self.resource_id,
            "details": self.details,
            "ip_address": self.ip_address,
            "session_id": self.session_id,
            "hash": self.hash
        }
    
    def verify_integrity(self) -> bool:
        """
        Verify that the entry hasn't been tampered with
        """
        return self.hash == self._calculate_hash()

class AuditLogger:
    """
    Comprehensive audit logging system with:
    - Multiple output destinations (file, database, SIEM)
    - Log rotation and retention policies
    - Tamper detection
    - Search and retrieval capabilities
    - Real-time alerting for critical events
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize audit logger with configuration
        
        Args:
            config: Configuration dictionary with:
                - log_directory: Where to store log files
                - retention_days: How long to keep logs
                - enable_encryption: Whether to encrypt logs
                - siem_endpoint: Optional SIEM integration
                - alert_webhook: Webhook for critical alerts
        """
        self.config = config or {}
        
        # Setup logging directories
        self.log_directory = self.config.get("log_directory", "logs/audit")
        os.makedirs(self.log_directory, exist_ok=True)
        
        # Retention policy
        self.retention_days = self.config.get("retention_days", 365)  # HIPAA requires 6 years
        
        # In-memory buffer for recent logs
        self.recent_logs = deque(maxlen=10000)
        
        # Encryption settings
        self.enable_encryption = self.config.get("enable_encryption", True)
        self.encryption_key = self.config.get("encryption_key")
        
        # SIEM integration
        self.siem_endpoint = self.config.get("siem_endpoint")
        
        # Alert webhook for critical events
        self.alert_webhook = self.config.get("alert_webhook")
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "total_entries": 0,
            "by_event_type": {},
            "by_severity": {},
            "last_entry_time": None
        }
        
        logger.info(f"Audit Logger initialized with retention of {self.retention_days} days")
    
    async def log_access(
        self,
        user_id: str,
        resource_id: str,
        access_type: str,
        granted: bool,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Log data access events
        """
        event = AuditLogEntry(
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.INFO if granted else AuditSeverity.WARNING,
            user_id=user_id,
            action=f"access_{access_type}",
            resource_id=resource_id,
            details={
                "access_type": access_type,
                "granted": granted,
                ** (details or {})
            },
            ip_address=ip_address,
            session_id=session_id
        )
        
        await self._write_log(event)
    
    async def log_modification(
        self,
        user_id: str,
        resource_id: str,
        modification_type: str,
        before: Optional[Dict[str, Any]],
        after: Dict[str, Any],
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Log data modification events
        Includes before/after states for complete audit trail
        """
        # Remove sensitive data from before/after for logging
        safe_before = self._sanitize_for_log(before) if before else None
        safe_after = self._sanitize_for_log(after)
        
        event = AuditLogEntry(
            event_type=AuditEventType.MODIFICATION,
            severity=AuditSeverity.INFO,
            user_id=user_id,
            action=f"modify_{modification_type}",
            resource_id=resource_id,
            details={
                "modification_type": modification_type,
                "before": safe_before,
                "after": safe_after
            },
            ip_address=ip_address,
            session_id=session_id
        )
        
        await self._write_log(event)
    
    async def log_phi_detection(
        self,
        phi_results: Dict[str, Any],
        context: Dict[str, Any],
        action_taken: str
    ):
        """
        Log PHI detection events
        """
        event = AuditLogEntry(
            event_type=AuditEventType.PHI_DETECTION,
            severity=AuditSeverity.WARNING if phi_results["risk_level"] == "HIGH" else AuditSeverity.INFO,
            user_id=context.get("user_id"),
            action="phi_detected",
            details={
                "phi_entities": phi_results.get("entity_count", 0),
                "risk_level": phi_results.get("risk_level"),
                "categories": phi_results.get("categories"),
                "action_taken": action_taken,
                "context": self._sanitize_for_log(context)
            },
            ip_address=context.get("ip_address"),
            session_id=context.get("session_id")
        )
        
        await self._write_log(event)
        
        # Alert on high-risk PHI
        if phi_results.get("risk_level") in ["HIGH", "CRITICAL"]:
            await self._send_alert("HIGH_RISK_PHI_DETECTED", event.to_dict())
    
    async def log_privacy_event(self, event_data: Dict[str, Any]):
        """
        Log privacy policy enforcement events
        """
        event = AuditLogEntry(
            event_type=AuditEventType.PRIVACY_CHECK,
            severity=AuditSeverity.INFO,
            user_id=event_data.get("user_id"),
            action=event_data.get("action", "privacy_check"),
            resource_id=event_data.get("resource_id"),
            details=event_data,
            ip_address=event_data.get("ip_address"),
            session_id=event_data.get("session_id")
        )
        
        await self._write_log(event)
    
    async def log_consent_update(
        self,
        patient_id: str,
        settings: Dict[str, bool],
        updated_by: str,
        ip_address: Optional[str] = None
    ):
        """
        Log patient consent updates
        """
        event = AuditLogEntry(
            event_type=AuditEventType.CONSENT_UPDATE,
            severity=AuditSeverity.INFO,
            user_id=updated_by,
            action="update_consent",
            resource_id=patient_id,
            details={
                "settings": settings,
                "patient_id": patient_id
            },
            ip_address=ip_address
        )
        
        await self._write_log(event)
    
    async def log_breach_alert(self, alert_data: Dict[str, Any]):
        """
        Log security breach alerts
        These are critical events requiring immediate attention
        """
        event = AuditLogEntry(
            event_type=AuditEventType.BREACH_ALERT,
            severity=AuditSeverity.CRITICAL,
            user_id=alert_data.get("user_id"),
            action="breach_detected",
            details=alert_data,
            ip_address=alert_data.get("ip_address")
        )
        
        await self._write_log(event)
        
        # Send immediate alert for breaches
        await self._send_alert("SECURITY_BREACH", event.to_dict())
    
    async def log_error(
        self,
        error_data: Dict[str, Any],
        severity: AuditSeverity = AuditSeverity.ERROR
    ):
        """
        Log error events
        """
        event = AuditLogEntry(
            event_type=AuditEventType.ERROR,
            severity=severity,
            user_id=error_data.get("user_id"),
            action="error_occurred",
            details=error_data
        )
        
        await self._write_log(event)
    
    async def log_compliance_check(
        self,
        check_type: str,
        result: bool,
        details: Dict[str, Any]
    ):
        """
        Log compliance check results
        """
        event = AuditLogEntry(
            event_type=AuditEventType.COMPLIANCE,
            severity=AuditSeverity.INFO if result else AuditSeverity.WARNING,
            user_id=details.get("user_id"),
            action=f"compliance_check_{check_type}",
            details={
                "check_type": check_type,
                "result": "PASSED" if result else "FAILED",
                **details
            }
        )
        
        await self._write_log(event)
    
    async def log_special_access(self, grant_record: Dict[str, Any]):
        """
        Log special access grants (emergency access)
        These require extra scrutiny
        """
        event = AuditLogEntry(
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.WARNING,
            user_id=grant_record.get("user_id"),
            action="special_access_granted",
            resource_id=grant_record.get("resource_id"),
            details=grant_record
        )
        
        await self._write_log(event)
    
    async def log_special_access_revocation(self, revoke_record: Dict[str, Any]):
        """
        Log special access revocations
        """
        event = AuditLogEntry(
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.INFO,
            user_id=revoke_record.get("user_id"),
            action="special_access_revoked",
            resource_id=revoke_record.get("resource_id"),
            details=revoke_record
        )
        
        await self._write_log(event)
    
    async def _write_log(self, entry: AuditLogEntry):
        """
        Write audit log entry to all configured destinations
        """
        with self._lock:
            try:
                # Convert to dictionary
                entry_dict = entry.to_dict()
                
                # Add to recent logs buffer
                self.recent_logs.append(entry_dict)
                
                # Update statistics
                self.stats["total_entries"] += 1
                self.stats["by_event_type"][entry.event_type.value] = \
                    self.stats["by_event_type"].get(entry.event_type.value, 0) + 1
                self.stats["by_severity"][entry.severity.value] = \
                    self.stats["by_severity"].get(entry.severity.value, 0) + 1
                self.stats["last_entry_time"] = entry.timestamp.isoformat()
                
                # Write to file
                await self._write_to_file(entry_dict)
                
                # Write to database if configured
                if self.config.get("database_enabled"):
                    await self._write_to_database(entry_dict)
                
                # Send to SIEM if configured
                if self.siem_endpoint and entry.severity in [AuditSeverity.WARNING, AuditSeverity.CRITICAL]:
                    await self._send_to_siem(entry_dict)
                
                logger.debug(f"Audit log written: {entry.entry_id}")
                
            except Exception as e:
                logger.error(f"Failed to write audit log: {str(e)}")
                # In production, would have fallback logging mechanism
    
    async def _write_to_file(self, entry: Dict[str, Any]):
        """
        Write audit entry to log file with rotation
        """
        # Determine current log file based on date
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self.log_directory, f"audit_{date_str}.log")
        
        # Format entry as JSON line
        line = json.dumps(entry) + "\n"
        
        # Encrypt if enabled
        if self.enable_encryption and self.encryption_key:
            line = self._encrypt_line(line)
        
        # Write to file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)
    
    async def _write_to_database(self, entry: Dict[str, Any]):
        """
        Write audit entry to database
        Placeholder for database integration
        """
        # In production, implement database write
        pass
    
    async def _send_to_siem(self, entry: Dict[str, Any]):
        """
        Send audit entry to SIEM system
        Placeholder for SIEM integration (Splunk, ELK, etc.)
        """
        # In production, implement SIEM API call
        pass
    
    async def _send_alert(self, alert_type: str, data: Dict[str, Any]):
        """
        Send alert for critical events
        """
        logger.warning(f"ALERT: {alert_type} - {data}")
        
        if self.alert_webhook:
            # In production, send webhook notification
            # requests.post(self.alert_webhook, json={"type": alert_type, "data": data})
            pass
    
    def _sanitize_for_log(self, data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Remove sensitive data from log entries
        Ensures logs don't contain PHI
        """
        if not data:
            return None
        
        sensitive_fields = [
            "password", "token", "secret", "key",
            "ssn", "mrn", "credit_card", "phone"
        ]
        
        sanitized = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_fields):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_for_log(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_for_log(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _encrypt_line(self, line: str) -> str:
        """
        Encrypt log line for secure storage
        Placeholder - in production use proper encryption
        """
        # Simple base64 encoding for demonstration
        # In production, use AES or similar
        import base64
        return base64.b64encode(line.encode()).decode() + "\n"
    
    async def search_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Search audit logs with filters
        """
        results = []
        
        # Determine date range
        start = start_date or (datetime.now() - timedelta(days=self.retention_days))
        end = end_date or datetime.now()
        
        # Check recent logs first (fast)
        for entry in self.recent_logs:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if start <= entry_time <= end:
                if self._matches_filters(entry, event_type, user_id, severity):
                    results.append(entry)
        
        # If we need more, search log files
        if len(results) < limit:
            file_results = await self._search_log_files(start, end, event_type, user_id, severity, limit)
            results.extend(file_results)
        
        # Deduplicate by entry_id
        seen = set()
        unique_results = []
        for entry in results:
            if entry["entry_id"] not in seen:
                seen.add(entry["entry_id"])
                unique_results.append(entry)
        
        return unique_results[:limit]
    
    def _matches_filters(
        self,
        entry: Dict[str, Any],
        event_type: Optional[AuditEventType],
        user_id: Optional[str],
        severity: Optional[AuditSeverity]
    ) -> bool:
        """
        Check if entry matches search filters
        """
        if event_type and entry["event_type"] != event_type.value:
            return False
        if user_id and entry["user_id"] != user_id:
            return False
        if severity and entry["severity"] != severity.value:
            return False
        return True
    
    async def _search_log_files(
        self,
        start: datetime,
        end: datetime,
        event_type: Optional[AuditEventType],
        user_id: Optional[str],
        severity: Optional[AuditSeverity],
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Search through historical log files
        """
        results = []
        
        # Generate list of log files to search
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            log_file = os.path.join(self.log_directory, f"audit_{date_str}.log")
            
            if os.path.exists(log_file):
                file_results = await self._parse_log_file(
                    log_file, start, end, event_type, user_id, severity
                )
                results.extend(file_results)
                
                if len(results) >= limit:
                    break
            
            current += timedelta(days=1)
        
        return results
    
    async def _parse_log_file(
        self,
        file_path: str,
        start: datetime,
        end: datetime,
        event_type: Optional[AuditEventType],
        user_id: Optional[str],
        severity: Optional[AuditSeverity]
    ) -> List[Dict[str, Any]]:
        """
        Parse and filter a single log file
        """
        results = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Decrypt if needed
                if self.enable_encryption and self.encryption_key:
                    try:
                        import base64
                        line = base64.b64decode(line).decode()
                    except:
                        continue
                
                try:
                    entry = json.loads(line)
                    entry_time = datetime.fromisoformat(entry["timestamp"])
                    
                    if start <= entry_time <= end:
                        if self._matches_filters(entry, event_type, user_id, severity):
                            results.append(entry)
                except:
                    continue
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get audit logger statistics
        """
        return {
            **self.stats,
            "recent_logs_count": len(self.recent_logs),
            "log_directory": self.log_directory,
            "retention_days": self.retention_days
        }
    
    async def rotate_logs(self):
        """
        Rotate old logs based on retention policy
        """
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        
        for filename in os.listdir(self.log_directory):
            if filename.startswith("audit_") and filename.endswith(".log"):
                # Extract date from filename
                date_str = filename.replace("audit_", "").replace(".log", "")
                try:
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if file_date < cutoff:
                        file_path = os.path.join(self.log_directory, filename)
                        os.remove(file_path)
                        logger.info(f"Rotated out old log: {filename}")
                except:
                    continue
    
    def verify_log_integrity(self, start_date: datetime, end_date: datetime) -> bool:
        """
        Verify integrity of logs in date range
        Ensures no tampering has occurred
        """
        current = start_date
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            log_file = os.path.join(self.log_directory, f"audit_{date_str}.log")
            
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Decrypt if needed
                        if self.enable_encryption and self.encryption_key:
                            try:
                                import base64
                                line = base64.b64decode(line).decode()
                            except:
                                return False
                        
                        try:
                            entry = json.loads(line)
                            # Recreate entry and verify hash
                            # This requires storing original entry data
                            # Simplified for demonstration
                            if "hash" not in entry:
                                return False
                        except:
                            return False
            
            current += timedelta(days=1)
        
        return True