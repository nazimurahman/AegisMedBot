"""
Compliance Agent Module for AegisMedBot Healthcare Platform
This module provides comprehensive compliance, privacy, and audit functionality
for ensuring HIPAA-style regulatory compliance in healthcare AI interactions.
"""

from agents.compliance_agent.privacy_guardian import PrivacyGuardian
from agents.compliance_agent.phi_detector import PHIDetector
from agents.compliance_agent.audit_logger import AuditLogger

__version__ = "1.0.0"
__all__ = [
    "PrivacyGuardian",
    "PHIDetector", 
    "AuditLogger"
]