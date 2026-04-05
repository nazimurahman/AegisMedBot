"""
Admission Repository for AegisMedBot

This module provides data access layer for admission operations.
Implements repository pattern for clean separation of database logic.
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
import logging

from ..models.admission import Admission, AdmissionStatus, AdmissionType

# Configure logging for this module
logger = logging.getLogger(__name__)

class AdmissionRepository:
    """
    Repository for Admission database operations.
    
    This class encapsulates all database operations related to patient admissions.
    It provides a clean interface for the service layer to interact with
    admission data without exposing SQLAlchemy details.
    
    Key Responsibilities:
    - Create, read, update, delete admission records
    - Query admissions with various filters
    - Calculate admission metrics and statistics
    - Support analytics and reporting
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize repository with database session.
        
        Args:
            db_session: SQLAlchemy database session for operations
        """
        self.db = db_session
    
    def create(self, admission_data: Dict[str, Any]) -> Admission:
        """
        Create a new admission record.
        
        Args:
            admission_data: Dictionary containing admission fields
            
        Returns:
            Created Admission object
            
        Raises:
            ValueError: If required fields are missing
        """
        try:
            # Create admission instance from dictionary
            # The ** operator unpacks the dictionary into keyword arguments
            admission = Admission(**admission_data)
            
            # Add to database session
            self.db.add(admission)
            
            # Commit to save changes permanently
            self.db.commit()
            
            # Refresh to get any database-generated values (like auto timestamps)
            self.db.refresh(admission)
            
            logger.info(f"Created admission {admission.id} for patient {admission.patient_id}")
            return admission
            
        except Exception as e:
            # If anything fails, rollback to maintain data consistency
            self.db.rollback()
            logger.error(f"Failed to create admission: {str(e)}")
            raise ValueError(f"Could not create admission: {str(e)}")
    
    def get_by_id(self, admission_id: str) -> Optional[Admission]:
        """
        Retrieve admission by its unique identifier.
        
        Args:
            admission_id: String UUID of the admission
            
        Returns:
            Admission object if found, None otherwise
        """
        # Query the database for admission with matching ID
        # filter_by is a simpler syntax for equality filters
        return self.db.query(Admission).filter_by(
            id=admission_id,
            is_deleted=False  # Exclude soft-deleted records
        ).first()
    
    def get_by_patient(self, patient_id: str, limit: int = 10) -> List[Admission]:
        """
        Get all admissions for a specific patient.
        
        Args:
            patient_id: Patient identifier
            limit: Maximum number of admissions to return
            
        Returns:
            List of admission objects, most recent first
        """
        # Query admissions for patient, ordered by date descending
        return self.db.query(Admission).filter(
            Admission.patient_id == patient_id,
            Admission.is_deleted == False
        ).order_by(
            desc(Admission.admission_date)  # Most recent first
        ).limit(limit).all()
    
    def get_current_admission(self, patient_id: str) -> Optional[Admission]:
        """
        Get currently active admission for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Active admission if exists, None otherwise
        """
        return self.db.query(Admission).filter(
            Admission.patient_id == patient_id,
            Admission.status == AdmissionStatus.ACTIVE,
            Admission.discharge_date.is_(None),  # Not discharged yet
            Admission.is_deleted == False
        ).first()
    
    def get_active_admissions(
        self, 
        department: Optional[str] = None,
        limit: int = 100
    ) -> List[Admission]:
        """
        Get all currently active admissions in the hospital.
        
        Args:
            department: Optional department filter
            limit: Maximum number of records to return
            
        Returns:
            List of active admissions
        """
        # Build query with filters
        query = self.db.query(Admission).filter(
            Admission.status == AdmissionStatus.ACTIVE,
            Admission.is_deleted == False
        )
        
        # Apply department filter if provided
        if department:
            query = query.filter(Admission.department == department)
        
        # Order by admission date (oldest first for rounding)
        return query.order_by(Admission.admission_date).limit(limit).all()
    
    def get_admissions_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        department: Optional[str] = None
    ) -> List[Admission]:
        """
        Get admissions within a date range.
        
        Args:
            start_date: Beginning of date range
            end_date: End of date range
            department: Optional department filter
            
        Returns:
            List of admissions in date range
        """
        query = self.db.query(Admission).filter(
            Admission.admission_date >= start_date,
            Admission.admission_date <= end_date,
            Admission.is_deleted == False
        )
        
        if department:
            query = query.filter(Admission.department == department)
        
        return query.order_by(Admission.admission_date).all()
    
    def update(self, admission_id: str, update_data: Dict[str, Any]) -> Optional[Admission]:
        """
        Update an existing admission record.
        
        Args:
            admission_id: Admission identifier
            update_data: Dictionary of fields to update
            
        Returns:
            Updated admission object if found, None otherwise
        """
        # Find the admission to update
        admission = self.get_by_id(admission_id)
        
        if not admission:
            logger.warning(f"Admission {admission_id} not found for update")
            return None
        
        try:
            # Update each field provided in update_data
            for key, value in update_data.items():
                if hasattr(admission, key):
                    setattr(admission, key, value)
            
            # Commit changes to database
            self.db.commit()
            self.db.refresh(admission)
            
            logger.info(f"Updated admission {admission_id}")
            return admission
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to update admission {admission_id}: {str(e)}")
            raise ValueError(f"Could not update admission: {str(e)}")
    
    def discharge_patient(
        self, 
        admission_id: str, 
        discharge_date: Optional[datetime] = None,
        discharge_disposition: Optional[str] = None
    ) -> Optional[Admission]:
        """
        Discharge a patient, ending their admission.
        
        Args:
            admission_id: Admission identifier
            discharge_date: When patient was discharged (defaults to now)
            discharge_disposition: Where patient is going after discharge
            
        Returns:
            Updated admission object
        """
        admission = self.get_by_id(admission_id)
        
        if not admission:
            return None
        
        # Set discharge information
        admission.discharge_date = discharge_date or datetime.now()
        admission.status = AdmissionStatus.DISCHARGED
        
        if discharge_disposition:
            admission.discharge_disposition = discharge_disposition
        
        # Commit changes
        self.db.commit()
        self.db.refresh(admission)
        
        logger.info(f"Discharged patient from admission {admission_id}")
        return admission
    
    def delete(self, admission_id: str, soft_delete: bool = True) -> bool:
        """
        Delete an admission record.
        
        Args:
            admission_id: Admission identifier
            soft_delete: If True, mark as deleted instead of removing
            
        Returns:
            True if successful, False otherwise
        """
        admission = self.get_by_id(admission_id)
        
        if not admission:
            return False
        
        try:
            if soft_delete:
                # Soft delete - just mark as deleted
                admission.is_deleted = True
            else:
                # Hard delete - remove from database
                self.db.delete(admission)
            
            self.db.commit()
            logger.info(f"{'Soft' if soft_delete else 'Hard'} deleted admission {admission_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to delete admission {admission_id}: {str(e)}")
            return False
    
    def get_department_metrics(
        self,
        department: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Calculate department metrics for analytics.
        
        Args:
            department: Department name
            start_date: Start of analysis period
            end_date: End of analysis period
            
        Returns:
            Dictionary with metrics including:
            - total_admissions
            - average_length_of_stay
            - readmission_rate
            - bed_turnover_rate
        """
        # Get admissions in department for date range
        admissions = self.get_admissions_by_date_range(
            start_date, end_date, department
        )
        
        if not admissions:
            return {
                "department": department,
                "total_admissions": 0,
                "average_length_of_stay": 0,
                "readmission_rate": 0,
                "bed_turnover_rate": 0
            }
        
        # Calculate total admissions
        total = len(admissions)
        
        # Calculate average length of stay
        los_list = []
        for admission in admissions:
            los = admission.calculate_length_of_stay()
            if los:
                los_list.append(los)
        
        avg_los = sum(los_list) / len(los_list) if los_list else 0
        
        # Calculate readmission rate
        readmissions = 0
        for admission in admissions:
            # Get patient's previous admissions
            previous = self.get_by_patient(admission.patient_id)
            if admission.is_readmission(previous):
                readmissions += 1
        
        readmission_rate = (readmissions / total) * 100 if total > 0 else 0
        
        return {
            "department": department,
            "total_admissions": total,
            "average_length_of_stay": round(avg_los, 2),
            "readmission_rate": round(readmission_rate, 2),
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat()
        }
    
    def get_bed_occupancy_report(self) -> Dict[str, Any]:
        """
        Generate current bed occupancy report.
        
        Returns:
            Dictionary with occupancy statistics by department
        """
        # Get all active admissions
        active = self.get_active_admissions()
        
        # Group by department
        department_stats = {}
        for admission in active:
            dept = admission.department
            if dept not in department_stats:
                department_stats[dept] = {
                    "occupied_beds": 0,
                    "patients": []
                }
            
            department_stats[dept]["occupied_beds"] += 1
            department_stats[dept]["patients"].append({
                "patient_id": admission.patient_id,
                "admission_id": admission.id,
                "room": admission.room_number,
                "admission_date": admission.admission_date.isoformat()
            })
        
        return {
            "total_active_admissions": len(active),
            "report_timestamp": datetime.now().isoformat(),
            "by_department": department_stats
        }
    
    def get_pending_discharges(self, hours_ahead: int = 24) -> List[Admission]:
        """
        Get admissions expected to be discharged soon.
        
        Args:
            hours_ahead: Look ahead window in hours
            
        Returns:
            List of admissions likely to be discharged
        """
        # This is a simplified implementation
        # In production, would use ML prediction or discharge planning data
        
        # Get all active admissions
        active = self.get_active_admissions()
        
        # Filter those with extended length of stay nearing discharge
        # This is a placeholder - real implementation would be more sophisticated
        pending = []
        for admission in active:
            los = admission.calculate_length_of_stay()
            
            # If length of stay exceeds typical for department
            # Mark as potential discharge (simplified logic)
            if los and los > 3:  # More than 3 days
                pending.append(admission)
        
        return pending
    
    def count_active_by_department(self) -> Dict[str, int]:
        """
        Count active admissions grouped by department.
        
        Returns:
            Dictionary mapping department to count
        """
        # Use SQL GROUP BY for efficient counting
        results = self.db.query(
            Admission.department,
            func.count(Admission.id).label('count')
        ).filter(
            Admission.status == AdmissionStatus.ACTIVE,
            Admission.is_deleted == False
        ).group_by(Admission.department).all()
        
        # Convert to dictionary
        return {r.department: r.count for r in results}
    
    def get_admission_timeline(
        self,
        patient_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get chronological timeline of patient's admissions.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            List of admission summaries in chronological order
        """
        admissions = self.get_by_patient(patient_id, limit=50)
        
        timeline = []
        for admission in admissions:
            timeline.append({
                "admission_id": admission.id,
                "date": admission.admission_date.isoformat(),
                "department": admission.department,
                "length_of_stay": admission.calculate_length_of_stay(),
                "diagnosis": admission.diagnosis.get("primary") if admission.diagnosis else None,
                "outcome": admission.discharge_disposition.value if admission.discharge_disposition else None
            })
        
        return timeline