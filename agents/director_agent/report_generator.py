"""
Report Generator Module for Director Intelligence Agent.

This module generates comprehensive executive reports for hospital leadership.
It supports multiple report types, formats, and delivery methods.

Features:
    - Daily executive summaries
    - Weekly performance reports
    - Monthly strategic reviews
    - Quarterly board presentations
    - Ad-hoc custom reports
    - Multi-format output (PDF, HTML, JSON, Markdown)
    - Email delivery
    - Dashboard integration
"""

import json
import markdown
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
import logging
from pathlib import Path
import jinja2
import aiofiles
import asyncio
from dataclasses import dataclass, field

# For PDF generation (optional)
try:
    from weasyprint import HTML
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("WeasyPrint not installed. PDF generation disabled.")

logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Types of reports that can be generated."""
    DAILY_EXECUTIVE = "daily_executive"
    WEEKLY_PERFORMANCE = "weekly_performance"
    MONTHLY_STRATEGIC = "monthly_strategic"
    QUARTERLY_BOARD = "quarterly_board"
    ANNUAL_REVIEW = "annual_review"
    ADHOC = "adhoc"
    ALERT_SUMMARY = "alert_summary"
    TREND_ANALYSIS = "trend_analysis"
    DEPARTMENT_REVIEW = "department_review"

class ReportFormat(Enum):
    """Output formats for reports."""
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    EMAIL = "email"

class DeliveryMethod(Enum):
    """Methods for delivering reports."""
    DASHBOARD = "dashboard"
    EMAIL = "email"
    FILE_SYSTEM = "file_system"
    API = "api"
    PRINT = "print"

@dataclass
class ReportMetadata:
    """Metadata for a generated report."""
    report_id: str
    report_type: ReportType
    title: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    generated_by: str
    format: ReportFormat
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    file_path: Optional[str] = None
    recipients: List[str] = field(default_factory=list)

class ReportGenerator:
    """
    Comprehensive report generator for hospital leadership.
    
    This class handles the creation, formatting, and delivery of various
    report types for the Medical Director and other hospital executives.
    """
    
    def __init__(
        self,
        agent_id: str,
        templates_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        email_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Report Generator.
        
        Args:
            agent_id: ID of the parent agent
            templates_dir: Directory containing report templates
            output_dir: Directory for generated reports
            email_config: Configuration for email delivery
        """
        self.agent_id = agent_id
        self.templates_dir = templates_dir or "./templates/reports"
        self.output_dir = output_dir or "./reports"
        self.email_config = email_config or {}
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 template environment
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir),
            autoescape=True
        )
        
        # Report cache
        self.report_cache: Dict[str, ReportMetadata] = {}
        
        logger.info(f"Report Generator initialized for agent {agent_id}")
    
    async def generate_daily_executive(
        self,
        data: Dict[str, Any],
        generated_by: str = "Director Agent"
    ) -> ReportMetadata:
        """
        Generate daily executive summary report.
        
        Args:
            data: Report data including KPIs, alerts, etc.
            generated_by: Name of the generator
            
        Returns:
            ReportMetadata for the generated report
        """
        report_type = ReportType.DAILY_EXECUTIVE
        today = datetime.now()
        
        report_data = {
            "title": f"Daily Executive Summary - {today.strftime('%Y-%m-%d')}",
            "generated_at": today.isoformat(),
            "period": "Last 24 hours",
            "summary": data.get("summary", {}),
            "critical_issues": data.get("critical_issues", []),
            "key_metrics": data.get("key_metrics", {}),
            "alerts": data.get("alerts", [])[:5],  # Top 5 alerts
            "recommendations": data.get("recommendations", [])[:3],
            "upcoming": data.get("upcoming_events", []),
            "weather": data.get("weather_alert"),
            "staffing": data.get("staffing_shortages", [])
        }
        
        metadata = await self._generate_report(
            report_type=report_type,
            title=report_data["title"],
            data=report_data,
            generated_by=generated_by,
            period_start=today - timedelta(days=1),
            period_end=today
        )
        
        return metadata
    
    async def generate_weekly_performance(
        self,
        data: Dict[str, Any],
        generated_by: str = "Director Agent"
    ) -> ReportMetadata:
        """
        Generate weekly performance report.
        
        Args:
            data: Report data
            generated_by: Name of the generator
            
        Returns:
            ReportMetadata for the generated report
        """
        report_type = ReportType.WEEKLY_PERFORMANCE
        today = datetime.now()
        week_start = today - timedelta(days=7)
        
        report_data = {
            "title": f"Weekly Performance Report - Week of {week_start.strftime('%Y-%m-%d')}",
            "generated_at": today.isoformat(),
            "period": "Last 7 days",
            "executive_summary": data.get("executive_summary", ""),
            "kpi_summary": data.get("kpi_summary", {}),
            "trends": data.get("trends", {}),
            "department_performance": data.get("department_performance", []),
            "financial_highlights": data.get("financial", {}),
            "quality_metrics": data.get("quality", {}),
            "patient_experience": data.get("patient_experience", {}),
            "staffing_metrics": data.get("staffing", {}),
            "incident_report": data.get("incidents", []),
            "achievements": data.get("achievements", []),
            "challenges": data.get("challenges", []),
            "recommendations": data.get("recommendations", [])
        }
        
        metadata = await self._generate_report(
            report_type=report_type,
            title=report_data["title"],
            data=report_data,
            generated_by=generated_by,
            period_start=week_start,
            period_end=today
        )
        
        return metadata
    
    async def generate_monthly_strategic(
        self,
        data: Dict[str, Any],
        generated_by: str = "Director Agent"
    ) -> ReportMetadata:
        """
        Generate monthly strategic review report.
        
        Args:
            data: Report data
            generated_by: Name of the generator
            
        Returns:
            ReportMetadata for the generated report
        """
        report_type = ReportType.MONTHLY_STRATEGIC
        today = datetime.now()
        month_start = today - timedelta(days=30)
        
        report_data = {
            "title": f"Monthly Strategic Review - {today.strftime('%B %Y')}",
            "generated_at": today.isoformat(),
            "period": "Last 30 days",
            "strategic_overview": data.get("strategic_overview", ""),
            "kpi_dashboard": data.get("kpi_dashboard", {}),
            "strategic_initiatives": data.get("initiatives", []),
            "financial_performance": data.get("financial_performance", {}),
            "clinical_outcomes": data.get("clinical_outcomes", {}),
            "operational_efficiency": data.get("operational_efficiency", {}),
            "patient_safety": data.get("patient_safety", {}),
            "staff_engagement": data.get("staff_engagement", {}),
            "market_analysis": data.get("market_analysis", {}),
            "competitor_benchmarking": data.get("benchmarking", {}),
            "strategic_recommendations": data.get("recommendations", []),
            "resource_allocation": data.get("resource_allocation", {}),
            "risk_assessment": data.get("risks", []),
            "next_month_priorities": data.get("priorities", [])
        }
        
        metadata = await self._generate_report(
            report_type=report_type,
            title=report_data["title"],
            data=report_data,
            generated_by=generated_by,
            period_start=month_start,
            period_end=today
        )
        
        return metadata
    
    async def generate_quarterly_board(
        self,
        data: Dict[str, Any],
        generated_by: str = "Director Agent"
    ) -> ReportMetadata:
        """
        Generate quarterly board presentation.
        
        Args:
            data: Report data
            generated_by: Name of the generator
            
        Returns:
            ReportMetadata for the generated report
        """
        report_type = ReportType.QUARTERLY_BOARD
        today = datetime.now()
        quarter_start = today - timedelta(days=90)
        
        # Determine quarter
        quarter = (today.month - 1) // 3 + 1
        year = today.year
        
        report_data = {
            "title": f"Q{quarter} {year} Board Presentation",
            "generated_at": today.isoformat(),
            "period": "Last quarter",
            "executive_summary": data.get("executive_summary", ""),
            "financial_results": data.get("financial_results", {}),
            "quality_scorecard": data.get("quality_scorecard", {}),
            "patient_safety": data.get("patient_safety", {}),
            "operational_metrics": data.get("operational_metrics", {}),
            "strategic_achievements": data.get("achievements", []),
            "key_initiatives": data.get("initiatives", []),
            "milestones": data.get("milestones", []),
            "challenges": data.get("challenges", []),
            "competitive_analysis": data.get("competitive_analysis", {}),
            "market_position": data.get("market_position", {}),
            "future_outlook": data.get("outlook", ""),
            "strategic_plan": data.get("strategic_plan", {}),
            "investment_requests": data.get("investments", []),
            "risk_register": data.get("risks", []),
            "appendices": data.get("appendices", {})
        }
        
        metadata = await self._generate_report(
            report_type=report_type,
            title=report_data["title"],
            data=report_data,
            generated_by=generated_by,
            period_start=quarter_start,
            period_end=today
        )
        
        return metadata
    
    async def generate_alert_summary(
        self,
        alerts: List[Dict[str, Any]],
        generated_by: str = "Director Agent"
    ) -> ReportMetadata:
        """
        Generate summary of current alerts.
        
        Args:
            alerts: List of active alerts
            generated_by: Name of the generator
            
        Returns:
            ReportMetadata for the generated report
        """
        report_type = ReportType.ALERT_SUMMARY
        today = datetime.now()
        
        # Categorize alerts
        critical_alerts = [a for a in alerts if a.get("severity") == "critical"]
        warning_alerts = [a for a in alerts if a.get("severity") == "warning"]
        
        report_data = {
            "title": f"Alert Summary - {today.strftime('%Y-%m-%d %H:%M')}",
            "generated_at": today.isoformat(),
            "total_alerts": len(alerts),
            "critical_count": len(critical_alerts),
            "warning_count": len(warning_alerts),
            "critical_alerts": critical_alerts,
            "warning_alerts": warning_alerts,
            "by_department": self._group_alerts_by_department(alerts),
            "by_kpi": self._group_alerts_by_kpi(alerts),
            "trend": self._analyze_alert_trend(alerts),
            "recommended_actions": self._generate_alert_recommendations(alerts)
        }
        
        metadata = await self._generate_report(
            report_type=report_type,
            title=report_data["title"],
            data=report_data,
            generated_by=generated_by,
            period_start=today - timedelta(days=1),
            period_end=today
        )
        
        return metadata
    
    async def generate_department_review(
        self,
        department_id: str,
        department_data: Dict[str, Any],
        generated_by: str = "Director Agent"
    ) -> ReportMetadata:
        """
        Generate detailed review for a specific department.
        
        Args:
            department_id: ID of the department
            department_data: Department performance data
            generated_by: Name of the generator
            
        Returns:
            ReportMetadata for the generated report
        """
        report_type = ReportType.DEPARTMENT_REVIEW
        today = datetime.now()
        
        report_data = {
            "title": f"Department Review: {department_data.get('name', department_id)}",
            "department_id": department_id,
            "department_name": department_data.get("name", department_id),
            "generated_at": today.isoformat(),
            "period": "Last 30 days",
            "performance_summary": department_data.get("summary", {}),
            "key_metrics": department_data.get("metrics", {}),
            "trends": department_data.get("trends", {}),
            "staffing": department_data.get("staffing", {}),
            "budget_vs_actual": department_data.get("budget", {}),
            "quality_indicators": department_data.get("quality", {}),
            "patient_feedback": department_data.get("patient_feedback", []),
            "incident_reports": department_data.get("incidents", []),
            "achievements": department_data.get("achievements", []),
            "challenges": department_data.get("challenges", []),
            "recommendations": department_data.get("recommendations", []),
            "resource_needs": department_data.get("resource_needs", [])
        }
        
        metadata = await self._generate_report(
            report_type=report_type,
            title=report_data["title"],
            data=report_data,
            generated_by=generated_by,
            period_start=today - timedelta(days=30),
            period_end=today
        )
        
        return metadata
    
    async def _generate_report(
        self,
        report_type: ReportType,
        title: str,
        data: Dict[str, Any],
        generated_by: str,
        period_start: datetime,
        period_end: datetime,
        formats: List[ReportFormat] = None
    ) -> ReportMetadata:
        """
        Internal method to generate reports in multiple formats.
        
        Args:
            report_type: Type of report
            title: Report title
            data: Report data
            generated_by: Generator name
            period_start: Start of reporting period
            period_end: End of reporting period
            formats: List of output formats
            
        Returns:
            ReportMetadata for the generated report
        """
        if formats is None:
            formats = [ReportFormat.JSON, ReportFormat.MARKDOWN]
        
        report_id = self._generate_report_id(report_type)
        
        metadata = ReportMetadata(
            report_id=report_id,
            report_type=report_type,
            title=title,
            generated_at=datetime.now(),
            period_start=period_start,
            period_end=period_end,
            generated_by=generated_by,
            format=formats[0],  # Primary format
            tags=[report_type.value, "automated"]
        )
        
        # Generate in each requested format
        for fmt in formats:
            file_path = await self._write_report_file(report_id, fmt, data)
            if fmt == metadata.format:
                metadata.file_path = file_path
        
        # Cache metadata
        self.report_cache[report_id] = metadata
        
        logger.info(f"Generated report: {report_id} - {title}")
        
        return metadata
    
    def _generate_report_id(self, report_type: ReportType) -> str:
        """
        Generate a unique report ID.
        
        Args:
            report_type: Type of report
            
        Returns:
            Unique report identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{report_type.value}_{timestamp}_{self.agent_id[:8]}"
    
    async def _write_report_file(
        self,
        report_id: str,
        format: ReportFormat,
        data: Dict[str, Any]
    ) -> str:
        """
        Write report to file in specified format.
        
        Args:
            report_id: Report identifier
            format: Output format
            data: Report data
            
        Returns:
            Path to the generated file
        """
        filename = f"{report_id}.{format.value}"
        filepath = Path(self.output_dir) / filename
        
        if format == ReportFormat.JSON:
            await self._write_json(filepath, data)
        elif format == ReportFormat.MARKDOWN:
            await self._write_markdown(filepath, data)
        elif format == ReportFormat.HTML:
            await self._write_html(filepath, data)
        elif format == ReportFormat.PDF:
            if PDF_SUPPORT:
                await self._write_pdf(filepath, data)
            else:
                logger.warning(f"PDF generation not supported for {report_id}")
                # Fallback to HTML
                filepath = Path(self.output_dir) / f"{report_id}.html"
                await self._write_html(filepath, data)
        
        logger.info(f"Report written to {filepath}")
        return str(filepath)
    
    async def _write_json(self, filepath: Path, data: Dict[str, Any]) -> None:
        """Write report as JSON."""
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(data, indent=2, default=str))
    
    async def _write_markdown(self, filepath: Path, data: Dict[str, Any]) -> None:
        """Write report as Markdown."""
        md_content = self._convert_to_markdown(data)
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(md_content)
    
    async def _write_html(self, filepath: Path, data: Dict[str, Any]) -> None:
        """Write report as HTML."""
        # First convert to markdown, then to HTML
        md_content = self._convert_to_markdown(data)
        html_content = markdown.markdown(
            md_content,
            extensions=['tables', 'fenced_code', 'codehilite']
        )
        
        # Wrap in basic HTML template
        full_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{data.get('title', 'Report')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .critical {{ color: #e74c3c; }}
        .warning {{ color: #f39c12; }}
        .good {{ color: #27ae60; }}
        .metric {{ margin: 10px 0; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>"""
        
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(full_html)
    
    async def _write_pdf(self, filepath: Path, data: Dict[str, Any]) -> None:
        """Write report as PDF."""
        if not PDF_SUPPORT:
            raise RuntimeError("PDF generation not supported")
        
        # First generate HTML
        html_filepath = filepath.with_suffix('.html')
        await self._write_html(html_filepath, data)
        
        # Convert HTML to PDF
        HTML(str(html_filepath)).write_pdf(str(filepath))
        
        # Clean up HTML file
        html_filepath.unlink()
    
    def _convert_to_markdown(self, data: Dict[str, Any], level: int = 1) -> str:
        """
        Convert report data to Markdown format.
        
        Args:
            data: Report data dictionary
            level: Heading level
            
        Returns:
            Markdown formatted string
        """
        lines = []
        
        # Title
        if level == 1 and 'title' in data:
            lines.append(f"# {data['title']}\n")
        
        # Timestamp and period
        if 'generated_at' in data:
            lines.append(f"*Generated: {data['generated_at']}*\n")
        
        if 'period' in data:
            lines.append(f"*Period: {data['period']}*\n")
        
        # Process each section
        for key, value in data.items():
            if key in ['title', 'generated_at', 'period']:
                continue
            
            # Format section header
            header = key.replace('_', ' ').title()
            lines.append(f"\n{'#' * (level + 1)} {header}\n")
            
            # Format based on value type
            if isinstance(value, dict):
                lines.append(self._convert_to_markdown(value, level + 1))
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    # Table data
                    lines.append(self._dict_list_to_table(value))
                else:
                    # Simple list
                    for item in value:
                        if isinstance(item, dict):
                            lines.append(self._dict_to_bullets(item))
                        else:
                            lines.append(f"- {item}")
            else:
                # Simple value
                lines.append(f"{value}\n")
        
        return '\n'.join(lines)
    
    def _dict_list_to_table(self, data: List[Dict[str, Any]]) -> str:
        """
        Convert list of dictionaries to markdown table.
        
        Args:
            data: List of dictionaries
            
        Returns:
            Markdown table string
        """
        if not data:
            return ""
        
        # Get all unique keys
        headers = set()
        for item in data:
            headers.update(item.keys())
        headers = sorted(list(headers))
        
        # Create table
        lines = []
        lines.append('| ' + ' | '.join(h.replace('_', ' ').title() for h in headers) + ' |')
        lines.append('|' + '|'.join([' --- '] * len(headers)) + '|')
        
        for item in data:
            row = []
            for h in headers:
                value = item.get(h, '')
                if isinstance(value, float):
                    value = f"{value:.2f}"
                row.append(str(value))
            lines.append('| ' + ' | '.join(row) + ' |')
        
        return '\n'.join(lines)
    
    def _dict_to_bullets(self, data: Dict[str, Any]) -> str:
        """
        Convert dictionary to bullet points.
        
        Args:
            data: Dictionary to format
            
        Returns:
            Bullet point string
        """
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"  - {key}:")
                for k, v in value.items():
                    lines.append(f"    - {k}: {v}")
            else:
                if isinstance(value, float):
                    value = f"{value:.2f}"
                lines.append(f"  - {key}: {value}")
        return '\n'.join(lines)
    
    def _group_alerts_by_department(
        self,
        alerts: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group alerts by department.
        
        Args:
            alerts: List of alerts
            
        Returns:
            Dictionary of department to alerts
        """
        grouped = {}
        for alert in alerts:
            dept = alert.get('department', 'Unknown')
            if dept not in grouped:
                grouped[dept] = []
            grouped[dept].append(alert)
        return grouped
    
    def _group_alerts_by_kpi(
        self,
        alerts: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group alerts by KPI.
        
        Args:
            alerts: List of alerts
            
        Returns:
            Dictionary of KPI to alerts
        """
        grouped = {}
        for alert in alerts:
            kpi = alert.get('kpi', 'Unknown')
            if kpi not in grouped:
                grouped[kpi] = []
            grouped[kpi].append(alert)
        return grouped
    
    def _analyze_alert_trend(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze trends in alerts.
        
        Args:
            alerts: List of alerts
            
        Returns:
            Trend analysis dictionary
        """
        # Group by day
        by_day = {}
        for alert in alerts:
            timestamp = alert.get('timestamp', '')
            if timestamp:
                day = timestamp[:10]  # YYYY-MM-DD
                if day not in by_day:
                    by_day[day] = []
                by_day[day].append(alert)
        
        # Calculate trend
        days = sorted(by_day.keys())
        counts = [len(by_day[d]) for d in days]
        
        if len(counts) >= 2:
            trend = "increasing" if counts[-1] > counts[0] else "decreasing" if counts[-1] < counts[0] else "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "by_day": {d: len(by_day[d]) for d in days},
            "total_alerts": len(alerts)
        }
    
    def _generate_alert_recommendations(
        self,
        alerts: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate recommendations based on alerts.
        
        Args:
            alerts: List of alerts
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Group by severity
        critical = [a for a in alerts if a.get('severity') == 'critical']
        warning = [a for a in alerts if a.get('severity') == 'warning']
        
        if critical:
            recommendations.append(f"Immediate action required: {len(critical)} critical alerts need attention")
        
        if warning:
            recommendations.append(f"Review {len(warning)} warning alerts within 24 hours")
        
        # Group by KPI
        by_kpi = self._group_alerts_by_kpi(alerts)
        for kpi, kpi_alerts in by_kpi.items():
            if len(kpi_alerts) >= 3:
                recommendations.append(f"Persistent issues with {kpi} - consider root cause analysis")
        
        return recommendations
    
    async def deliver_report(
        self,
        report_id: str,
        method: DeliveryMethod,
        recipients: Optional[List[str]] = None
    ) -> bool:
        """
        Deliver a generated report through specified method.
        
        Args:
            report_id: ID of the report to deliver
            method: Delivery method
            recipients: List of recipients (for email, etc.)
            
        Returns:
            True if delivery successful, False otherwise
        """
        if report_id not in self.report_cache:
            logger.error(f"Report {report_id} not found in cache")
            return False
        
        metadata = self.report_cache[report_id]
        
        if method == DeliveryMethod.DASHBOARD:
            # Report is already available through dashboard
            logger.info(f"Report {report_id} available on dashboard")
            return True
        
        elif method == DeliveryMethod.EMAIL:
            return await self._deliver_by_email(metadata, recipients)
        
        elif method == DeliveryMethod.FILE_SYSTEM:
            # Already saved to file system
            logger.info(f"Report {report_id} available at {metadata.file_path}")
            return True
        
        elif method == DeliveryMethod.API:
            # Would implement API delivery
            logger.info(f"API delivery for {report_id} not implemented")
            return False
        
        return False
    
    async def _deliver_by_email(
        self,
        metadata: ReportMetadata,
        recipients: Optional[List[str]]
    ) -> bool:
        """
        Deliver report via email.
        
        Args:
            metadata: Report metadata
            recipients: Email recipients
            
        Returns:
            True if email sent successfully
        """
        if not recipients:
            logger.warning(f"No recipients specified for report {metadata.report_id}")
            return False
        
        # In production, this would use SMTP or email service
        # This is a placeholder implementation
        logger.info(f"Would email report {metadata.report_id} to {recipients}")
        
        # Simulate email preparation
        email_data = {
            "to": recipients,
            "subject": metadata.title,
            "body": f"Please find the report attached: {metadata.title}",
            "attachments": [metadata.file_path] if metadata.file_path else []
        }
        
        # Log the email (in production, actually send)
        logger.info(f"Email prepared: {json.dumps(email_data, indent=2)}")
        
        return True
    
    def get_report_metadata(self, report_id: str) -> Optional[ReportMetadata]:
        """
        Get metadata for a generated report.
        
        Args:
            report_id: Report identifier
            
        Returns:
            ReportMetadata if found, None otherwise
        """
        return self.report_cache.get(report_id)
    
    def list_reports(
        self,
        report_type: Optional[ReportType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10
    ) -> List[ReportMetadata]:
        """
        List generated reports with optional filtering.
        
        Args:
            report_type: Filter by report type
            start_date: Filter by generation date start
            end_date: Filter by generation date end
            limit: Maximum number of reports to return
            
        Returns:
            List of ReportMetadata objects
        """
        reports = list(self.report_cache.values())
        
        # Apply filters
        if report_type:
            reports = [r for r in reports if r.report_type == report_type]
        
        if start_date:
            reports = [r for r in reports if r.generated_at >= start_date]
        
        if end_date:
            reports = [r for r in reports if r.generated_at <= end_date]
        
        # Sort by generation date (newest first)
        reports.sort(key=lambda x: x.generated_at, reverse=True)
        
        return reports[:limit]

# Factory function

def create_report_generator(
    agent_id: str,
    templates_dir: Optional[str] = None,
    output_dir: Optional[str] = None
) -> ReportGenerator:
    """
    Create a configured ReportGenerator instance.
    
    Args:
        agent_id: ID of the parent agent
        templates_dir: Templates directory
        output_dir: Output directory
        
    Returns:
        Configured ReportGenerator
    """
    return ReportGenerator(
        agent_id=agent_id,
        templates_dir=templates_dir,
        output_dir=output_dir
    )

if __name__ == "__main__":
    # Example usage
    async def main():
        generator = create_report_generator("director_001")
        
        # Simulate data
        data = {
            "title": "Test Report",
            "summary": {"total_kpis": 10, "average_score": 85},
            "key_metrics": {
                "mortality_rate": 1.2,
                "readmission_rate": 14.5
            },
            "alerts": [
                {"severity": "warning", "kpi": "mortality_rate", "value": 1.2}
            ],
            "recommendations": ["Review ICU protocols"]
        }
        
        # Generate report
        metadata = await generator.generate_daily_executive(data)
        print(f"Generated report: {metadata.report_id}")
        
        # List reports
        reports = generator.list_reports(limit=5)
        print(f"Recent reports: {len(reports)}")

    asyncio.run(main())