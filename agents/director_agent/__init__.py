"""
Director Agent Module for AegisMedBot Hospital Intelligence Platform.

This module provides strategic intelligence capabilities for hospital leadership,
including KPI analysis, performance metrics, and executive report generation.
The Director Agent serves as the high-level strategic advisor to the Medical Director
and hospital administration.

Key Features:
    - Hospital-wide KPI monitoring and analysis
    - Department performance tracking
    - Financial metrics analysis
    - Clinical outcomes assessment
    - Executive report generation
    - Strategic recommendations
    - Trend analysis and forecasting
    - Benchmarking against industry standards

Author: AegisMedBot Team
Version: 1.0.0
"""

from .director_intelligence import DirectorIntelligenceAgent
from .kpi_analyzer import KPIAnalyzer
from .report_generator import ReportGenerator

__all__ = [
    'DirectorIntelligenceAgent',
    'KPIAnalyzer',
    'ReportGenerator'
]

__version__ = '1.0.0'
__author__ = 'AegisMedBot Team'