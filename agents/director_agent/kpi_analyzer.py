"""
KPI Analyzer Module for Director Intelligence Agent.

This module provides comprehensive analysis of Key Performance Indicators
for hospital management. It includes statistical analysis, trend detection,
benchmarking, and predictive analytics for all hospital metrics.

The analyzer supports:
    - Multi-dimensional KPI analysis
    - Statistical process control
    - Anomaly detection
    - Correlation analysis
    - Benchmarking against industry standards
    - Predictive modeling
    - Performance scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from scipy.signal import find_peaks
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AnalysisPeriod(Enum):
    """Time periods for analysis."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class StatisticalMethod(Enum):
    """Statistical methods for analysis."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    STD_DEV = "std_dev"
    VARIANCE = "variance"
    PERCENTILE = "percentile"
    Z_SCORE = "z_score"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"

@dataclass
class StatisticalSummary:
    """Statistical summary of a KPI dataset."""
    count: int
    mean: float
    median: float
    mode: Optional[float]
    std_dev: float
    variance: float
    min_value: float
    max_value: float
    range: float
    q1: float  # First quartile
    q3: float  # Third quartile
    iqr: float  # Interquartile range
    skewness: float
    kurtosis: float
    outliers: List[float]

@dataclass
class TrendAnalysis:
    """Trend analysis results for a KPI."""
    direction: str  # 'up', 'down', 'stable'
    slope: float
    intercept: float
    r_squared: float  # Coefficient of determination
    p_value: float  # Statistical significance
    confidence_interval: Tuple[float, float]
    seasonality: Optional[Dict[str, float]]
    changepoints: List[datetime]

@dataclass
class BenchmarkComparison:
    """Benchmark comparison results."""
    kpi_name: str
    current_value: float
    benchmark_value: float
    benchmark_source: str
    percentile_rank: float  # 0-100
    performance_gap: float  # Current - Benchmark
    performance_ratio: float  # Current / Benchmark
    assessment: str  # 'above', 'meeting', 'below'

class KPIAnalyzer:
    """
    Comprehensive KPI Analyzer for hospital metrics.
    
    This class provides advanced analytics capabilities for all hospital KPIs,
    including statistical analysis, trend detection, anomaly identification,
    and benchmarking against industry standards.
    """
    
    def __init__(
        self,
        kpi_definitions: Dict[str, Dict[str, Any]],
        thresholds: Dict[str, Any],
        benchmarking_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the KPI Analyzer.
        
        Args:
            kpi_definitions: Dictionary of KPI definitions
            thresholds: Dictionary of threshold values
            benchmarking_data: Optional benchmarking data from industry sources
        """
        self.kpi_definitions = kpi_definitions
        self.thresholds = thresholds
        self.benchmarking_data = benchmarking_data or self._load_default_benchmarks()
        
        # Analysis cache
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
        logger.info("KPI Analyzer initialized successfully")
    
    def _load_default_benchmarks(self) -> Dict[str, Any]:
        """
        Load default benchmarking data from industry standards.
        
        Returns:
            Dictionary of benchmark values by KPI
        """
        return {
            "mortality_rate": {
                "source": "CMS Hospital Compare",
                "national_average": 1.2,
                "top_performer": 0.8,
                "teaching_hospitals": 1.4,
                "community_hospitals": 1.1
            },
            "readmission_rate": {
                "source": "CMS Hospital Compare",
                "national_average": 15.3,
                "top_performer": 12.0,
                "teaching_hospitals": 16.5,
                "community_hospitals": 14.8
            },
            "hospital_acquired_infections": {
                "source": "CDC NHSN",
                "national_average": 1.5,
                "top_performer": 0.8,
                "teaching_hospitals": 1.8,
                "community_hospitals": 1.3
            },
            "average_length_of_stay": {
                "source": "AHA Annual Survey",
                "national_average": 5.2,
                "top_performer": 4.5,
                "teaching_hospitals": 5.8,
                "community_hospitals": 4.9
            },
            "bed_occupancy_rate": {
                "source": "AHA Annual Survey",
                "national_average": 78.5,
                "top_performer": 85.0,
                "teaching_hospitals": 82.3,
                "community_hospitals": 75.8
            },
            "emergency_department_wait_time": {
                "source": "CMS Hospital Compare",
                "national_average": 28.0,
                "top_performer": 18.0,
                "teaching_hospitals": 32.5,
                "community_hospitals": 25.0
            },
            "patient_satisfaction_score": {
                "source": "HCAHPS",
                "national_average": 8.2,
                "top_performer": 9.1,
                "teaching_hospitals": 8.0,
                "community_hospitals": 8.4
            }
        }
    
    def calculate_statistics(
        self,
        data: List[float],
        method: StatisticalMethod = StatisticalMethod.MEAN
    ) -> float:
        """
        Calculate basic statistics for a dataset.
        
        Args:
            data: List of numerical values
            method: Statistical method to apply
            
        Returns:
            Calculated statistic
        """
        if not data:
            return 0.0
        
        data_array = np.array(data)
        
        if method == StatisticalMethod.MEAN:
            return float(np.mean(data_array))
        elif method == StatisticalMethod.MEDIAN:
            return float(np.median(data_array))
        elif method == StatisticalMethod.MODE:
            mode_result = stats.mode(data_array)
            return float(mode_result.mode[0]) if len(mode_result.mode) > 0 else 0.0
        elif method == StatisticalMethod.STD_DEV:
            return float(np.std(data_array))
        elif method == StatisticalMethod.VARIANCE:
            return float(np.var(data_array))
        elif method == StatisticalMethod.PERCENTILE:
            return float(np.percentile(data_array, 75))  # Default to 75th percentile
        else:
            return float(np.mean(data_array))
    
    def get_comprehensive_statistics(self, data: List[float]) -> StatisticalSummary:
        """
        Generate comprehensive statistical summary for a dataset.
        
        Args:
            data: List of numerical values
            
        Returns:
            StatisticalSummary object with all statistics
        """
        if not data:
            return StatisticalSummary(
                count=0, mean=0.0, median=0.0, mode=None,
                std_dev=0.0, variance=0.0, min_value=0.0,
                max_value=0.0, range=0.0, q1=0.0, q3=0.0,
                iqr=0.0, skewness=0.0, kurtosis=0.0, outliers=[]
            )
        
        data_array = np.array(data)
        
        # Basic statistics
        count = len(data)
        mean = float(np.mean(data_array))
        median = float(np.median(data_array))
        
        # Mode calculation
        try:
            mode_result = stats.mode(data_array)
            mode = float(mode_result.mode[0]) if len(mode_result.mode) > 0 else None
        except:
            mode = None
        
        std_dev = float(np.std(data_array))
        variance = float(np.var(data_array))
        min_value = float(np.min(data_array))
        max_value = float(np.max(data_array))
        data_range = max_value - min_value
        
        # Quartiles
        q1 = float(np.percentile(data_array, 25))
        q3 = float(np.percentile(data_array, 75))
        iqr = q3 - q1
        
        # Shape statistics
        skewness = float(stats.skew(data_array))
        kurtosis = float(stats.kurtosis(data_array))
        
        # Detect outliers using IQR method
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        
        return StatisticalSummary(
            count=count,
            mean=mean,
            median=median,
            mode=mode,
            std_dev=std_dev,
            variance=variance,
            min_value=min_value,
            max_value=max_value,
            range=data_range,
            q1=q1,
            q3=q3,
            iqr=iqr,
            skewness=skewness,
            kurtosis=kurtosis,
            outliers=outliers
        )
    
    def analyze_trend(
        self,
        timestamps: List[datetime],
        values: List[float],
        period: AnalysisPeriod = AnalysisPeriod.DAILY
    ) -> TrendAnalysis:
        """
        Perform trend analysis on time series data.
        
        Args:
            timestamps: List of timestamps
            values: List of corresponding values
            period: Analysis period granularity
            
        Returns:
            TrendAnalysis object with trend results
        """
        if len(values) < 3:
            return TrendAnalysis(
                direction="insufficient_data",
                slope=0.0,
                intercept=0.0,
                r_squared=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                seasonality=None,
                changepoints=[]
            )
        
        # Convert to numpy arrays
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2
        
        # Determine direction
        if abs(slope) < 0.01:  # Less than 1% change per unit
            direction = "stable"
        elif slope > 0:
            direction = "up"
        else:
            direction = "down"
        
        # Confidence interval (95%)
        confidence_interval = (
            intercept + slope * len(values) - 1.96 * std_err,
            intercept + slope * len(values) + 1.96 * std_err
        )
        
        # Detect seasonality (simple method)
        seasonality = self._detect_seasonality(values, period)
        
        # Detect changepoints
        changepoints = self._detect_changepoints(timestamps, values)
        
        return TrendAnalysis(
            direction=direction,
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            p_value=p_value,
            confidence_interval=confidence_interval,
            seasonality=seasonality,
            changepoints=changepoints
        )
    
    def _detect_seasonality(
        self,
        values: List[float],
        period: AnalysisPeriod
    ) -> Optional[Dict[str, float]]:
        """
        Detect seasonal patterns in the data.
        
        Args:
            values: List of values
            period: Expected periodicity
            
        Returns:
            Dictionary with seasonality information or None
        """
        if len(values) < 14:  # Need at least 2 weeks for seasonality
            return None
        
        # Determine expected cycle length based on period
        cycle_lengths = {
            AnalysisPeriod.HOURLY: 24,  # Daily cycle
            AnalysisPeriod.DAILY: 7,     # Weekly cycle
            AnalysisPeriod.WEEKLY: 4,     # Monthly cycle
            AnalysisPeriod.MONTHLY: 12,   # Yearly cycle
        }
        
        expected_cycle = cycle_lengths.get(period, 7)
        
        # Autocorrelation to detect cycles
        values_array = np.array(values)
        autocorr = [1] + [np.corrcoef(values_array[:-i], values_array[i:])[0, 1] 
                          for i in range(1, min(30, len(values)//2))]
        
        # Find peaks in autocorrelation
        peaks, properties = find_peaks(autocorr, height=0.3)
        
        if len(peaks) > 0:
            # Check if peak corresponds to expected cycle
            cycle_strength = autocorr[peaks[0]]
            if abs(peaks[0] - expected_cycle) <= 2:  # Close to expected cycle
                return {
                    "cycle_length": int(peaks[0]),
                    "strength": float(cycle_strength),
                    "pattern_detected": True
                }
        
        return {
            "cycle_length": None,
            "strength": 0.0,
            "pattern_detected": False
        }
    
    def _detect_changepoints(
        self,
        timestamps: List[datetime],
        values: List[float],
        sensitivity: float = 3.0
    ) -> List[datetime]:
        """
        Detect significant change points in time series.
        
        Args:
            timestamps: List of timestamps
            values: List of values
            sensitivity: Number of standard deviations for change detection
            
        Returns:
            List of timestamps where significant changes occurred
        """
        if len(values) < 10:
            return []
        
        changepoints = []
        values_array = np.array(values)
        
        # Calculate rolling statistics
        window = min(7, len(values) // 4)
        
        for i in range(window, len(values) - window):
            # Compare before and after windows
            before = values_array[i - window:i]
            after = values_array[i:i + window]
            
            before_mean = np.mean(before)
            after_mean = np.mean(after)
            before_std = np.std(before)
            
            # Check for significant change
            if before_std > 0:
                z_score = abs(after_mean - before_mean) / (before_std / np.sqrt(window))
                if z_score > sensitivity:
                    changepoints.append(timestamps[i])
        
        return changepoints
    
    def detect_anomalies(
        self,
        values: List[float],
        method: str = "zscore",
        threshold: float = 3.0
    ) -> List[Tuple[int, float, str]]:
        """
        Detect anomalies in the data.
        
        Args:
            values: List of values
            method: Detection method ('zscore', 'iqr', 'isolation_forest')
            threshold: Threshold for anomaly detection
            
        Returns:
            List of tuples (index, value, reason)
        """
        if len(values) < 5:
            return []
        
        anomalies = []
        values_array = np.array(values)
        
        if method == "zscore":
            # Z-score method
            mean = np.mean(values_array)
            std = np.std(values_array)
            
            if std > 0:
                z_scores = np.abs((values_array - mean) / std)
                for i, z in enumerate(z_scores):
                    if z > threshold:
                        anomalies.append((i, values[i], f"Z-score: {z:.2f}"))
        
        elif method == "iqr":
            # IQR method
            q1 = np.percentile(values_array, 25)
            q3 = np.percentile(values_array, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for i, val in enumerate(values):
                if val < lower_bound:
                    anomalies.append((i, val, f"Below IQR lower bound: {lower_bound:.2f}"))
                elif val > upper_bound:
                    anomalies.append((i, val, f"Above IQR upper bound: {upper_bound:.2f}"))
        
        return anomalies
    
    def calculate_correlations(
        self,
        data_dict: Dict[str, List[float]]
    ) -> pd.DataFrame:
        """
        Calculate correlations between multiple KPIs.
        
        Args:
            data_dict: Dictionary mapping KPI names to lists of values
            
        Returns:
            DataFrame with correlation matrix
        """
        # Create DataFrame from dictionary
        df = pd.DataFrame(data_dict)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Store for later use
        self.correlation_matrix = corr_matrix
        
        return corr_matrix
    
    def get_strong_correlations(
        self,
        kpi_name: str,
        threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Get KPIs strongly correlated with a given KPI.
        
        Args:
            kpi_name: Name of the KPI to analyze
            threshold: Correlation threshold
            
        Returns:
            List of (kpi_name, correlation) tuples
        """
        if self.correlation_matrix is None or kpi_name not in self.correlation_matrix.columns:
            return []
        
        correlations = []
        series = self.correlation_matrix[kpi_name]
        
        for other_kpi, corr in series.items():
            if other_kpi != kpi_name and abs(corr) >= threshold:
                correlations.append((other_kpi, float(corr)))
        
        return sorted(correlations, key=lambda x: abs(x[1]), reverse=True)
    
    def benchmark_kpi(
        self,
        kpi_name: str,
        current_value: float,
        benchmark_type: str = "national_average"
    ) -> BenchmarkComparison:
        """
        Compare a KPI against industry benchmarks.
        
        Args:
            kpi_name: Name of the KPI
            current_value: Current KPI value
            benchmark_type: Type of benchmark to use
            
        Returns:
            BenchmarkComparison object
        """
        benchmark_data = self.benchmarking_data.get(kpi_name, {})
        
        if not benchmark_data:
            return BenchmarkComparison(
                kpi_name=kpi_name,
                current_value=current_value,
                benchmark_value=0.0,
                benchmark_source="unknown",
                percentile_rank=50.0,
                performance_gap=0.0,
                performance_ratio=1.0,
                assessment="unknown"
            )
        
        benchmark_value = benchmark_data.get(benchmark_type, benchmark_data.get("national_average", 0.0))
        benchmark_source = benchmark_data.get("source", "unknown")
        
        # Calculate percentile rank (simplified)
        # In production, this would use actual distribution data
        kpi_def = self.kpi_definitions.get(kpi_name, {})
        direction = kpi_def.get("direction", "lower_is_better")
        
        if direction == "lower_is_better":
            # Lower is better
            if current_value <= benchmark_value:
                percentile_rank = 75 + 25 * (1 - current_value / benchmark_value)
            else:
                percentile_rank = 50 - 50 * (current_value / benchmark_value - 1)
        else:
            # Higher is better
            if current_value >= benchmark_value:
                percentile_rank = 75 + 25 * (current_value / benchmark_value - 1)
            else:
                percentile_rank = 50 - 50 * (1 - current_value / benchmark_value)
        
        # Cap at 0-100
        percentile_rank = max(0, min(100, percentile_rank))
        
        # Calculate performance gap and ratio
        performance_gap = current_value - benchmark_value
        if benchmark_value != 0:
            performance_ratio = current_value / benchmark_value
        else:
            performance_ratio = 1.0
        
        # Determine assessment
        if direction == "lower_is_better":
            if current_value < benchmark_value * 0.9:
                assessment = "above"  # Better than benchmark
            elif current_value > benchmark_value * 1.1:
                assessment = "below"  # Worse than benchmark
            else:
                assessment = "meeting"
        else:
            if current_value > benchmark_value * 1.1:
                assessment = "above"
            elif current_value < benchmark_value * 0.9:
                assessment = "below"
            else:
                assessment = "meeting"
        
        return BenchmarkComparison(
            kpi_name=kpi_name,
            current_value=current_value,
            benchmark_value=benchmark_value,
            benchmark_source=benchmark_source,
            percentile_rank=percentile_rank,
            performance_gap=performance_gap,
            performance_ratio=performance_ratio,
            assessment=assessment
        )
    
    def predict_future_values(
        self,
        values: List[float],
        periods_ahead: int = 7,
        method: str = "exponential_smoothing"
    ) -> List[float]:
        """
        Predict future values using time series forecasting.
        
        Args:
            values: Historical values
            periods_ahead: Number of periods to predict
            method: Forecasting method
            
        Returns:
            List of predicted values
        """
        if len(values) < 3:
            return [values[-1] if values else 0.0] * periods_ahead
        
        if method == "exponential_smoothing":
            # Simple exponential smoothing
            alpha = 0.3  # Smoothing factor
            predictions = []
            last_value = values[-1]
            
            for i in range(periods_ahead):
                if i == 0:
                    # First prediction uses last value and trend
                    if len(values) >= 2:
                        trend = values[-1] - values[-2]
                    else:
                        trend = 0
                    predicted = last_value + alpha * trend
                else:
                    # Subsequent predictions use previous prediction
                    predicted = predictions[-1]
                
                predictions.append(predicted)
            
            return predictions
        
        elif method == "moving_average":
            # Moving average
            window = min(7, len(values))
            ma = np.mean(values[-window:])
            return [ma] * periods_ahead
        
        else:
            # Linear trend
            x = np.arange(len(values))
            y = np.array(values)
            z = np.polyfit(x, y, 1)
            
            predictions = []
            for i in range(periods_ahead):
                next_x = len(values) + i
                predicted = z[0] * next_x + z[1]
                predictions.append(predicted)
            
            return predictions
    
    def calculate_kpi_score(
        self,
        kpi_name: str,
        value: float,
        scale: Tuple[float, float] = (0, 100)
    ) -> float:
        """
        Calculate a normalized score for a KPI.
        
        Args:
            kpi_name: Name of the KPI
            value: Current value
            scale: Min and max of output scale
            
        Returns:
            Normalized score
        """
        if kpi_name not in self.thresholds:
            return scale[1] / 2  # Default to middle value
        
        threshold = self.thresholds[kpi_name]
        kpi_def = self.kpi_definitions.get(kpi_name, {})
        direction = kpi_def.get("direction", "lower_is_better")
        
        min_output, max_output = scale
        
        if direction == "lower_is_better":
            if value <= threshold.target_min:
                # Excellent - at or below target
                return max_output
            elif value >= threshold.critical_upper:
                # Critical - at or above critical upper
                return min_output
            else:
                # Linear interpolation between target and critical
                score_range = threshold.critical_upper - threshold.target_min
                if score_range > 0:
                    position = (value - threshold.target_min) / score_range
                    return max_output * (1 - position)
                else:
                    return max_output / 2
        else:
            # Higher is better
            if value >= threshold.target_max:
                # Excellent - at or above target
                return max_output
            elif value <= threshold.critical_lower:
                # Critical - at or below critical lower
                return min_output
            else:
                # Linear interpolation between critical and target
                score_range = threshold.target_max - threshold.critical_lower
                if score_range > 0:
                    position = (value - threshold.critical_lower) / score_range
                    return max_output * position
                else:
                    return max_output / 2
    
    def generate_performance_dashboard(
        self,
        current_kpis: Dict[str, float],
        historical_data: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive performance dashboard.
        
        Args:
            current_kpis: Current KPI values
            historical_data: Historical KPI data
            
        Returns:
            Dictionary with dashboard data
        """
        dashboard = {
            "generated_at": datetime.now().isoformat(),
            "summary": {},
            "kpi_details": [],
            "trends": {},
            "anomalies": [],
            "correlations": [],
            "benchmarks": [],
            "predictions": {}
        }
        
        # Analyze each KPI
        for kpi_name, current_value in current_kpis.items():
            history = historical_data.get(kpi_name, [])
            
            # Statistical summary
            if history:
                stats_summary = self.get_comprehensive_statistics(history)
                
                # Trend analysis
                if len(history) >= 7:
                    # Create dummy timestamps (in production, use real timestamps)
                    timestamps = [datetime.now() - timedelta(days=i) for i in range(len(history)-1, -1, -1)]
                    trend = self.analyze_trend(timestamps, history)
                    dashboard["trends"][kpi_name] = {
                        "direction": trend.direction,
                        "slope": trend.slope,
                        "r_squared": trend.r_squared,
                        "p_value": trend.p_value
                    }
                
                # Anomaly detection
                anomalies = self.detect_anomalies(history)
                for idx, val, reason in anomalies:
                    dashboard["anomalies"].append({
                        "kpi": kpi_name,
                        "index": idx,
                        "value": val,
                        "reason": reason,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Benchmarking
                benchmark = self.benchmark_kpi(kpi_name, current_value)
                dashboard["benchmarks"].append({
                    "kpi": kpi_name,
                    "current": current_value,
                    "benchmark": benchmark.benchmark_value,
                    "assessment": benchmark.assessment,
                    "percentile": benchmark.percentile_rank
                })
                
                # Predictions
                if len(history) >= 14:
                    predictions = self.predict_future_values(history, periods_ahead=7)
                    dashboard["predictions"][kpi_name] = predictions
                
                # KPI score
                score = self.calculate_kpi_score(kpi_name, current_value)
                
                dashboard["kpi_details"].append({
                    "name": kpi_name,
                    "current_value": current_value,
                    "score": score,
                    "statistics": {
                        "mean": stats_summary.mean,
                        "median": stats_summary.median,
                        "std_dev": stats_summary.std_dev,
                        "min": stats_summary.min_value,
                        "max": stats_summary.max_value,
                        "outliers_count": len(stats_summary.outliers)
                    }
                })
        
        # Calculate correlations if we have enough data
        if len(historical_data) >= 3:
            try:
                corr_matrix = self.calculate_correlations(historical_data)
                dashboard["correlations"] = corr_matrix.to_dict()
            except:
                pass
        
        # Overall summary
        avg_score = np.mean([d["score"] for d in dashboard["kpi_details"]]) if dashboard["kpi_details"] else 0
        dashboard["summary"] = {
            "total_kpis": len(dashboard["kpi_details"]),
            "average_score": round(avg_score, 1),
            "total_anomalies": len(dashboard["anomalies"]),
            "kpis_below_benchmark": len([b for b in dashboard["benchmarks"] if b["assessment"] == "below"]),
            "kpis_above_benchmark": len([b for b in dashboard["benchmarks"] if b["assessment"] == "above"])
        }
        
        return dashboard

# Utility functions

def create_kpi_analyzer(
    kpi_definitions: Dict[str, Any],
    thresholds: Dict[str, Any]
) -> KPIAnalyzer:
    """
    Factory function to create a KPI Analyzer instance.
    
    Args:
        kpi_definitions: KPI definitions dictionary
        thresholds: Thresholds dictionary
        
    Returns:
        Configured KPIAnalyzer instance
    """
    return KPIAnalyzer(kpi_definitions, thresholds)

if __name__ == "__main__":
    # Example usage
    from director_intelligence import DirectorIntelligenceAgent
    
    agent = DirectorIntelligenceAgent("test_agent")
    analyzer = KPIAnalyzer(agent.kpi_definitions, agent.thresholds)
    
    # Simulate data
    historical = {
        "mortality_rate": [1.2, 1.3, 1.1, 1.4, 1.2, 1.3, 1.5, 1.2, 1.1, 1.3],
        "readmission_rate": [14.5, 15.2, 14.8, 16.1, 15.3, 14.9, 15.7, 14.6, 15.1, 14.8]
    }
    
    current = {
        "mortality_rate": 1.4,
        "readmission_rate": 15.2
    }
    
    dashboard = analyzer.generate_performance_dashboard(current, historical)
    print(json.dumps(dashboard, indent=2))