"""
Executive dashboard component for AegisMedBot.
Displays hospital metrics, KPIs, and real-time operational data.
"""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import random
from loguru import logger

class DashboardComponent:
    """
    Manages the executive dashboard for hospital leadership.
    Displays real-time metrics, KPIs, and operational insights.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dashboard component with configuration.
        
        Args:
            config: Application configuration dictionary
        """
        self.config = config
        self.refresh_interval = 30  # Refresh every 30 seconds
        self.metrics_history = []  # Store historical metrics for trends
        
        logger.info("Dashboard component initialized")
    
    def create_interface(self):
        """
        Create and configure the dashboard interface.
        """
        logger.debug("Building dashboard interface")
        
        with gr.Column():
            # Header with key metrics
            with gr.Row():
                gr.Markdown("## 🏥 Hospital Operations Dashboard")
                self.refresh_button = gr.Button("🔄 Refresh Data", variant="primary", scale=0)
            
            # Key Performance Indicators (KPIs) row
            with gr.Row():
                self.kpi_cards = self._create_kpi_cards()
            
            # Charts row
            with gr.Row():
                with gr.Column(scale=1):
                    self.occupancy_chart = gr.Plot(label="Bed Occupancy by Department")
                with gr.Column(scale=1):
                    self.patient_flow_chart = gr.Plot(label="Patient Flow (24h)")
            
            # Second charts row
            with gr.Row():
                with gr.Column(scale=1):
                    self.staff_utilization = gr.Plot(label="Staff Utilization")
                with gr.Column(scale=1):
                    self.quality_metrics = gr.Plot(label="Quality Metrics")
            
            # Detailed metrics table
            with gr.Row():
                self.metrics_table = gr.Dataframe(
                    label="Department Performance Metrics",
                    headers=["Department", "Patients", "Avg Wait Time", "Staff Ratio", "Occupancy %", "Quality Score"],
                    interactive=False
                )
            
            # Alerts and notifications
            with gr.Row():
                self.alerts_panel = gr.HTML(label="Critical Alerts")
            
            # Auto-refresh using JavaScript (simulated with button click)
            self.refresh_button.click(
                fn=self._refresh_dashboard,
                inputs=[],
                outputs=[self.occupancy_chart, self.patient_flow_chart, 
                        self.staff_utilization, self.quality_metrics,
                        self.metrics_table, self.alerts_panel] + 
                        list(self.kpi_cards.values())
            )
            
            # Initial load
            self._refresh_dashboard()
            
            logger.info("Dashboard interface built successfully")
    
    def _create_kpi_cards(self) -> Dict[str, gr.HTML]:
        """
        Create KPI cards for key metrics.
        
        Returns:
            Dictionary of KPI HTML components
        """
        kpis = {}
        
        with gr.Row():
            kpis["bed_occupancy"] = gr.HTML(
                value=self._format_kpi_card("Bed Occupancy", "87%", "↑ 3%", "normal"),
                elem_classes="metric-card"
            )
            
            kpis["avg_wait_time"] = gr.HTML(
                value=self._format_kpi_card("Avg Wait Time", "24 min", "↓ 5 min", "good"),
                elem_classes="metric-card"
            )
            
            kpis["patient_satisfaction"] = gr.HTML(
                value=self._format_kpi_card("Patient Satisfaction", "4.2/5", "↑ 0.3", "good"),
                elem_classes="metric-card"
            )
            
            kpis["staff_utilization"] = gr.HTML(
                value=self._format_kpi_card("Staff Utilization", "92%", "↑ 2%", "warning"),
                elem_classes="metric-card"
            )
        
        return kpis
    
    def _format_kpi_card(self, title: str, value: str, trend: str, status: str) -> str:
        """
        Format HTML for KPI card.
        
        Args:
            title: KPI title
            value: Current value
            trend: Trend indicator
            status: Status color (good, warning, critical)
            
        Returns:
            HTML string for KPI card
        """
        status_colors = {
            "good": "#059669",
            "warning": "#b45309",
            "critical": "#b91c1c",
            "normal": "#0b4f6c"
        }
        
        color = status_colors.get(status, "#0b4f6c")
        
        return f"""
        <div style="text-align: center;">
            <div class="metric-label">{title}</div>
            <div class="metric-value" style="color: {color};">{value}</div>
            <div style="font-size: 0.875rem; color: {trend.startswith('↑') and '#059669' or '#b91c1c'}">
                {trend}
            </div>
        </div>
        """
    
    def _refresh_dashboard(self):
        """
        Refresh all dashboard data with current metrics.
        
        Returns:
            Updated dashboard components
        """
        logger.info("Refreshing dashboard data")
        
        # Generate simulated hospital data
        # In production, this would fetch from actual APIs
        occupancy_data = self._generate_occupancy_data()
        patient_flow_data = self._generate_patient_flow_data()
        staff_data = self._generate_staff_utilization_data()
        quality_data = self._generate_quality_metrics()
        department_metrics = self._generate_department_metrics()
        alerts = self._generate_alerts()
        
        # Create visualizations
        occupancy_chart = self._create_occupancy_chart(occupancy_data)
        patient_flow_chart = self._create_patient_flow_chart(patient_flow_data)
        staff_chart = self._create_staff_chart(staff_data)
        quality_chart = self._create_quality_chart(quality_data)
        
        # Update KPI cards
        kpi_values = self._calculate_kpis(occupancy_data, patient_flow_data, staff_data)
        
        return (
            occupancy_chart,
            patient_flow_chart,
            staff_chart,
            quality_chart,
            department_metrics,
            alerts,
            kpi_values["bed_occupancy"],
            kpi_values["avg_wait_time"],
            kpi_values["patient_satisfaction"],
            kpi_values["staff_utilization"]
        )
    
    def _generate_occupancy_data(self) -> pd.DataFrame:
        """
        Generate simulated bed occupancy data.
        
        Returns:
            DataFrame with occupancy by department
        """
        departments = ["ICU", "Cardiology", "Emergency", "Surgery", "Pediatrics", "Maternity"]
        
        data = []
        for dept in departments:
            data.append({
                "department": dept,
                "total_beds": random.randint(20, 50),
                "occupied_beds": random.randint(15, 45),
                "available_beds": random.randint(0, 15),
                "occupancy_rate": random.uniform(0.6, 0.95)
            })
        
        return pd.DataFrame(data)
    
    def _generate_patient_flow_data(self) -> pd.DataFrame:
        """
        Generate simulated patient flow data for last 24 hours.
        
        Returns:
            DataFrame with hourly patient counts
        """
        hours = list(range(24))
        current_hour = datetime.now().hour
        
        data = []
        for hour_offset in range(24):
            hour = (current_hour - hour_offset) % 24
            data.append({
                "hour": f"{hour:02d}:00",
                "admissions": random.randint(2, 10),
                "discharges": random.randint(1, 8),
                "transfers": random.randint(0, 5),
                "er_visits": random.randint(5, 20)
            })
        
        # Sort by hour
        df = pd.DataFrame(data)
        df = df.sort_values("hour")
        
        return df
    
    def _generate_staff_utilization_data(self) -> pd.DataFrame:
        """
        Generate simulated staff utilization data.
        
        Returns:
            DataFrame with staff metrics by department
        """
        departments = ["ICU", "Cardiology", "Emergency", "Surgery", "Pediatrics", "Maternity"]
        
        data = []
        for dept in departments:
            data.append({
                "department": dept,
                "available_staff": random.randint(5, 20),
                "scheduled_staff": random.randint(4, 18),
                "on_duty": random.randint(3, 15),
                "utilization_rate": random.uniform(0.7, 0.95)
            })
        
        return pd.DataFrame(data)
    
    def _generate_quality_metrics(self) -> pd.DataFrame:
        """
        Generate simulated quality metrics.
        
        Returns:
            DataFrame with quality indicators
        """
        metrics = [
            "Patient Satisfaction",
            "Readmission Rate",
            "Infection Rate",
            "Medication Errors",
            "Fall Rate",
            "Mortality Rate"
        ]
        
        data = []
        for metric in metrics:
            if "Rate" in metric:
                value = random.uniform(0.01, 0.15)
                target = 0.05
            elif "Satisfaction" in metric:
                value = random.uniform(3.5, 4.8)
                target = 4.0
            else:
                value = random.uniform(0, 10)
                target = 5
            
            data.append({
                "metric": metric,
                "value": round(value, 2),
                "target": target,
                "status": "good" if value <= target * 1.1 else "warning" if value <= target * 1.2 else "critical"
            })
        
        return pd.DataFrame(data)
    
    def _generate_department_metrics(self) -> pd.DataFrame:
        """
        Generate detailed department metrics table.
        
        Returns:
            DataFrame with department-level metrics
        """
        departments = ["ICU", "Cardiology", "Emergency", "Surgery", "Pediatrics", "Maternity"]
        
        data = []
        for dept in departments:
            data.append([
                dept,
                random.randint(10, 50),  # Patients
                f"{random.randint(5, 45)} min",  # Avg wait time
                f"{random.uniform(1, 3):.1f}:1",  # Staff ratio
                f"{random.randint(65, 95)}%",  # Occupancy
                f"{random.randint(85, 100)}%"  # Quality score
            ])
        
        return pd.DataFrame(
            data,
            columns=["Department", "Patients", "Avg Wait Time", "Staff Ratio", "Occupancy %", "Quality Score"]
        )
    
    def _generate_alerts(self) -> str:
        """
        Generate critical alerts based on current metrics.
        
        Returns:
            HTML string with alerts
        """
        alerts = []
        
        # Simulate alerts based on random conditions
        if random.random() > 0.7:
            alerts.append(("critical", "ICU at 95% capacity - Consider diverting non-critical patients"))
        
        if random.random() > 0.6:
            alerts.append(("warning", "ER wait times exceeding 45 minutes in the last hour"))
        
        if random.random() > 0.8:
            alerts.append(("critical", "Ventilator shortage in Respiratory Care"))
        
        if random.random() > 0.5:
            alerts.append(("info", "Staff meeting at 3:00 PM in Conference Room A"))
        
        if not alerts:
            alerts.append(("info", "All systems operating normally"))
        
        # Format alerts as HTML
        alert_html = "<div style='max-height: 200px; overflow-y: auto;'>"
        for severity, message in alerts:
            if severity == "critical":
                alert_html += f"<div class='alert-critical'>🔴 <strong>CRITICAL:</strong> {message}</div>"
            elif severity == "warning":
                alert_html += f"<div class='alert-warning'>🟡 <strong>WARNING:</strong> {message}</div>"
            else:
                alert_html += f"<div class='alert-info'>ℹ️ <strong>INFO:</strong> {message}</div>"
        
        alert_html += "</div>"
        
        return alert_html
    
    def _calculate_kpis(self, occupancy: pd.DataFrame, flow: pd.DataFrame, staff: pd.DataFrame) -> Dict[str, str]:
        """
        Calculate KPI values and format for display.
        
        Returns:
            Dictionary of formatted KPI HTML strings
        """
        # Calculate overall occupancy
        total_occupied = occupancy["occupied_beds"].sum()
        total_beds = occupancy["total_beds"].sum()
        occupancy_rate = (total_occupied / total_beds) * 100
        
        # Calculate average wait time
        avg_wait = flow["er_visits"].mean() * 3  # Simulated wait time
        
        # Calculate satisfaction (simulated)
        satisfaction = 4.2 + random.uniform(-0.3, 0.3)
        
        # Calculate staff utilization
        total_on_duty = staff["on_duty"].sum()
        total_scheduled = staff["scheduled_staff"].sum()
        utilization = (total_on_duty / total_scheduled) * 100 if total_scheduled > 0 else 0
        
        # Determine trends (simulated)
        occupancy_trend = "↑" if random.random() > 0.5 else "↓"
        wait_trend = "↓" if avg_wait < 25 else "↑"
        satisfaction_trend = "↑" if random.random() > 0.4 else "↓"
        utilization_trend = "↑" if utilization > 85 else "↓"
        
        return {
            "bed_occupancy": self._format_kpi_card(
                "Bed Occupancy",
                f"{occupancy_rate:.0f}%",
                f"{occupancy_trend} {random.randint(1, 5)}%",
                "warning" if occupancy_rate > 90 else "normal"
            ),
            "avg_wait_time": self._format_kpi_card(
                "Avg Wait Time",
                f"{avg_wait:.0f} min",
                f"{wait_trend} {random.randint(1, 10)} min",
                "good" if avg_wait < 20 else "warning" if avg_wait < 30 else "critical"
            ),
            "patient_satisfaction": self._format_kpi_card(
                "Patient Satisfaction",
                f"{satisfaction:.1f}/5",
                f"{satisfaction_trend} {random.uniform(0.1, 0.5):.1f}",
                "good" if satisfaction > 4.0 else "warning"
            ),
            "staff_utilization": self._format_kpi_card(
                "Staff Utilization",
                f"{utilization:.0f}%",
                f"{utilization_trend} {random.randint(1, 3)}%",
                "warning" if utilization > 90 else "good"
            )
        }
    
    def _create_occupancy_chart(self, data: pd.DataFrame) -> go.Figure:
        """
        Create occupancy bar chart.
        
        Args:
            data: Occupancy DataFrame
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add bars for occupied beds
        fig.add_trace(go.Bar(
            name="Occupied",
            x=data["department"],
            y=data["occupied_beds"],
            marker_color="#0b4f6c",
            text=data["occupied_beds"],
            textposition="auto"
        ))
        
        # Add bars for available beds
        fig.add_trace(go.Bar(
            name="Available",
            x=data["department"],
            y=data["available_beds"],
            marker_color="#1a7f9e",
            text=data["available_beds"],
            textposition="auto"
        ))
        
        fig.update_layout(
            title="Bed Occupancy by Department",
            xaxis_title="Department",
            yaxis_title="Number of Beds",
            barmode="stack",
            template="plotly_white",
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def _create_patient_flow_chart(self, data: pd.DataFrame) -> go.Figure:
        """
        Create patient flow line chart.
        
        Args:
            data: Patient flow DataFrame
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add line for admissions
        fig.add_trace(go.Scatter(
            name="Admissions",
            x=data["hour"],
            y=data["admissions"],
            mode="lines+markers",
            line=dict(color="#0b4f6c", width=3),
            marker=dict(size=8)
        ))
        
        # Add line for discharges
        fig.add_trace(go.Scatter(
            name="Discharges",
            x=data["hour"],
            y=data["discharges"],
            mode="lines+markers",
            line=dict(color="#059669", width=3),
            marker=dict(size=8)
        ))
        
        # Add line for ER visits
        fig.add_trace(go.Scatter(
            name="ER Visits",
            x=data["hour"],
            y=data["er_visits"],
            mode="lines+markers",
            line=dict(color="#b45309", width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Patient Flow - Last 24 Hours",
            xaxis_title="Time",
            yaxis_title="Number of Patients",
            template="plotly_white",
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def _create_staff_chart(self, data: pd.DataFrame) -> go.Figure:
        """
        Create staff utilization chart.
        
        Args:
            data: Staff utilization DataFrame
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add bars for staff on duty
        fig.add_trace(go.Bar(
            name="On Duty",
            x=data["department"],
            y=data["on_duty"],
            marker_color="#0b4f6c",
            text=data["on_duty"],
            textposition="auto"
        ))
        
        # Add line for utilization rate
        fig.add_trace(go.Scatter(
            name="Utilization %",
            x=data["department"],
            y=data["utilization_rate"] * 100,
            mode="lines+markers",
            line=dict(color="#b45309", width=3, dash="dash"),
            marker=dict(size=10),
            yaxis="y2"
        ))
        
        fig.update_layout(
            title="Staff Utilization by Department",
            xaxis_title="Department",
            yaxis_title="Number of Staff",
            yaxis2=dict(
                title="Utilization %",
                overlaying="y",
                side="right",
                range=[0, 100]
            ),
            template="plotly_white",
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def _create_quality_chart(self, data: pd.DataFrame) -> go.Figure:
        """
        Create quality metrics gauge chart.
        
        Args:
            data: Quality metrics DataFrame
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Create a horizontal bar chart for metrics
        colors = []
        for _, row in data.iterrows():
            if row["status"] == "good":
                colors.append("#059669")
            elif row["status"] == "warning":
                colors.append("#b45309")
            else:
                colors.append("#b91c1c")
        
        fig.add_trace(go.Bar(
            name="Current Value",
            x=data["value"],
            y=data["metric"],
            orientation="h",
            marker_color=colors,
            text=data["value"],
            textposition="auto"
        ))
        
        # Add target line
        fig.add_trace(go.Scatter(
            name="Target",
            x=data["target"],
            y=data["metric"],
            mode="markers",
            marker=dict(symbol="line-ns", size=15, color="black", line=dict(width=2)),
            showlegend=True
        ))
        
        fig.update_layout(
            title="Quality Metrics vs Targets",
            xaxis_title="Value",
            yaxis_title="Metric",
            template="plotly_white",
            height=400,
            showlegend=True,
            barmode="group"
        )
        
        return fig