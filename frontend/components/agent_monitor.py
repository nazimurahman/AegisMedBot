"""
Agent monitor component for AegisMedBot.
Provides real-time monitoring of AI agent status, performance, and activities.
"""

import gradio as gr
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import random
import time
from loguru import logger

class AgentMonitorComponent:
    """
    Manages the agent monitoring interface.
    Displays real-time status, performance metrics, and agent activities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize agent monitor with configuration.
        
        Args:
            config: Application configuration dictionary
        """
        self.config = config
        self.monitoring_active = False
        self.metrics_history = []
        
        # Define agent configurations
        self.agents = {
            "clinical_agent": {
                "name": "Clinical Knowledge Agent",
                "icon": "🏥",
                "description": "Medical literature and guidelines",
                "status": "active",
                "tasks_completed": 0,
                "avg_response_time": 0,
                "avg_confidence": 0,
                "last_active": None
            },
            "risk_agent": {
                "name": "Risk Prediction Agent",
                "icon": "⚠️",
                "description": "Patient risk assessment",
                "status": "active",
                "tasks_completed": 0,
                "avg_response_time": 0,
                "avg_confidence": 0,
                "last_active": None
            },
            "operations_agent": {
                "name": "Operations Agent",
                "icon": "⚙️",
                "description": "Resource and flow optimization",
                "status": "active",
                "tasks_completed": 0,
                "avg_response_time": 0,
                "avg_confidence": 0,
                "last_active": None
            },
            "director_agent": {
                "name": "Director Intelligence Agent",
                "icon": "📊",
                "description": "Executive insights and KPIs",
                "status": "active",
                "tasks_completed": 0,
                "avg_response_time": 0,
                "avg_confidence": 0,
                "last_active": None
            },
            "compliance_agent": {
                "name": "Compliance Agent",
                "icon": "🔒",
                "description": "Privacy and audit logging",
                "status": "active",
                "tasks_completed": 0,
                "avg_response_time": 0,
                "avg_confidence": 0,
                "last_active": None
            },
            "research_agent": {
                "name": "Research Assistant",
                "icon": "🔬",
                "description": "Literature research and summarization",
                "status": "active",
                "tasks_completed": 0,
                "avg_response_time": 0,
                "avg_confidence": 0,
                "last_active": None
            }
        }
        
        logger.info("Agent monitor component initialized")
    
    def create_interface(self):
        """
        Create and configure the agent monitoring interface.
        """
        logger.debug("Building agent monitor interface")
        
        with gr.Column():
            # Header with controls
            with gr.Row():
                gr.Markdown("## 🤖 AI Agent Monitor")
                
                self.refresh_rate = gr.Dropdown(
                    choices=["Real-time", "5 sec", "10 sec", "30 sec", "Paused"],
                    value="10 sec",
                    label="Refresh Rate",
                    scale=1
                )
                
                self.monitor_toggle = gr.Button(
                    "▶️ Start Monitoring",
                    variant="primary",
                    scale=1
                )
                
                self.reset_stats = gr.Button(
                    "🔄 Reset Stats",
                    variant="secondary",
                    scale=1
                )
            
            # Agent status cards
            with gr.Row():
                self.agent_cards = self._create_agent_cards()
            
            # Performance metrics
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⏱️ Response Times")
                    self.response_time_chart = gr.Plot(label="Agent Response Times")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 Confidence Scores")
                    self.confidence_chart = gr.Plot(label="Agent Confidence Levels")
            
            # Activity timeline
            with gr.Row():
                gr.Markdown("### 📋 Recent Agent Activities")
                self.activity_timeline = gr.Dataframe(
                    headers=["Time", "Agent", "Task", "Duration", "Confidence", "Status"],
                    interactive=False
                )
            
            # System health metrics
            with gr.Row():
                with gr.Column(scale=1):
                    self.system_health = gr.HTML(label="System Health")
                
                with gr.Column(scale=1):
                    self.resource_usage = gr.Plot(label="Resource Usage")
            
            # Connect event handlers
            self.monitor_toggle.click(
                fn=self._toggle_monitoring,
                inputs=[self.monitor_toggle],
                outputs=[self.monitor_toggle]
            )
            
            self.reset_stats.click(
                fn=self._reset_statistics,
                inputs=[],
                outputs=[self.activity_timeline]
            )
            
            # Start periodic updates
            self._start_periodic_updates()
            
            logger.info("Agent monitor interface built successfully")
    
    def _create_agent_cards(self) -> Dict[str, gr.HTML]:
        """
        Create status cards for each agent.
        
        Returns:
            Dictionary of agent card HTML components
        """
        cards = {}
        
        # Create a grid of agent cards (2 rows, 3 columns)
        agent_list = list(self.agents.items())
        
        for i in range(0, len(agent_list), 3):
            with gr.Row():
                for j in range(3):
                    if i + j < len(agent_list):
                        agent_id, agent_info = agent_list[i + j]
                        card_id = f"agent_card_{agent_id}"
                        
                        cards[agent_id] = gr.HTML(
                            value=self._format_agent_card(agent_id, agent_info),
                            elem_classes="metric-card"
                        )
        
        return cards
    
    def _format_agent_card(self, agent_id: str, agent_info: Dict) -> str:
        """
        Format HTML for agent status card.
        
        Args:
            agent_id: Agent identifier
            agent_info: Agent information dictionary
            
        Returns:
            HTML string for agent card
        """
        # Determine status class
        status_class = {
            "active": "agent-status-active",
            "busy": "agent-status-busy",
            "error": "agent-status-error",
            "idle": "agent-status-active"
        }.get(agent_info["status"], "agent-status-active")
        
        # Format status emoji
        status_emoji = {
            "active": "🟢",
            "busy": "🟡",
            "error": "🔴",
            "idle": "⚪"
        }.get(agent_info["status"], "⚪")
        
        return f"""
        <div style="padding: 1rem;">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 2rem; margin-right: 0.5rem;">{agent_info['icon']}</span>
                <div>
                    <h3 style="margin: 0; color: #0b4f6c;">{agent_info['name']}</h3>
                    <p style="margin: 0; color: #64748b; font-size: 0.875rem;">{agent_info['description']}</p>
                </div>
            </div>
            <div style="margin-top: 1rem;">
                <span class="{status_class}">
                    {status_emoji} {agent_info['status'].title()}
                </span>
            </div>
            <div style="margin-top: 0.5rem; display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                <div>
                    <div style="font-size: 0.75rem; color: #64748b;">Tasks</div>
                    <div style="font-size: 1.25rem; font-weight: 600;">{agent_info['tasks_completed']}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748b;">Response</div>
                    <div style="font-size: 1.25rem; font-weight: 600;">{agent_info['avg_response_time']:.0f}ms</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748b;">Confidence</div>
                    <div style="font-size: 1.25rem; font-weight: 600;">{agent_info['avg_confidence']*100:.0f}%</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748b;">Last Active</div>
                    <div style="font-size: 0.875rem;">{agent_info['last_active'] or 'Never'}</div>
                </div>
            </div>
        </div>
        """
    
    def _toggle_monitoring(self, button_text: str) -> str:
        """
        Toggle monitoring on/off.
        
        Args:
            button_text: Current button text
            
        Returns:
            Updated button text
        """
        if "Start" in button_text:
            self.monitoring_active = True
            logger.info("Monitoring started")
            return "⏸️ Pause Monitoring"
        else:
            self.monitoring_active = False
            logger.info("Monitoring paused")
            return "▶️ Start Monitoring"
    
    def _reset_statistics(self):
        """
        Reset all agent statistics.
        
        Returns:
            Empty activity timeline
        """
        logger.info("Resetting agent statistics")
        
        # Reset agent metrics
        for agent_id in self.agents:
            self.agents[agent_id]["tasks_completed"] = 0
            self.agents[agent_id]["avg_response_time"] = 0
            self.agents[agent_id]["avg_confidence"] = 0
            self.agents[agent_id]["last_active"] = None
        
        # Clear history
        self.metrics_history = []
        
        return pd.DataFrame(columns=["Time", "Agent", "Task", "Duration", "Confidence", "Status"])
    
    def _start_periodic_updates(self):
        """
        Start periodic updates for monitoring data.
        This would typically use JavaScript for real updates, but here we simulate.
        """
        # In a real implementation, this would use JavaScript callbacks
        # For Gradio, we'll rely on refresh events
        pass
    
    def _update_monitoring_data(self):
        """
        Update all monitoring data with current metrics.
        This method would be called periodically.
        
        Returns:
            Updated UI components
        """
        if not self.monitoring_active:
            return
        
        # Update agent metrics
        self._simulate_agent_activity()
        
        # Update charts
        response_chart = self._create_response_time_chart()
        confidence_chart = self._create_confidence_chart()
        
        # Update activity timeline
        activities = self._generate_activity_data()
        
        # Update system health
        health_html = self._get_system_health()
        
        # Update resource usage
        resource_chart = self._create_resource_chart()
        
        # Update agent cards
        updated_cards = {}
        for agent_id, agent_info in self.agents.items():
            updated_cards[agent_id] = self._format_agent_card(agent_id, agent_info)
        
        return (
            response_chart,
            confidence_chart,
            activities,
            health_html,
            resource_chart,
            *list(updated_cards.values())
        )
    
    def _simulate_agent_activity(self):
        """
        Simulate agent activity for demonstration.
        In production, this would fetch real metrics from the backend.
        """
        current_time = datetime.now()
        
        for agent_id, agent_info in self.agents.items():
            # Randomly update metrics
            if random.random() > 0.7:  # 30% chance of activity
                # Simulate a task
                task_duration = random.uniform(100, 2000)
                task_confidence = random.uniform(0.6, 0.95)
                
                # Update cumulative metrics
                agent_info["tasks_completed"] += 1
                
                # Update moving average for response time
                n = agent_info["tasks_completed"]
                old_avg = agent_info["avg_response_time"]
                agent_info["avg_response_time"] = (old_avg * (n-1) + task_duration) / n
                
                # Update moving average for confidence
                old_conf = agent_info["avg_confidence"]
                agent_info["avg_confidence"] = (old_conf * (n-1) + task_confidence) / n
                
                agent_info["last_active"] = current_time.strftime("%H:%M:%S")
                
                # Add to history
                self.metrics_history.append({
                    "timestamp": current_time,
                    "agent": agent_id,
                    "duration": task_duration,
                    "confidence": task_confidence
                })
                
                # Update status randomly
                if random.random() > 0.8:
                    agent_info["status"] = random.choice(["active", "busy", "active", "active"])
                else:
                    agent_info["status"] = "active"
    
    def _create_response_time_chart(self) -> go.Figure:
        """
        Create response time chart for agents.
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Prepare data
        agents = []
        response_times = []
        colors = []
        
        for agent_id, agent_info in self.agents.items():
            agents.append(agent_info["name"])
            response_times.append(agent_info["avg_response_time"])
            
            # Color based on performance
            if agent_info["avg_response_time"] < 500:
                colors.append("#059669")  # Good
            elif agent_info["avg_response_time"] < 1000:
                colors.append("#b45309")  # Warning
            else:
                colors.append("#b91c1c")  # Critical
        
        fig.add_trace(go.Bar(
            x=agents,
            y=response_times,
            marker_color=colors,
            text=[f"{t:.0f}ms" for t in response_times],
            textposition="auto"
        ))
        
        fig.update_layout(
            title="Average Response Times by Agent",
            xaxis_title="Agent",
            yaxis_title="Response Time (ms)",
            template="plotly_white",
            height=300,
            showlegend=False
        )
        
        # Add threshold line
        fig.add_hline(
            y=1000,
            line_dash="dash",
            line_color="red",
            annotation_text="Warning Threshold",
            annotation_position="bottom right"
        )
        
        return fig
    
    def _create_confidence_chart(self) -> go.Figure:
        """
        Create confidence score chart for agents.
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Prepare data
        agents = []
        confidences = []
        colors = []
        
        for agent_id, agent_info in self.agents.items():
            agents.append(agent_info["name"])
            confidences.append(agent_info["avg_confidence"] * 100)
            
            # Color based on confidence
            if agent_info["avg_confidence"] >= 0.8:
                colors.append("#059669")  # Good
            elif agent_info["avg_confidence"] >= 0.6:
                colors.append("#b45309")  # Warning
            else:
                colors.append("#b91c1c")  # Critical
        
        fig.add_trace(go.Bar(
            x=agents,
            y=confidences,
            marker_color=colors,
            text=[f"{c:.0f}%" for c in confidences],
            textposition="auto"
        ))
        
        fig.update_layout(
            title="Average Confidence Scores by Agent",
            xaxis_title="Agent",
            yaxis_title="Confidence (%)",
            template="plotly_white",
            height=300,
            showlegend=False,
            yaxis=dict(range=[0, 100])
        )
        
        # Add threshold lines
        fig.add_hline(
            y=80,
            line_dash="dash",
            line_color="green",
            annotation_text="Target",
            annotation_position="bottom right"
        )
        
        fig.add_hline(
            y=60,
            line_dash="dash",
            line_color="orange",
            annotation_text="Minimum",
            annotation_position="bottom right"
        )
        
        return fig
    
    def _generate_activity_data(self) -> pd.DataFrame:
        """
        Generate recent activity data.
        
        Returns:
            DataFrame with recent activities
        """
        if not self.metrics_history:
            return pd.DataFrame(columns=["Time", "Agent", "Task", "Duration", "Confidence", "Status"])
        
        # Get last 10 activities
        recent = self.metrics_history[-10:]
        recent.reverse()  # Show most recent first
        
        data = []
        for activity in recent:
            agent_name = self.agents[activity["agent"]]["name"]
            task_types = ["Query", "Analysis", "Retrieval", "Prediction", "Report"]
            
            data.append([
                activity["timestamp"].strftime("%H:%M:%S"),
                agent_name,
                random.choice(task_types),
                f"{activity['duration']:.0f}ms",
                f"{activity['confidence']*100:.0f}%",
                random.choice(["✅", "✅", "✅", "⚠️", "✅"])  # Random status
            ])
        
        return pd.DataFrame(
            data,
            columns=["Time", "Agent", "Task", "Duration", "Confidence", "Status"]
        )
    
    def _get_system_health(self) -> str:
        """
        Generate system health HTML.
        
        Returns:
            HTML string with system health metrics
        """
        # Calculate overall metrics
        total_tasks = sum(a["tasks_completed"] for a in self.agents.values())
        avg_confidence = np.mean([a["avg_confidence"] for a in self.agents.values()])
        
        # Determine overall status
        if avg_confidence >= 0.8:
            status = "🟢 Excellent"
            status_color = "#059669"
        elif avg_confidence >= 0.6:
            status = "🟡 Good"
            status_color = "#b45309"
        else:
            status = "🔴 Needs Attention"
            status_color = "#b91c1c"
        
        # Calculate error rate (simulated)
        error_rate = random.uniform(0, 0.05)
        
        html = f"""
        <div style="padding: 1rem;">
            <h4>System Health Overview</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div>
                    <div style="font-size: 0.875rem; color: #64748b;">Overall Status</div>
                    <div style="font-size: 1.5rem; font-weight: 600; color: {status_color};">{status}</div>
                </div>
                <div>
                    <div style="font-size: 0.875rem; color: #64748b;">Total Tasks</div>
                    <div style="font-size: 1.5rem; font-weight: 600;">{total_tasks}</div>
                </div>
                <div>
                    <div style="font-size: 0.875rem; color: #64748b;">Avg Confidence</div>
                    <div style="font-size: 1.5rem; font-weight: 600;">{avg_confidence*100:.1f}%</div>
                </div>
                <div>
                    <div style="font-size: 0.875rem; color: #64748b;">Error Rate</div>
                    <div style="font-size: 1.5rem; font-weight: 600;">{error_rate*100:.2f}%</div>
                </div>
            </div>
            <div style="margin-top: 1rem;">
                <div style="font-size: 0.875rem; color: #64748b;">Active Agents</div>
                <div style="display: flex; gap: 0.5rem; margin-top: 0.25rem;">
                    {''.join([f"<span class='agent-status-active'>🟢 {a['name'].split()[0]}</span>" 
                             for a in self.agents.values() if a['status'] == 'active'][:4])}
                </div>
            </div>
        </div>
        """
        
        return html
    
    def _create_resource_chart(self) -> go.Figure:
        """
        Create resource usage chart.
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Simulate resource usage
        categories = ["CPU", "Memory", "API Calls", "Database", "Vector DB"]
        current_usage = [random.uniform(20, 80) for _ in categories]
        limits = [100, 100, 1000, 100, 100]
        
        fig.add_trace(go.Bar(
            name="Current Usage",
            x=categories,
            y=current_usage,
            marker_color="#0b4f6c",
            text=[f"{u:.0f}%" for u in current_usage],
            textposition="auto"
        ))
        
        fig.add_trace(go.Scatter(
            name="Limit",
            x=categories,
            y=limits,
            mode="lines+markers",
            line=dict(color="red", width=2, dash="dash"),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Resource Usage",
            xaxis_title="Resource",
            yaxis_title="Usage (%)",
            template="plotly_white",
            height=300,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig