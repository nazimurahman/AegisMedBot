"""
Main entry point for AegisMedBot frontend application.
This file initializes the Gradio interface and orchestrates all UI components.
"""

import gradio as gr
from typing import Dict, Any, Optional, List, Tuple
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# Import custom components
from components.chat_interface import ChatInterface
from components.dashboard import DashboardComponent
from components.analytics import AnalyticsComponent
from components.agent_monitor import AgentMonitorComponent

# Configure logging for production monitoring
from loguru import logger
import sys

# Remove default handler and add custom formatter
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/frontend_{time}.log",
    rotation="500 MB",
    retention="30 days",
    compression="zip",
    level="DEBUG"
)

# Load environment variables
load_dotenv()

class AegisMedBotFrontend:
    """
    Main frontend application class that manages the Gradio interface.
    This class orchestrates all UI components and handles user interactions.
    """
    
    def __init__(self):
        """
        Initialize the frontend application with configuration and components.
        """
        logger.info("Initializing AegisMedBot Frontend Application")
        
        # Application configuration from environment variables
        self.config = {
            "api_base_url": os.getenv("API_BASE_URL", "http://localhost:8000"),
            "ws_url": os.getenv("WS_URL", "ws://localhost:8000"),
            "environment": os.getenv("ENVIRONMENT", "development"),
            "app_title": "AegisMedBot - Hospital Intelligence Platform",
            "app_version": "1.0.0",
            "company_name": "Aegis Health Systems",
            "support_email": "support@aegismedbot.com"
        }
        
        logger.info(f"Configuration loaded: Environment={self.config['environment']}, API URL={self.config['api_base_url']}")
        
        # Initialize UI components
        self.chat_interface = ChatInterface(self.config)
        self.dashboard_component = DashboardComponent(self.config)
        self.analytics_component = AnalyticsComponent(self.config)
        self.agent_monitor = AgentMonitorComponent(self.config)
        
        # Track active sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("All components initialized successfully")
    
    def create_interface(self) -> gr.Blocks:
        """
        Create and configure the main Gradio interface with all tabs.
        
        Returns:
            gr.Blocks: Configured Gradio interface
        """
        logger.info("Creating Gradio interface with multiple tabs")
        
        # Define custom CSS for professional healthcare styling
        custom_css = """
        /* Main container styling for healthcare professional look */
        .gradio-container {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #f8fafc;
        }
        
        /* Header styling with hospital branding */
        .app-header {
            background: linear-gradient(135deg, #0b4f6c 0%, #1a7f9e 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 0 0 1rem 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        /* Chat message bubbles for clinical conversations */
        .user-message {
            background-color: #e3f2fd;
            border-radius: 1rem 1rem 1rem 0;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid #0b4f6c;
        }
        
        .assistant-message {
            background-color: white;
            border-radius: 1rem 1rem 0 1rem;
            padding: 1rem;
            margin: 0.5rem 0;
            border-right: 4px solid #1a7f9e;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* Confidence indicators for AI responses */
        .confidence-high {
            color: #059669;
            font-weight: 600;
            background-color: #d1fae5;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            display: inline-block;
            font-size: 0.875rem;
        }
        
        .confidence-medium {
            color: #b45309;
            background-color: #ffedd5;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            display: inline-block;
            font-size: 0.875rem;
        }
        
        .confidence-low {
            color: #b91c1c;
            background-color: #fee2e2;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            display: inline-block;
            font-size: 0.875rem;
        }
        
        /* Metric cards for dashboard display */
        .metric-card {
            background: white;
            border-radius: 0.75rem;
            padding: 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid #e2e8f0;
            transition: transform 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #0b4f6c;
            line-height: 1.2;
        }
        
        .metric-label {
            color: #64748b;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Status indicators for agents */
        .agent-status-active {
            color: #059669;
            background-color: #d1fae5;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            display: inline-flex;
            align-items: center;
        }
        
        .agent-status-busy {
            color: #b45309;
            background-color: #ffedd5;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
        }
        
        .agent-status-error {
            color: #b91c1c;
            background-color: #fee2e2;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
        }
        
        /* Button styling for medical actions */
        .primary-button {
            background-color: #0b4f6c;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.375rem;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .primary-button:hover {
            background-color: #1a7f9e;
        }
        
        .secondary-button {
            background-color: white;
            color: #0b4f6c;
            border: 1px solid #0b4f6c;
            padding: 0.5rem 1rem;
            border-radius: 0.375rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .secondary-button:hover {
            background-color: #f0f9ff;
        }
        
        /* Alert and notification styling */
        .alert-critical {
            background-color: #fee2e2;
            border-left: 4px solid #b91c1c;
            padding: 1rem;
            border-radius: 0.375rem;
            margin: 0.5rem 0;
        }
        
        .alert-warning {
            background-color: #ffedd5;
            border-left: 4px solid #b45309;
            padding: 1rem;
            border-radius: 0.375rem;
        }
        
        .alert-info {
            background-color: #e0f2fe;
            border-left: 4px solid #0284c7;
            padding: 1rem;
            border-radius: 0.375rem;
        }
        """
        
        # Create the main Gradio blocks interface
        with gr.Blocks(
            title=self.config["app_title"],
            theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
            css=custom_css,
            analytics_enabled=False  # Disable Gradio analytics for privacy
        ) as demo:
            
            # Application header with branding
            with gr.Row(elem_classes="app-header"):
                gr.Markdown(f"""
                # 🏥 {self.config['app_title']}
                ### Version {self.config['app_version']} | {self.config['company_name']}
                
                Secure AI-powered clinical intelligence for hospital leadership and medical staff.
                
                *All interactions are encrypted and audited for HIPAA compliance.*
                """)
            
            # Create tabs for different functional areas
            with gr.Tabs() as tabs:
                
                # Tab 1: Chat Assistant - Main interaction point
                with gr.TabItem("💬 Clinical Assistant", id="chat_tab"):
                    self.chat_interface.create_interface()
                
                # Tab 2: Executive Dashboard - Hospital metrics
                with gr.TabItem("📊 Executive Dashboard", id="dashboard_tab"):
                    self.dashboard_component.create_interface()
                
                # Tab 3: Analytics & Insights - Data analysis
                with gr.TabItem("📈 Analytics & Insights", id="analytics_tab"):
                    self.analytics_component.create_interface()
                
                # Tab 4: Agent Monitor - System status
                with gr.TabItem("🤖 Agent Monitor", id="monitor_tab"):
                    self.agent_monitor.create_interface()
            
            # Footer with system status and support information
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"""
                    ---
                    **System Status:** 🟢 Operational  
                    **Environment:** {self.config['environment'].title()}  
                    **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    """)
                
                with gr.Column(scale=1):
                    gr.Markdown(f"""
                    **Support:** {self.config['support_email']}  
                    **Documentation:** [API Reference]({self.config['api_base_url']}/docs)  
                    **Privacy Policy:** [HIPAA Compliance](#)
                    """)
        
        logger.info("Gradio interface created successfully")
        return demo
    
    def launch(self):
        """
        Launch the Gradio application with production settings.
        """
        logger.info("Launching AegisMedBot frontend application")
        
        # Create the interface
        demo = self.create_interface()
        
        # Configure launch parameters based on environment
        if self.config["environment"] == "production":
            # Production settings - more secure
            demo.launch(
                server_name="0.0.0.0",  # Listen on all interfaces
                server_port=7860,
                share=False,  # Don't create public link
                debug=False,
                auth=None,  # Authentication handled by reverse proxy
                ssl_verify=False,
                quiet=True  # Reduce console output
            )
        else:
            # Development settings - more verbose
            demo.launch(
                server_name="127.0.0.1",
                server_port=7860,
                share=False,
                debug=True,
                show_api=True
            )
        
        logger.info("Application launched successfully")

# Entry point for running the application
if __name__ == "__main__":
    """
    Main entry point when script is executed directly.
    """
    try:
        # Create and launch the frontend application
        app = AegisMedBotFrontend()
        app.launch()
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise