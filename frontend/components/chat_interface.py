"""
Chat interface component for AegisMedBot.
Handles user conversations with the multi-agent AI system.
"""

import gradio as gr
from typing import Dict, Any, Optional, List, Tuple
import requests
import json
import asyncio
import websockets
from datetime import datetime
import uuid
from loguru import logger
import time

class ChatInterface:
    """
    Manages the chat interface for interacting with AI agents.
    Provides real-time conversation with streaming responses.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the chat interface with configuration.
        
        Args:
            config: Application configuration dictionary containing API endpoints and settings
        """
        self.config = config
        self.api_base_url = config["api_base_url"]
        self.ws_url = config["ws_url"]
        
        # Session management
        self.conversation_id: Optional[str] = None
        self.session_token: Optional[str] = None
        self.message_history: List[Dict[str, Any]] = []
        
        # Rate limiting for API calls
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        
        logger.info("Chat interface initialized")
    
    def create_interface(self):
        """
        Create and configure the chat interface components.
        This method is called by the main app to build the UI.
        """
        logger.debug("Building chat interface components")
        
        with gr.Row():
            # Main chat area - takes 3/4 of the width
            with gr.Column(scale=3):
                # Chatbot display for conversation history
                self.chatbot = gr.Chatbot(
                    label="Clinical Conversation",
                    height=500,
                    bubble_full_width=False,
                    avatar_images=(None, "🏥"),  # User avatar, Bot avatar
                    show_label=True,
                    elem_classes="chat-window"
                )
                
                # Message input area with suggestions
                with gr.Row():
                    self.message_input = gr.Textbox(
                        label="Type your clinical question",
                        placeholder="Ask about patient care, hospital operations, clinical guidelines...",
                        lines=2,
                        scale=4
                    )
                    
                    self.send_button = gr.Button(
                        "Send",
                        variant="primary",
                        scale=1,
                        elem_classes="primary-button"
                    )
                
                # Control buttons row
                with gr.Row():
                    self.clear_button = gr.Button(
                        "Clear Conversation",
                        variant="secondary",
                        elem_classes="secondary-button"
                    )
                    
                    self.new_conversation_button = gr.Button(
                        "New Conversation",
                        variant="secondary",
                        elem_classes="secondary-button"
                    )
                    
                    self.export_button = gr.Button(
                        "Export Conversation",
                        variant="secondary",
                        elem_classes="secondary-button"
                    )
            
            # Sidebar - takes 1/4 of the width
            with gr.Column(scale=1):
                # Session information panel
                with gr.Group():
                    gr.Markdown("### 📋 Session Info")
                    
                    self.session_id_display = gr.Textbox(
                        label="Conversation ID",
                        value="Not started",
                        interactive=False
                    )
                    
                    self.message_count_display = gr.Number(
                        label="Messages",
                        value=0,
                        interactive=False
                    )
                    
                    self.active_agent_display = gr.Textbox(
                        label="Active Agent",
                        value="None",
                        interactive=False
                    )
                    
                    self.confidence_display = gr.HTML(
                        label="Confidence",
                        value="<span class='confidence-medium'>Waiting...</span>"
                    )
                
                # Suggested queries for quick access
                with gr.Group():
                    gr.Markdown("### 🎯 Suggested Queries")
                    
                    self.suggestion_buttons = []
                    suggestions = [
                        "What are the latest sepsis guidelines?",
                        "Show ICU bed occupancy",
                        "Predict post-op complication risk",
                        "Summarize today's hospital KPIs",
                        "Check drug interactions",
                        "Analyze ER patient flow"
                    ]
                    
                    for suggestion in suggestions:
                        btn = gr.Button(
                            suggestion,
                            size="sm",
                            elem_classes="secondary-button"
                        )
                        self.suggestion_buttons.append(btn)
                
                # Feedback section
                with gr.Group():
                    gr.Markdown("### ⭐ Feedback")
                    
                    self.feedback_rating = gr.Radio(
                        choices=["👍 Helpful", "👎 Not Helpful", "⚠️ Needs Review"],
                        label="Rate this response",
                        visible=False
                    )
                    
                    self.feedback_text = gr.Textbox(
                        label="Additional comments",
                        placeholder="Your feedback helps improve the system...",
                        lines=3,
                        visible=False
                    )
                    
                    self.submit_feedback = gr.Button(
                        "Submit Feedback",
                        variant="secondary",
                        visible=False
                    )
        
        # Set up event handlers for all interactive elements
        self._setup_event_handlers()
        
        logger.info("Chat interface components built successfully")
    
    def _setup_event_handlers(self):
        """
        Set up all event handlers for chat interactions.
        Connects UI elements to backend functions.
        """
        logger.debug("Setting up chat interface event handlers")
        
        # Send message handlers
        self.send_button.click(
            fn=self._handle_send_message,
            inputs=[self.message_input, self.chatbot],
            outputs=[self.message_input, self.chatbot, self.message_count_display, 
                    self.active_agent_display, self.confidence_display]
        )
        
        self.message_input.submit(
            fn=self._handle_send_message,
            inputs=[self.message_input, self.chatbot],
            outputs=[self.message_input, self.chatbot, self.message_count_display,
                    self.active_agent_display, self.confidence_display]
        )
        
        # Clear conversation handler
        self.clear_button.click(
            fn=self._handle_clear_conversation,
            inputs=[],
            outputs=[self.chatbot, self.message_count_display, self.session_id_display,
                    self.active_agent_display, self.confidence_display]
        )
        
        # New conversation handler
        self.new_conversation_button.click(
            fn=self._handle_new_conversation,
            inputs=[],
            outputs=[self.chatbot, self.message_count_display, self.session_id_display,
                    self.active_agent_display, self.confidence_display]
        )
        
        # Export conversation handler
        self.export_button.click(
            fn=self._handle_export_conversation,
            inputs=[self.chatbot],
            outputs=[]
        )
        
        # Suggestion button handlers
        for btn in self.suggestion_buttons:
            btn.click(
                fn=self._handle_suggestion_click,
                inputs=[btn],
                outputs=[self.message_input]
            )
        
        # Feedback handlers
        self.feedback_rating.change(
            fn=self._show_feedback_text,
            inputs=[self.feedback_rating],
            outputs=[self.feedback_text, self.submit_feedback]
        )
        
        self.submit_feedback.click(
            fn=self._handle_feedback_submit,
            inputs=[self.feedback_rating, self.feedback_text, self.chatbot],
            outputs=[self.feedback_rating, self.feedback_text, self.submit_feedback]
        )
    
    def _handle_send_message(self, message: str, history: List[Tuple]) -> Tuple[str, List[Tuple], int, str, str]:
        """
        Process user message and get AI response.
        
        Args:
            message: User's input message
            history: Current conversation history
            
        Returns:
            Tuple containing updated message input, conversation history, message count,
            active agent name, and confidence HTML
        """
        # Input validation
        if not message or not message.strip():
            logger.warning("Empty message received")
            return "", history, len(history), "None", "<span class='confidence-medium'>No message</span>"
        
        # Rate limiting check
        current_time = time.time()
        if current_time - self.last_request_time < self.min_request_interval:
            logger.warning("Rate limit exceeded")
            history.append((message, "Please wait a moment before sending another message."))
            return "", history, len(history), "None", "<span class='confidence-medium'>Rate limited</span>"
        
        self.last_request_time = current_time
        
        # Add user message to history
        history.append((message, None))
        
        try:
            # Prepare API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.session_token}" if self.session_token else ""
            }
            
            payload = {
                "query": message,
                "conversation_id": self.conversation_id,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "user_agent": "AegisMedBot-Frontend"
                }
            }
            
            logger.info(f"Sending request to {self.api_base_url}/api/v1/chat/")
            
            # Make API call to backend
            response = requests.post(
                f"{self.api_base_url}/api/v1/chat/",
                json=payload,
                headers=headers,
                timeout=30  # 30 second timeout
            )
            
            # Handle successful response
            if response.status_code == 200:
                data = response.json()
                
                # Update conversation state
                self.conversation_id = data["conversation_id"]
                self.session_id_display.value = self.conversation_id[:8] + "..."  # Truncate for display
                
                # Format the response with metadata
                formatted_response = self._format_response(data)
                
                # Update history with AI response
                history[-1] = (message, formatted_response)
                
                # Update session information
                message_count = len(history)
                active_agent = data["agent"].replace("_", " ").title()
                confidence_html = self._get_confidence_html(data["confidence"])
                
                logger.info(f"Response received from agent: {data['agent']} with confidence {data['confidence']}")
                
                # Show feedback options after response
                self.feedback_rating.visible = True
                
                return "", history, message_count, active_agent, confidence_html
            
            else:
                # Handle API error
                error_msg = f"API Error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                history[-1] = (message, f"⚠️ {error_msg}")
                
                return "", history, len(history), "Error", "<span class='confidence-low'>Error</span>"
        
        except requests.exceptions.ConnectionError:
            # Handle connection errors
            error_msg = "Cannot connect to backend server. Please ensure the server is running."
            logger.error(error_msg)
            history[-1] = (message, f"⚠️ {error_msg}")
            
            return "", history, len(history), "Offline", "<span class='confidence-low'>Offline</span>"
        
        except requests.exceptions.Timeout:
            # Handle timeout errors
            error_msg = "Request timed out. The server is taking too long to respond."
            logger.error(error_msg)
            history[-1] = (message, f"⚠️ {error_msg}")
            
            return "", history, len(history), "Timeout", "<span class='confidence-low'>Timeout</span>"
        
        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            history[-1] = (message, f"⚠️ {error_msg}")
            
            return "", history, len(history), "Error", "<span class='confidence-low'>Error</span>"
    
    def _format_response(self, data: Dict[str, Any]) -> str:
        """
        Format the API response with proper styling and metadata.
        
        Args:
            data: Response data from API
            
        Returns:
            Formatted response string with HTML styling
        """
        response_text = data["response"]
        
        # Get confidence indicator
        confidence = data["confidence"]
        if confidence >= 0.8:
            confidence_indicator = "🟢 High Confidence"
        elif confidence >= 0.5:
            confidence_indicator = "🟡 Medium Confidence"
        else:
            confidence_indicator = "🔴 Low Confidence - Please Verify"
        
        # Add human intervention warning if needed
        human_warning = ""
        if data.get("requires_human"):
            human_warning = "\n\n⚠️ **This response requires human review** - A clinician has been notified."
        
        # Add sources if available
        sources_text = ""
        if data.get("sources"):
            sources = data["sources"][:3]  # Show top 3 sources
            sources_text = "\n\n**Sources:**\n"
            for source in sources:
                if isinstance(source, dict):
                    source_name = source.get("source", "Unknown")
                    source_score = source.get("score", 0)
                    sources_text += f"• {source_name} (relevance: {source_score:.2f})\n"
                else:
                    sources_text += f"• {source}\n"
        
        # Add suggestions if available
        suggestions_text = ""
        if data.get("suggestions"):
            suggestions = data["suggestions"][:3]  # Show top 3 suggestions
            suggestions_text = "\n\n**Suggested follow-up questions:**\n"
            for suggestion in suggestions:
                suggestions_text += f"• {suggestion}\n"
        
        # Combine all parts
        formatted = f"""
{response_text}

---
📊 *{confidence_indicator}*
🤖 *Agent: {data['agent'].replace('_', ' ').title()}*
⏱️ *Response time: {data.get('processing_time_ms', 0):.0f}ms*
{human_warning}
{sources_text}
{suggestions_text}
        """
        
        return formatted
    
    def _get_confidence_html(self, confidence: float) -> str:
        """
        Generate HTML for confidence indicator.
        
        Args:
            confidence: Confidence score between 0 and 1
            
        Returns:
            HTML string with appropriate styling
        """
        if confidence >= 0.8:
            return f"<span class='confidence-high'>High Confidence ({confidence:.2f})</span>"
        elif confidence >= 0.5:
            return f"<span class='confidence-medium'>Medium Confidence ({confidence:.2f})</span>"
        else:
            return f"<span class='confidence-low'>Low Confidence ({confidence:.2f})</span>"
    
    def _handle_clear_conversation(self) -> Tuple[List, int, str, str, str]:
        """
        Clear the current conversation history.
        
        Returns:
            Updated UI state
        """
        logger.info("Clearing conversation history")
        self.message_history = []
        return [], 0, "Not started", "None", "<span class='confidence-medium'>Cleared</span>"
    
    def _handle_new_conversation(self) -> Tuple[List, int, str, str, str]:
        """
        Start a new conversation with a fresh session.
        
        Returns:
            Updated UI state
        """
        logger.info("Starting new conversation")
        self.conversation_id = None
        self.message_history = []
        return [], 0, "New", "None", "<span class='confidence-medium'>Ready</span>"
    
    def _handle_export_conversation(self, history: List[Tuple]):
        """
        Export the conversation history as JSON.
        
        Args:
            history: Current conversation history
        """
        if not history:
            gr.Warning("No conversation to export")
            return
        
        # Prepare export data
        export_data = {
            "conversation_id": self.conversation_id,
            "timestamp": datetime.now().isoformat(),
            "messages": []
        }
        
        for user_msg, assistant_msg in history:
            export_data["messages"].append({
                "role": "user",
                "content": user_msg,
                "timestamp": datetime.now().isoformat()
            })
            if assistant_msg:
                export_data["messages"].append({
                    "role": "assistant",
                    "content": assistant_msg,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Create filename with timestamp
        filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save to file
        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)
        
        gr.Info(f"Conversation exported to {filename}")
        logger.info(f"Conversation exported to {filename}")
    
    def _handle_suggestion_click(self, suggestion: str) -> str:
        """
        Handle click on suggestion button.
        
        Args:
            suggestion: The suggested query text
            
        Returns:
            The suggestion text to populate in input
        """
        logger.debug(f"Suggestion clicked: {suggestion}")
        return suggestion
    
    def _show_feedback_text(self, rating: str) -> Tuple[gr.Textbox, gr.Button]:
        """
        Show feedback text input when rating is selected.
        
        Args:
            rating: Selected rating value
            
        Returns:
            Updated visibility states
        """
        if rating:
            return gr.Textbox(visible=True), gr.Button(visible=True)
        return gr.Textbox(visible=False), gr.Button(visible=False)
    
    def _handle_feedback_submit(self, rating: str, feedback: str, history: List[Tuple]):
        """
        Submit user feedback to backend.
        
        Args:
            rating: User's rating
            feedback: Additional feedback text
            history: Current conversation history
            
        Returns:
            Reset feedback components
        """
        try:
            # Prepare feedback data
            feedback_data = {
                "conversation_id": self.conversation_id,
                "rating": rating,
                "feedback_text": feedback,
                "timestamp": datetime.now().isoformat(),
                "message_count": len(history)
            }
            
            # Send to backend
            response = requests.post(
                f"{self.api_base_url}/api/v1/chat/feedback",
                json=feedback_data,
                headers={"Authorization": f"Bearer {self.session_token}" if self.session_token else ""}
            )
            
            if response.status_code == 200:
                gr.Info("Thank you for your feedback!")
                logger.info("Feedback submitted successfully")
            else:
                gr.Warning("Failed to submit feedback")
                logger.warning(f"Feedback submission failed: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error submitting feedback: {str(e)}")
            gr.Warning("Could not submit feedback")
        
        # Reset feedback components
        return gr.Radio(visible=False), gr.Textbox(visible=False), gr.Button(visible=False)