"""
Unit tests for the FastAPI backend endpoints.
Tests API routing, authentication, request validation, and response formatting.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from unittest.mock import patch, AsyncMock, Mock
import json

# Import the FastAPI app
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.main import app
from backend.core.config import settings
from backend.models.schemas.response import ChatResponse


class TestChatAPI:
    """
    Test suite for chat-related API endpoints.
    Tests message processing, conversation management, and WebSocket connections.
    """
    
    @pytest.fixture
    def client(self):
        """
        Create a test client for the FastAPI application.
        
        Returns:
            TestClient: Configured test client for API testing
        """
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """
        Create authentication headers for testing.
        
        Returns:
            dict: Headers with authentication token
        """
        # In production, this would be a real JWT token
        return {"Authorization": "Bearer test_token_123"}
    
    def test_root_endpoint(self, client):
        """
        Test the root endpoint returns correct API information.
        Verifies API metadata is accessible.
        """
        response = client.get("/")
        
        # Verify response status
        assert response.status_code == 200
        
        # Verify response content
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert data["name"] == settings.PROJECT_NAME
        assert data["status"] == "operational"
    
    def test_health_check(self, client):
        """
        Test health check endpoint for monitoring.
        Verifies API health status for Kubernetes probes.
        """
        response = client.get("/health")
        
        # Verify health check response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "services" in data
    
    @patch('backend.api.routes.chat.get_orchestrator')
    @patch('backend.api.routes.chat.get_current_user')
    def test_chat_endpoint_success(self, mock_user, mock_orchestrator, client, auth_headers):
        """
        Test successful chat message processing.
        Verifies API correctly handles valid chat requests.
        """
        # Mock the orchestrator response
        mock_orchestrator_instance = AsyncMock()
        mock_orchestrator_instance.process_message.return_value = {
            "conversation_id": "conv_test_123",
            "response": {"response": "This is a test response from the AI assistant"},
            "agent": "clinical_agent",
            "confidence": 0.85,
            "requires_human": False
        }
        mock_orchestrator.return_value = mock_orchestrator_instance
        
        # Mock current user
        mock_user.return_value = {"id": "test_user", "role": "doctor"}
        
        # Send chat request
        request_data = {
            "query": "What are the symptoms of pneumonia?",
            "conversation_id": None,
            "patient_id": None,
            "metadata": {}
        }
        
        response = client.post(
            "/api/v1/chat/",
            json=request_data,
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data
        assert "response" in data
        assert "agent" in data
        assert "confidence" in data
        assert data["response"] is not None
    
    @patch('backend.api.routes.chat.get_orchestrator')
    @patch('backend.api.routes.chat.get_current_user')
    def test_chat_endpoint_with_conversation_id(self, mock_user, mock_orchestrator, client, auth_headers):
        """
        Test chat with existing conversation ID.
        Verifies API correctly handles continuing conversations.
        """
        # Mock the orchestrator response
        mock_orchestrator_instance = AsyncMock()
        mock_orchestrator_instance.process_message.return_value = {
            "conversation_id": "existing_conv_456",
            "response": {"response": "Following up on our previous discussion..."},
            "agent": "clinical_agent",
            "confidence": 0.9,
            "requires_human": False
        }
        mock_orchestrator.return_value = mock_orchestrator_instance
        
        # Mock current user
        mock_user.return_value = {"id": "test_user", "role": "doctor"}
        
        # Send chat request with existing conversation
        request_data = {
            "query": "Tell me more about treatment options",
            "conversation_id": "existing_conv_456",
            "patient_id": None,
            "metadata": {}
        }
        
        response = client.post(
            "/api/v1/chat/",
            json=request_data,
            headers=auth_headers
        )
        
        # Verify response uses same conversation ID
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == "existing_conv_456"
    
    def test_chat_endpoint_missing_query(self, client, auth_headers):
        """
        Test chat endpoint with missing query parameter.
        Verifies API properly validates required fields.
        """
        # Send request without query field
        request_data = {
            "conversation_id": None,
            "metadata": {}
        }
        
        response = client.post(
            "/api/v1/chat/",
            json=request_data,
            headers=auth_headers
        )
        
        # Should return validation error
        assert response.status_code == 422  # Unprocessable Entity
        
        # Error details should indicate missing field
        error_data = response.json()
        assert "detail" in error_data
    
    def test_chat_endpoint_empty_query(self, client, auth_headers):
        """
        Test chat endpoint with empty query string.
        Verifies API rejects empty messages.
        """
        # Send request with empty query
        request_data = {
            "query": "",
            "conversation_id": None,
            "metadata": {}
        }
        
        response = client.post(
            "/api/v1/chat/",
            json=request_data,
            headers=auth_headers
        )
        
        # Should return validation error for empty string
        assert response.status_code == 422
    
    @patch('backend.api.routes.chat.get_orchestrator')
    @patch('backend.api.routes.chat.get_current_user')
    def test_chat_endpoint_with_patient_context(self, mock_user, mock_orchestrator, client, auth_headers):
        """
        Test chat with patient context.
        Verifies API correctly handles patient-specific queries.
        """
        # Mock the orchestrator response
        mock_orchestrator_instance = AsyncMock()
        mock_orchestrator_instance.process_message.return_value = {
            "conversation_id": "conv_patient_789",
            "response": {"response": "Based on patient history..."},
            "agent": "risk_agent",
            "confidence": 0.88,
            "requires_human": False
        }
        mock_orchestrator.return_value = mock_orchestrator_instance
        
        # Mock current user
        mock_user.return_value = {"id": "test_user", "role": "doctor"}
        
        # Send request with patient ID
        request_data = {
            "query": "What is this patient's risk for complications?",
            "conversation_id": None,
            "patient_id": "P12345",
            "metadata": {"include_history": True}
        }
        
        response = client.post(
            "/api/v1/chat/",
            json=request_data,
            headers=auth_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["agent"] == "risk_agent"
    
    @patch('backend.api.routes.chat.submit_feedback')
    def test_feedback_endpoint(self, mock_feedback, client, auth_headers):
        """
        Test feedback submission endpoint.
        Verifies API correctly accepts user feedback for model improvement.
        """
        # Mock feedback submission
        mock_feedback.return_value = {"status": "success"}
        
        # Send feedback
        feedback_data = {
            "conversation_id": "conv_test_123",
            "rating": 5,
            "feedback_text": "Very helpful response"
        }
        
        response = client.post(
            "/api/v1/chat/feedback",
            params=feedback_data,
            headers=auth_headers
        )
        
        # Verify feedback accepted
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


class TestAuthenticationAPI:
    """
    Test suite for authentication and authorization.
    Tests token validation, role-based access, and security headers.
    """
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_unauthorized_access(self, client):
        """
        Test access without authentication token.
        Verifies API rejects unauthenticated requests.
        """
        # Send request without auth headers
        response = client.post(
            "/api/v1/chat/",
            json={"query": "test"}
        )
        
        # Should return unauthorized
        assert response.status_code == 401
    
    def test_invalid_token(self, client):
        """
        Test access with invalid authentication token.
        Verifies API validates token format and signature.
        """
        # Send request with invalid token
        headers = {"Authorization": "Bearer invalid_token"}
        
        response = client.post(
            "/api/v1/chat/",
            json={"query": "test"},
            headers=headers
        )
        
        # Should return unauthorized for invalid token
        assert response.status_code == 401
    
    @patch('backend.api.routes.chat.get_current_user')
    def test_rate_limiting(self, mock_user, client):
        """
        Test rate limiting functionality.
        Verifies API enforces rate limits to prevent abuse.
        """
        # Mock authenticated user
        mock_user.return_value = {"id": "test_user", "role": "doctor"}
        
        headers = {"Authorization": "Bearer valid_token"}
        
        # Make many rapid requests
        responses = []
        for i in range(150):  # Exceed typical rate limit
            response = client.post(
                "/api/v1/chat/",
                json={"query": f"test message {i}"},
                headers=headers
            )
            responses.append(response.status_code)
        
        # Some requests should be rate limited (429 status code)
        assert 429 in responses


class TestAdminAPI:
    """
    Test suite for administrative endpoints.
    Tests system monitoring, agent management, and configuration.
    """
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def admin_headers(self):
        """
        Create admin authentication headers.
        
        Returns:
            dict: Headers with admin privileges
        """
        return {"Authorization": "Bearer admin_token_123"}
    
    @patch('backend.api.routes.admin.get_orchestrator')
    def test_get_agent_status(self, mock_orchestrator, client, admin_headers):
        """
        Test agent status endpoint.
        Verifies admin can monitor agent health and performance.
        """
        # Mock orchestrator metrics
        mock_orchestrator_instance = Mock()
        mock_orchestrator_instance.get_metrics.return_value = {
            "total_conversations": 42,
            "avg_processing_time_ms": 125.5,
            "agent_stats": {
                "clinical_agent": {"status": "active", "current_task": None},
                "risk_agent": {"status": "active", "current_task": "task_123"}
            }
        }
        mock_orchestrator.return_value = mock_orchestrator_instance
        
        # Get agent status
        response = client.get(
            "/api/v1/admin/agents/status",
            headers=admin_headers
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "total_conversations" in data
        assert "agent_stats" in data
        assert len(data["agent_stats"]) > 0
    
    @patch('backend.api.routes.admin.get_orchestrator')
    def test_reset_conversation(self, mock_orchestrator, client, admin_headers):
        """
        Test conversation reset functionality.
        Verifies admin can clear conversation context when needed.
        """
        # Mock orchestrator context manager
        mock_orchestrator_instance = Mock()
        mock_context_manager = Mock()
        mock_context_manager.clear_context = Mock()
        mock_orchestrator_instance.context_manager = mock_context_manager
        mock_orchestrator.return_value = mock_orchestrator_instance
        
        # Reset conversation
        response = client.post(
            "/api/v1/admin/conversations/conv_123/reset",
            headers=admin_headers
        )
        
        # Verify reset successful
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "conversation_id" in data
        assert data["conversation_id"] == "conv_123"