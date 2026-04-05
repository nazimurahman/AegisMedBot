"""
Unit tests for the agent system components.
Tests each agent's functionality, message handling, and response generation.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import json

# Import agent classes to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.base_agent import BaseAgent, AgentMessage, AgentResponse, AgentStatus
from agents.clinical_agent.clinical_agent import ClinicalAgent
from agents.risk_agent.risk_predictor import RiskPredictorAgent
from agents.operations_agent.operations_agent import OperationsAgent
from agents.orchestrator.agent_orchestrator import AgentOrchestrator


class TestBaseAgent:
    """
    Test suite for the base agent abstract class.
    Verifies core agent functionality like message handling and status management.
    """
    
    @pytest.fixture
    def base_agent(self):
        """
        Create a concrete implementation of BaseAgent for testing.
        This fixture provides a testable agent instance.
        
        Returns:
            ConcreteAgent: A test implementation of the abstract base agent
        """
        class ConcreteAgent(BaseAgent):
            """Concrete implementation for testing abstract base class."""
            
            async def process(self, message: AgentMessage) -> AgentResponse:
                """Simple test implementation of process method."""
                return AgentResponse(
                    message_id=message.message_id,
                    content={"test": "response"},
                    confidence=0.9,
                    processing_time_ms=10.0
                )
        
        return ConcreteAgent(
            name="test_agent",
            role="tester",
            description="Test agent for unit testing",
            config={"test": True}
        )
    
    def test_agent_initialization(self, base_agent):
        """
        Test that agent initializes with correct attributes.
        Verifies name, role, description, and status are set properly.
        """
        # Check basic properties
        assert base_agent.name == "test_agent"
        assert base_agent.role == "tester"
        assert base_agent.description == "Test agent for unit testing"
        assert base_agent.status == AgentStatus.IDLE
        assert base_agent.current_task is None
        assert base_agent.config == {"test": True}
    
    def test_update_status(self, base_agent):
        """
        Test status updates for agent lifecycle.
        Verifies agent can transition through different processing states.
        """
        # Test status change to processing
        base_agent.update_status(AgentStatus.PROCESSING, task_id="task_123")
        assert base_agent.status == AgentStatus.PROCESSING
        assert base_agent.current_task == "task_123"
        
        # Test status change to completed
        base_agent.update_status(AgentStatus.COMPLETED)
        assert base_agent.status == AgentStatus.COMPLETED
        
        # Test status change to error
        base_agent.update_status(AgentStatus.ERROR)
        assert base_agent.status == AgentStatus.ERROR
    
    def test_validate_input_valid(self, base_agent):
        """
        Test message validation with valid input.
        Verifies that properly formatted messages pass validation.
        """
        # Create a valid agent message
        valid_message = AgentMessage(
            conversation_id="conv_123",
            sender="user",
            recipient="test_agent",
            message_type="request",
            content={"query": "test query"}
        )
        
        # Should return True for valid message
        assert base_agent.validate_input(valid_message) is True
    
    def test_validate_input_invalid(self, base_agent):
        """
        Test message validation with invalid input.
        Verifies that malformed messages are rejected.
        """
        # Create an invalid message object (missing required fields)
        # This uses a mock object that doesn't have required attributes
        invalid_message = Mock()
        invalid_message.conversation_id = "conv_123"
        # Missing content attribute
        
        # Should return False for invalid message
        assert base_agent.validate_input(invalid_message) is False
    
    @pytest.mark.asyncio
    async def test_escalate_to_human(self, base_agent):
        """
        Test human escalation functionality.
        Verifies that agent properly escalates tasks requiring human intervention.
        """
        # Create a test message
        message = AgentMessage(
            conversation_id="conv_123",
            sender="user",
            recipient="test_agent",
            message_type="request",
            content={"query": "critical medical decision"}
        )
        
        # Escalate the message
        response = await base_agent.escalate_to_human(
            message=message,
            reason="Low confidence in automated decision"
        )
        
        # Verify escalation response
        assert response.requires_human_confirmation is True
        assert response.confidence == 0.0
        assert response.content["status"] == "escalated"
        assert response.content["reason"] == "Low confidence in automated decision"
        assert response.content["human_intervention_required"] is True
        
        # Verify agent status updated
        assert base_agent.status == AgentStatus.WAITING_FOR_HUMAN
    
    @pytest.mark.asyncio
    async def test_can_handle_default(self, base_agent):
        """
        Test default can_handle implementation.
        Verifies base implementation returns neutral confidence score.
        """
        # Default implementation should return 0.5 for any task
        confidence = await base_agent.can_handle(
            task_type="any_task",
            context={"test": "context"}
        )
        
        # Default confidence should be 0.5
        assert confidence == 0.5


class TestClinicalAgent:
    """
    Test suite for the Clinical Knowledge Agent.
    Tests medical information retrieval and clinical decision support.
    """
    
    @pytest.fixture
    def clinical_agent(self):
        """
        Create a ClinicalAgent instance for testing.
        Uses mocked dependencies to avoid external API calls.
        
        Returns:
            ClinicalAgent: Configured clinical agent for testing
        """
        with patch('agents.clinical_agent.tools.medical_retriever.MedicalRetriever'):
            with patch('agents.clinical_agent.tools.drug_interaction.DrugInteractionChecker'):
                agent = ClinicalAgent(config={"test_mode": True})
                return agent
    
    @pytest.mark.asyncio
    async def test_process_drug_interaction_query(self, clinical_agent):
        """
        Test handling of drug interaction queries.
        Verifies agent correctly identifies and processes medication interaction requests.
        """
        # Create message asking about drug interaction
        message = AgentMessage(
            conversation_id="conv_123",
            sender="user",
            recipient="clinical_agent",
            message_type="request",
            content={
                "query": "Does aspirin interact with ibuprofen?"
            }
        )
        
        # Mock the drug checker response
        mock_interaction = {
            "interactions": [
                {
                    "drugs": "aspirin and ibuprofen",
                    "severity": "moderate",
                    "description": "Increased risk of gastrointestinal bleeding"
                }
            ],
            "confidence": 0.85
        }
        clinical_agent.drug_checker.check_interactions = AsyncMock(return_value=mock_interaction)
        
        # Mock the medical retriever response
        mock_literature = [{
            "source": "PubMed",
            "title": "NSAID interactions study",
            "content": "Research findings...",
            "score": 0.9
        }]
        clinical_agent.medical_retriever.retrieve = AsyncMock(return_value=mock_literature)
        
        # Process the message
        response = await clinical_agent.process(message)
        
        # Verify response structure
        assert response.message_id == message.message_id
        assert response.confidence > 0
        assert "response" in response.content
        assert "medications_checked" in response.content
        assert "interactions_found" in response.content
        
        # Verify the response contains interaction information
        response_text = response.content["response"]
        assert "Drug Interaction Analysis" in response_text or "interaction" in response_text.lower()
    
    @pytest.mark.asyncio
    async def test_process_guideline_query(self, clinical_agent):
        """
        Test handling of clinical guideline queries.
        Verifies agent retrieves and formats clinical guidelines correctly.
        """
        # Create message asking for clinical guidelines
        message = AgentMessage(
            conversation_id="conv_123",
            sender="user",
            recipient="clinical_agent",
            message_type="request",
            content={
                "query": "What are the guidelines for hypertension treatment?"
            }
        )
        
        # Mock the guideline retrieval
        mock_guidelines = {
            "summary": "First-line treatment includes ACE inhibitors or ARBs",
            "confidence": 0.9,
            "sources": ["ACC/AHA Guidelines 2023"]
        }
        clinical_agent.medical_retriever.retrieve_guidelines = AsyncMock(return_value=mock_guidelines)
        
        # Process the message
        response = await clinical_agent.process(message)
        
        # Verify response
        assert response.confidence > 0
        assert "response" in response.content
        assert "condition" in response.content or "guidelines" in response.content
        assert response.requires_human_confirmation is False
    
    @pytest.mark.asyncio
    async def test_process_disease_info_query(self, clinical_agent):
        """
        Test handling of disease information queries.
        Verifies agent provides comprehensive disease information including symptoms and treatment.
        """
        # Create message asking about disease
        message = AgentMessage(
            conversation_id="conv_123",
            sender="user",
            recipient="clinical_agent",
            message_type="request",
            content={
                "query": "Tell me about diabetes mellitus"
            }
        )
        
        # Mock disease info retrieval
        mock_disease_info = {
            "summary": "Diabetes is a metabolic disorder characterized by high blood sugar",
            "symptoms": ["increased thirst", "frequent urination"],
            "diagnosis": ["fasting blood glucose", "HbA1c"],
            "treatment": ["insulin", "metformin"],
            "references": ["PubMed article 12345"],
            "confidence": 0.9
        }
        clinical_agent.medical_retriever.retrieve_disease_info = AsyncMock(return_value=mock_disease_info)
        
        # Process the message
        response = await clinical_agent.process(message)
        
        # Verify comprehensive disease information
        assert response.content["disease"] == "diabetes" or "diabetes" in response.content["response"].lower()
        assert "symptoms" in response.content
        assert "treatment" in response.content
        assert response.confidence > 0.8
    
    def test_classify_query(self, clinical_agent):
        """
        Test query classification logic.
        Verifies agent correctly categorizes different types of medical queries.
        """
        # Test drug interaction classification
        query_type = clinical_agent._classify_query("What drugs interact with warfarin?")
        assert query_type == "drug_interaction"
        
        # Test guideline classification
        query_type = clinical_agent._classify_query("What are the clinical guidelines for pneumonia?")
        assert query_type == "guideline"
        
        # Test disease info classification
        query_type = clinical_agent._classify_query("What are the symptoms of heart failure?")
        assert query_type == "disease_info"
        
        # Test general classification
        query_type = clinical_agent._classify_query("Tell me about medical research")
        assert query_type == "general"
    
    def test_extract_medications(self, clinical_agent):
        """
        Test medication name extraction from text.
        Verifies agent can identify drug names in natural language queries.
        """
        query = "I'm taking aspirin and lisinopril, any interactions?"
        medications = clinical_agent._extract_medications(query)
        
        # Should extract aspirin and lisinopril
        assert "aspirin" in medications
        assert "lisinopril" in medications
        
        # Test with no medications
        medications = clinical_agent._extract_medications("What is the best treatment?")
        assert len(medications) == 0
    
    def test_extract_condition(self, clinical_agent):
        """
        Test medical condition extraction from text.
        Verifies agent identifies medical conditions in user queries.
        """
        query = "What are the treatments for hypertension and diabetes?"
        conditions = clinical_agent._extract_condition(query)
        
        # Should extract at least one condition
        assert conditions is not None
        
        # Test with no condition
        conditions = clinical_agent._extract_condition("Tell me about hospital operations")
        assert conditions is None


class TestRiskPredictorAgent:
    """
    Test suite for the Risk Prediction Agent.
    Tests patient risk assessment and complication prediction functionality.
    """
    
    @pytest.fixture
    def risk_agent(self):
        """
        Create a RiskPredictorAgent instance for testing.
        
        Returns:
            RiskPredictorAgent: Configured risk agent for testing
        """
        with patch('ml_training.models.lstm_predictor.LSTMPredictor'):
            agent = RiskPredictorAgent(config={"test_mode": True})
            return agent
    
    @pytest.mark.asyncio
    async def test_predict_icu_risk(self, risk_agent):
        """
        Test ICU admission risk prediction.
        Verifies agent correctly assesses patient risk for ICU admission.
        """
        # Create patient data for risk assessment
        patient_data = {
            "age": 75,
            "heart_rate": 110,
            "blood_pressure_systolic": 85,
            "respiratory_rate": 28,
            "oxygen_saturation": 88,
            "temperature": 38.5,
            "chronic_conditions": ["COPD", "Heart Failure"]
        }
        
        message = AgentMessage(
            conversation_id="conv_123",
            sender="user",
            recipient="risk_agent",
            message_type="request",
            content={
                "query": "Predict ICU risk for this patient",
                "patient_data": patient_data
            }
        )
        
        # Mock prediction result
        mock_prediction = {
            "icu_risk": 0.85,
            "mortality_risk": 0.42,
            "complication_risk": 0.73,
            "risk_factors": ["advanced age", "hypotension", "hypoxia"],
            "confidence": 0.9
        }
        risk_agent.predict = AsyncMock(return_value=mock_prediction)
        
        # Process the message
        response = await risk_agent.process(message)
        
        # Verify prediction response
        assert "icu_risk" in response.content
        assert response.content["icu_risk"] == 0.85
        assert "risk_factors" in response.content
        assert response.confidence > 0
    
    @pytest.mark.asyncio
    async def test_missing_patient_data(self, risk_agent):
        """
        Test handling of missing patient data.
        Verifies agent gracefully handles incomplete information.
        """
        # Create message with incomplete data
        message = AgentMessage(
            conversation_id="conv_123",
            sender="user",
            recipient="risk_agent",
            message_type="request",
            content={
                "query": "Predict risk",
                "patient_data": {"age": 65}  # Missing vital signs
            }
        )
        
        # Process should handle missing data without crashing
        response = await risk_agent.process(message)
        
        # Should still provide some response
        assert response is not None
        # Confidence should be lower due to missing data
        assert response.confidence < 0.8


class TestOperationsAgent:
    """
    Test suite for the Hospital Operations Agent.
    Tests bed occupancy analysis, patient flow, and resource allocation.
    """
    
    @pytest.fixture
    def operations_agent(self):
        """
        Create an OperationsAgent instance for testing.
        
        Returns:
            OperationsAgent: Configured operations agent for testing
        """
        agent = OperationsAgent(config={"test_mode": True})
        return agent
    
    @pytest.mark.asyncio
    async def test_bed_occupancy_query(self, operations_agent):
        """
        Test bed occupancy information retrieval.
        Verifies agent can provide current bed availability status.
        """
        message = AgentMessage(
            conversation_id="conv_123",
            sender="user",
            recipient="operations_agent",
            message_type="request",
            content={
                "query": "What is the current ICU bed occupancy?"
            }
        )
        
        # Mock bed occupancy data
        mock_occupancy = {
            "total_beds": 20,
            "occupied_beds": 15,
            "available_beds": 5,
            "occupancy_rate": 0.75,
            "department": "ICU",
            "timestamp": datetime.now().isoformat()
        }
        operations_agent.get_bed_occupancy = AsyncMock(return_value=mock_occupancy)
        
        # Process the message
        response = await operations_agent.process(message)
        
        # Verify occupancy information
        assert "response" in response.content
        assert "bed" in response.content["response"].lower() or "occupancy" in response.content["response"].lower()
        assert response.confidence > 0
    
    @pytest.mark.asyncio
    async def test_patient_flow_prediction(self, operations_agent):
        """
        Test patient flow prediction functionality.
        Verifies agent can predict ER wait times and patient movement.
        """
        message = AgentMessage(
            conversation_id="conv_123",
            sender="user",
            recipient="operations_agent",
            message_type="request",
            content={
                "query": "Predict patient flow for the next 24 hours"
            }
        )
        
        # Mock flow prediction
        mock_flow = {
            "predicted_admissions": 12,
            "peak_hours": ["14:00", "18:00"],
            "expected_wait_time": 45,
            "recommended_staffing": "increase evening shift",
            "confidence": 0.82
        }
        operations_agent.predict_patient_flow = AsyncMock(return_value=mock_flow)
        
        # Process the message
        response = await operations_agent.process(message)
        
        # Verify prediction results
        assert response is not None
        assert "response" in response.content
        assert response.confidence > 0


class TestAgentOrchestrator:
    """
    Test suite for the Agent Orchestrator.
    Tests agent coordination, task delegation, and conversation management.
    """
    
    @pytest.fixture
    def orchestrator(self):
        """
        Create an AgentOrchestrator instance with registered test agents.
        
        Returns:
            AgentOrchestrator: Configured orchestrator for testing
        """
        orchestrator = AgentOrchestrator(config={"human_threshold": 0.7})
        
        # Register mock agents
        mock_agent = Mock(spec=BaseAgent)
        mock_agent.name = "test_agent"
        mock_agent.role = "tester"
        mock_agent.status = AgentStatus.IDLE
        
        # Create async mock for process method
        async def mock_process(message):
            return AgentResponse(
                message_id=message.message_id,
                content={"response": "test response"},
                confidence=0.9,
                processing_time_ms=5.0
            )
        mock_agent.process = mock_process
        
        orchestrator.register_agent(mock_agent)
        
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_process_message_success(self, orchestrator):
        """
        Test successful message processing through orchestrator.
        Verifies end-to-end message handling with agent delegation.
        """
        # Send a test message
        result = await orchestrator.process_message(
            message={"query": "test query"},
            conversation_id=None
        )
        
        # Verify result structure
        assert "conversation_id" in result
        assert "response" in result
        assert "agent" in result
        assert "confidence" in result
        assert "processing_time_ms" in result
        
        # Verify response has expected fields
        assert result["response"] is not None
        assert result["confidence"] >= 0
    
    @pytest.mark.asyncio
    async def test_conversation_context_persistence(self, orchestrator):
        """
        Test conversation context persistence across multiple messages.
        Verifies orchestrator maintains context between interactions.
        """
        # First message
        result1 = await orchestrator.process_message(
            message={"query": "first message"},
            conversation_id=None
        )
        
        conversation_id = result1["conversation_id"]
        
        # Second message with same conversation ID
        result2 = await orchestrator.process_message(
            message={"query": "second message"},
            conversation_id=conversation_id
        )
        
        # Should maintain same conversation ID
        assert result2["conversation_id"] == conversation_id
        
        # Context should be retrievable
        context = orchestrator.context_manager.get_context(conversation_id)
        assert context is not None
    
    def test_get_metrics(self, orchestrator):
        """
        Test metrics collection functionality.
        Verifies orchestrator tracks performance metrics correctly.
        """
        # Get metrics before any processing
        metrics = orchestrator.get_metrics()
        
        # Should have metrics structure
        assert "total_conversations" in metrics
        assert "avg_processing_time_ms" in metrics
        assert "agent_stats" in metrics
        
        # Metrics should be numbers
        assert isinstance(metrics["total_conversations"], int)
        assert isinstance(metrics["avg_processing_time_ms"], float)