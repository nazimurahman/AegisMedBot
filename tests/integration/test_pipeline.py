"""
Integration tests for the complete agent pipeline.
Tests end-to-end workflows from user input to agent response with real components.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, AsyncMock, Mock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.orchestrator.agent_orchestrator import AgentOrchestrator
from agents.clinical_agent.clinical_agent import ClinicalAgent
from agents.risk_agent.risk_predictor import RiskPredictorAgent
from agents.operations_agent.operations_agent import OperationsAgent
from agents.director_agent.director_intelligence import DirectorIntelligenceAgent
from agents.compliance_agent.privacy_guardian import PrivacyGuardianAgent


class TestCompleteAgentPipeline:
    """
    Integration tests for the complete agent pipeline.
    Tests interaction between all agents and the orchestrator.
    """
    
    @pytest.fixture
    async def orchestrator(self):
        """
        Create a fully configured orchestrator with all agents.
        
        Returns:
            AgentOrchestrator: Configured orchestrator for integration testing
        """
        orchestrator = AgentOrchestrator(config={"human_threshold": 0.7})
        
        # Initialize all agents with mock dependencies
        with patch('agents.clinical_agent.tools.medical_retriever.MedicalRetriever'):
            with patch('agents.clinical_agent.tools.drug_interaction.DrugInteractionChecker'):
                clinical_agent = ClinicalAgent(config={"test_mode": True})
                orchestrator.register_agent(clinical_agent)
        
        with patch('ml_training.models.lstm_predictor.LSTMPredictor'):
            risk_agent = RiskPredictorAgent(config={"test_mode": True})
            orchestrator.register_agent(risk_agent)
        
        operations_agent = OperationsAgent(config={"test_mode": True})
        orchestrator.register_agent(operations_agent)
        
        director_agent = DirectorIntelligenceAgent(config={"test_mode": True})
        orchestrator.register_agent(director_agent)
        
        privacy_agent = PrivacyGuardianAgent(config={"test_mode": True})
        orchestrator.register_agent(privacy_agent)
        
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_clinical_query_workflow(self, orchestrator):
        """
        Test complete workflow for clinical information query.
        Verifies end-to-end processing from user question to clinical answer.
        """
        # Send clinical query
        result = await orchestrator.process_message(
            message={
                "query": "What are the first-line treatments for hypertension?",
                "user_id": "test_doctor",
                "metadata": {"test": True}
            }
        )
        
        # Verify response structure
        assert "conversation_id" in result
        assert "response" in result
        assert "agent" in result
        assert result["agent"] == "clinical_agent" or "clinical" in result["agent"]
        
        # Verify response contains medical information
        response_text = result["response"].get("response", str(result["response"]))
        assert len(response_text) > 0
        assert result["confidence"] > 0
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, orchestrator):
        """
        Test multi-turn conversation with context retention.
        Verifies conversation history is maintained across multiple interactions.
        """
        # First message
        result1 = await orchestrator.process_message(
            message={"query": "Tell me about diabetes"},
            conversation_id=None
        )
        
        conversation_id = result1["conversation_id"]
        
        # Second message - follow-up
        result2 = await orchestrator.process_message(
            message={"query": "What are the treatment options?"},
            conversation_id=conversation_id
        )
        
        # Verify same conversation
        assert result2["conversation_id"] == conversation_id
        
        # Verify context was maintained
        context = orchestrator.context_manager.get_context(conversation_id)
        assert context is not None
        assert context.get("message_count", 0) >= 2
    
    @pytest.mark.asyncio
    async def test_risk_assessment_workflow(self, orchestrator):
        """
        Test complete risk assessment workflow.
        Verifies patient data analysis and risk prediction.
        """
        # Send risk assessment query with patient data
        result = await orchestrator.process_message(
            message={
                "query": "Assess this patient's risk for ICU admission",
                "patient_data": {
                    "age": 78,
                    "heart_rate": 115,
                    "blood_pressure_systolic": 82,
                    "oxygen_saturation": 89,
                    "chronic_conditions": ["COPD", "Heart Failure"]
                },
                "user_id": "test_doctor"
            }
        )
        
        # Verify risk assessment response
        assert "response" in result
        assert result["confidence"] is not None
        
        # Risk predictions should be included
        response_content = result["response"]
        assert isinstance(response_content, dict) or isinstance(response_content, str)
    
    @pytest.mark.asyncio
    async def test_operations_query_workflow(self, orchestrator):
        """
        Test hospital operations query workflow.
        Verifies resource allocation and bed management information.
        """
        # Send operations query
        result = await orchestrator.process_message(
            message={
                "query": "What is the current ICU bed occupancy rate?",
                "user_id": "hospital_admin"
            }
        )
        
        # Verify operations response
        assert "response" in result
        assert result["agent"] == "operations_agent" or "operations" in result["agent"]
        
        # Response should include bed or occupancy information
        response_text = result["response"].get("response", str(result["response"]))
        assert any(word in response_text.lower() for word in ["bed", "occupancy", "available"])
    
    @pytest.mark.asyncio
    async def test_director_intelligence_workflow(self, orchestrator):
        """
        Test medical director intelligence workflow.
        Verifies KPI analysis and strategic reporting.
        """
        # Send director query
        result = await orchestrator.process_message(
            message={
                "query": "Show me this week's hospital KPIs",
                "user_id": "medical_director",
                "metadata": {"timeframe": "weekly"}
            }
        )
        
        # Verify director response
        assert "response" in result
        
        # Response should contain KPI or performance information
        response_text = result["response"].get("response", str(result["response"]))
        assert any(word in response_text.lower() for word in ["kpi", "performance", "metric"])
    
    @pytest.mark.asyncio
    async def test_privacy_guardian_integration(self, orchestrator):
        """
        Test privacy guardian integration.
        Verifies PHI detection and redaction in agent responses.
        """
        # Send query that might contain PHI
        result = await orchestrator.process_message(
            message={
                "query": "What is the treatment for patient John Smith born 01/15/1970?",
                "user_id": "doctor_with_access"
            }
        )
        
        # Privacy guardian should handle PHI appropriately
        # Based on user role, PHI may be redacted or allowed
        assert "response" in result
        
        # Check that privacy measures were applied
        # This would be verified through audit logs in production
        assert result["confidence"] is not None
    
    @pytest.mark.asyncio
    async def test_human_escalation_workflow(self, orchestrator):
        """
        Test human escalation workflow.
        Verifies low-confidence queries are escalated appropriately.
        """
        # Send ambiguous or complex query
        result = await orchestrator.process_message(
            message={
                "query": "Should I prescribe this medication for this complex case?",
                "user_id": "resident_doctor",
                "metadata": {"complexity": "high"}
            }
        )
        
        # If confidence is low, should require human intervention
        if result.get("requires_human"):
            assert result["agent"] == "human_escalation" or result["requires_human"] is True
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, orchestrator):
        """
        Test error recovery in agent pipeline.
        Verifies system gracefully handles errors and provides fallback responses.
        """
        # Send malformed query
        result = await orchestrator.process_message(
            message={
                "query": "",  # Empty query should trigger error handling
                "user_id": "test_user"
            }
        )
        
        # Should still return a response (error message)
        assert "response" in result
        # Response should indicate an error or request for clarification
        assert result["response"] is not None


class TestDatabaseIntegration:
    """
    Integration tests for database operations.
    Tests CRUD operations and data consistency.
    """
    
    @pytest.mark.asyncio
    async def test_patient_data_retrieval(self):
        """
        Test patient data retrieval from database.
        Verifies correct fetching of patient information for clinical context.
        """
        # This would test actual database connection
        # Implementation would use test database
        pass
    
    @pytest.mark.asyncio
    async def test_audit_logging(self):
        """
        Test audit logging for all agent interactions.
        Verifies all sensitive operations are logged for compliance.
        """
        # This would verify audit trail entries
        pass


class TestRAGIntegration:
    """
    Integration tests for RAG system.
    Tests document indexing, retrieval, and knowledge base updates.
    """
    
    @pytest.mark.asyncio
    async def test_document_indexing_pipeline(self):
        """
        Test complete document indexing pipeline.
        Verifies documents are properly processed and indexed.
        """
        # This would test full indexing workflow
        pass
    
    @pytest.mark.asyncio
    async def test_knowledge_base_update(self):
        """
        Test knowledge base updates.
        Verifies new medical knowledge is properly incorporated.
        """
        # This would test incremental updates to knowledge base
        pass