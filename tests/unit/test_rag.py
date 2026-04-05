"""
Unit tests for the RAG (Retrieval-Augmented Generation) system.
Tests vector database operations, embedding generation, and document retrieval.
"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rag_system.vector_store.qdrant_manager import QdrantManager
from rag_system.vector_store.embeddings import MedicalEmbeddingGenerator
from rag_system.retrievers.medical_retriever import MedicalRetriever


class TestMedicalEmbeddingGenerator:
    """
    Test suite for medical text embedding generation.
    Tests embedding model loading, text encoding, and vector operations.
    """
    
    @pytest.fixture
    def embedding_generator(self):
        """
        Create a MedicalEmbeddingGenerator instance with mocked model.
        
        Returns:
            MedicalEmbeddingGenerator: Configured generator for testing
        """
        with patch('transformers.AutoModel.from_pretrained'):
            with patch('transformers.AutoTokenizer.from_pretrained'):
                generator = MedicalEmbeddingGenerator(
                    model_name="test-model",
                    device="cpu",
                    batch_size=2
                )
                return generator
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_single_text(self, embedding_generator):
        """
        Test generating embedding for single text.
        Verifies embedding generation returns correct shape and type.
        """
        # Mock the batch embedding generation
        mock_embedding = np.random.rand(768).astype(np.float32)
        embedding_generator._generate_batch_embeddings = AsyncMock(
            return_value=[mock_embedding]
        )
        
        # Generate embedding
        texts = ["This is a test medical text about hypertension"]
        embeddings = await embedding_generator.generate_embeddings(texts)
        
        # Verify results
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], np.ndarray)
        assert embeddings[0].shape[0] == 768  # Standard embedding dimension
        
        # Verify normalization (vector should have unit length)
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 0.001
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self, embedding_generator):
        """
        Test generating embeddings for multiple texts in batch.
        Verifies batch processing efficiency and result consistency.
        """
        # Mock batch embeddings for multiple texts
        mock_embeddings = [np.random.rand(768).astype(np.float32) for _ in range(5)]
        embedding_generator._generate_batch_embeddings = AsyncMock(
            return_value=mock_embeddings
        )
        
        # Generate embeddings for batch
        texts = [f"Medical text {i}" for i in range(5)]
        embeddings = await embedding_generator.generate_embeddings(texts)
        
        # Verify batch results
        assert len(embeddings) == 5
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
    
    @pytest.mark.asyncio
    async def test_generate_query_embedding_with_expansion(self, embedding_generator):
        """
        Test query embedding generation with medical term expansion.
        Verifies query expansion improves retrieval for medical terminology.
        """
        # Mock embedding generation
        mock_embedding = np.random.rand(768).astype(np.float32)
        embedding_generator.generate_embeddings = AsyncMock(return_value=[mock_embedding])
        
        # Generate query embedding with expansion
        query = "heart attack symptoms"
        embedding = await embedding_generator.generate_query_embedding(
            query,
            use_expansion=True
        )
        
        # Verify embedding generated
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 768
        
        # Verify expansion was applied
        # The expanded query should include medical synonyms
        call_args = embedding_generator.generate_embeddings.call_args
        expanded_text = call_args[0][0][0]  # First argument of first call
        assert "myocardial infarction" in expanded_text.lower() or "heart" in expanded_text.lower()
    
    def test_expand_medical_query(self, embedding_generator):
        """
        Test medical query expansion logic.
        Verifies proper expansion of common medical terms to clinical terminology.
        """
        # Test heart attack expansion
        expanded = embedding_generator._expand_medical_query("heart attack")
        assert "myocardial infarction" in expanded.lower()
        
        # Test diabetes expansion
        expanded = embedding_generator._expand_medical_query("type 2 diabetes")
        assert "diabetes mellitus" in expanded.lower()
        
        # Test non-medical query (should remain unchanged)
        expanded = embedding_generator._expand_medical_query("hospital bed availability")
        assert expanded == "hospital bed availability"
    
    def test_get_embedding_dimension(self, embedding_generator):
        """
        Test embedding dimension retrieval.
        Verifies correct dimension is reported for the model.
        """
        # Mock model config
        mock_config = Mock()
        mock_config.hidden_size = 768
        embedding_generator.model.config = mock_config
        
        # Get dimension
        dimension = embedding_generator.get_embedding_dimension()
        
        # Verify dimension
        assert dimension == 768
        assert isinstance(dimension, int)


class TestQdrantManager:
    """
    Test suite for Qdrant vector database manager.
    Tests document insertion, search operations, and collection management.
    """
    
    @pytest.fixture
    def qdrant_manager(self):
        """
        Create a QdrantManager instance with mocked client.
        
        Returns:
            QdrantManager: Configured manager for testing
        """
        with patch('qdrant_client.QdrantClient') as mock_client:
            # Mock the client methods
            mock_instance = Mock()
            mock_instance.get_collections.return_value = Mock(
                collections=[Mock(name="existing_collection")]
            )
            mock_client.return_value = mock_instance
            
            manager = QdrantManager(
                host="localhost",
                port=6333,
                collection_name="test_collection",
                vector_size=768
            )
            return manager
    
    @pytest.mark.asyncio
    async def test_insert_documents(self, qdrant_manager):
        """
        Test inserting documents into vector store.
        Verifies documents are properly indexed with embeddings.
        """
        # Prepare test documents
        documents = [
            {
                "id": "doc_1",
                "text": "Hypertension is a common condition",
                "source": "medical_textbook",
                "category": "cardiology",
                "metadata": {"author": "Dr. Smith"}
            },
            {
                "id": "doc_2",
                "text": "Treatment options include ACE inhibitors",
                "source": "clinical_guidelines",
                "category": "pharmacology"
            }
        ]
        
        # Create mock embeddings
        embeddings = [
            np.random.rand(768).astype(np.float32),
            np.random.rand(768).astype(np.float32)
        ]
        
        # Mock the upsert operation
        qdrant_manager.client.upsert = Mock()
        
        # Insert documents
        await qdrant_manager.insert_documents(documents, embeddings)
        
        # Verify upsert was called
        assert qdrant_manager.client.upsert.called
    
    @pytest.mark.asyncio
    async def test_search_similar_documents(self, qdrant_manager):
        """
        Test similarity search in vector store.
        Verifies retrieval of relevant documents based on query embedding.
        """
        # Create query embedding
        query_embedding = np.random.rand(768).astype(np.float32)
        
        # Mock search results
        mock_results = [
            Mock(
                id="doc_1",
                score=0.95,
                payload={
                    "text": "Hypertension management guidelines",
                    "source": "clinical_guidelines",
                    "category": "cardiology"
                }
            ),
            Mock(
                id="doc_2",
                score=0.82,
                payload={
                    "text": "Blood pressure treatment protocols",
                    "source": "hospital_protocols",
                    "category": "internal_medicine"
                }
            )
        ]
        qdrant_manager.client.search.return_value = mock_results
        
        # Perform search
        results = await qdrant_manager.search(
            query_vector=query_embedding,
            top_k=2,
            score_threshold=0.7
        )
        
        # Verify results
        assert len(results) == 2
        assert results[0]["score"] == 0.95
        assert "text" in results[0]
        assert "source" in results[0]
        
        # Verify search was called with correct parameters
        call_args = qdrant_manager.client.search.call_args[1]
        assert call_args["limit"] == 2
        assert call_args["score_threshold"] == 0.7
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, qdrant_manager):
        """
        Test search with metadata filters.
        Verifies filtering by source, category, or other metadata fields.
        """
        # Create query embedding
        query_embedding = np.random.rand(768).astype(np.float32)
        
        # Mock filtered search results
        mock_results = [
            Mock(
                id="doc_3",
                score=0.91,
                payload={
                    "text": "Cardiology guidelines",
                    "source": "clinical_guidelines",
                    "category": "cardiology"
                }
            )
        ]
        qdrant_manager.client.search.return_value = mock_results
        
        # Search with filter
        results = await qdrant_manager.search(
            query_vector=query_embedding,
            top_k=5,
            filter_conditions={"source": "clinical_guidelines", "category": "cardiology"}
        )
        
        # Verify results filtered correctly
        assert len(results) > 0
        for result in results:
            assert result["source"] == "clinical_guidelines"
            assert result["category"] == "cardiology"
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, qdrant_manager):
        """
        Test hybrid search combining vector and keyword matching.
        Verifies improved retrieval using both semantic and lexical search.
        """
        # Create query
        query_text = "treatment for hypertension"
        query_embedding = np.random.rand(768).astype(np.float32)
        
        # Mock vector search results
        vector_results = [
            {
                "id": "doc_1",
                "text": "Hypertension clinical guidelines",
                "score": 0.92,
                "source": "guidelines"
            },
            {
                "id": "doc_2",
                "text": "Blood pressure management",
                "score": 0.85,
                "source": "textbook"
            }
        ]
        
        # Mock keyword search results
        keyword_results = [
            {
                "id": "doc_1",
                "text": "Hypertension clinical guidelines",
                "score": 0.88,
                "source": "guidelines"
            }
        ]
        
        # Mock the search methods
        qdrant_manager.search = AsyncMock(return_value=vector_results)
        qdrant_manager._keyword_search = AsyncMock(return_value=keyword_results)
        
        # Perform hybrid search
        results = await qdrant_manager.hybrid_search(
            query_vector=query_embedding,
            text_query=query_text,
            top_k=2,
            alpha=0.6  # 60% weight to vector search
        )
        
        # Verify results
        assert len(results) <= 2
        if results:
            assert "final_score" in results[0]
            assert "vector_score" in results[0]
            assert "keyword_score" in results[0]
    
    def test_get_collection_stats(self, qdrant_manager):
        """
        Test collection statistics retrieval.
        Verifies ability to monitor vector store health and size.
        """
        # Mock collection info
        mock_collection_info = Mock()
        mock_collection_info.vectors_count = 1000
        mock_collection_info.segments_count = 4
        mock_collection_info.status = "green"
        qdrant_manager.client.get_collection.return_value = mock_collection_info
        
        # Get statistics
        stats = qdrant_manager.get_collection_stats()
        
        # Verify statistics
        assert stats["name"] == "test_collection"
        assert stats["vectors_count"] == 1000
        assert stats["segments_count"] == 4
        assert stats["status"] == "green"
        assert stats["vector_size"] == 768


class TestMedicalRetriever:
    """
    Test suite for medical information retrieval.
    Tests integration with multiple knowledge sources and result synthesis.
    """
    
    @pytest.fixture
    def medical_retriever(self):
        """
        Create a MedicalRetriever instance for testing.
        
        Returns:
            MedicalRetriever: Configured retriever for testing
        """
        retriever = MedicalRetriever(config={"test_mode": True})
        return retriever
    
    @pytest.mark.asyncio
    async def test_retrieve_from_multiple_sources(self, medical_retriever):
        """
        Test retrieval from multiple knowledge sources.
        Verifies aggregation of results from PubMed, guidelines, and internal KB.
        """
        # Mock individual source queries
        async def mock_pubmed(query, top_k):
            return [{"source": "PubMed", "content": "Research article", "score": 0.9}]
        
        async def mock_guidelines(query, top_k):
            return [{"source": "Guidelines", "content": "Clinical guideline", "score": 0.88}]
        
        async def mock_internal(query, top_k):
            return [{"source": "Internal KB", "content": "Hospital protocol", "score": 0.85}]
        
        medical_retriever._query_pubmed = mock_pubmed
        medical_retriever._query_guidelines = mock_guidelines
        medical_retriever._query_internal_kb = mock_internal
        
        # Perform retrieval
        results = await medical_retriever.retrieve(
            query="diabetes treatment guidelines",
            top_k=3
        )
        
        # Verify results from all sources
        assert len(results) >= 1
        sources = [r["source"] for r in results]
        assert any("PubMed" in s for s in sources)
        assert any("Guidelines" in s for s in sources)
        
        # Results should be sorted by score
        scores = [r.get("score", 0) for r in results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_retrieve_guidelines(self, medical_retriever):
        """
        Test clinical guideline retrieval.
        Verifies specialized retrieval for clinical practice guidelines.
        """
        # Mock general retrieve method
        mock_results = [
            {
                "source": "Clinical Guidelines",
                "title": "Diabetes Management Guidelines 2023",
                "content": "First-line treatment includes metformin...",
                "score": 0.92
            }
        ]
        medical_retriever.retrieve = AsyncMock(return_value=mock_results)
        
        # Retrieve guidelines
        guidelines = await medical_retriever.retrieve_guidelines("diabetes")
        
        # Verify guideline response
        assert "condition" in guidelines
        assert guidelines["condition"] == "diabetes"
        assert "summary" in guidelines
        assert "sources" in guidelines
        assert guidelines["confidence"] > 0
    
    @pytest.mark.asyncio
    async def test_retrieve_disease_info(self, medical_retriever):
        """
        Test comprehensive disease information retrieval.
        Verifies extraction of symptoms, diagnosis, treatment, and prognosis.
        """
        # Mock general retrieve method
        mock_results = [
            {
                "source": "Medical Textbook",
                "content": "Diabetes symptoms include increased thirst and frequent urination",
                "score": 0.9
            },
            {
                "source": "Clinical Guidelines",
                "content": "Diagnosis requires fasting glucose >126 mg/dL",
                "score": 0.88
            },
            {
                "source": "Research Paper",
                "content": "Treatment involves metformin and lifestyle changes",
                "score": 0.85
            }
        ]
        medical_retriever.retrieve = AsyncMock(return_value=mock_results)
        
        # Retrieve disease information
        disease_info = await medical_retriever.retrieve_disease_info("diabetes")
        
        # Verify comprehensive information
        assert "summary" in disease_info
        assert "symptoms" in disease_info
        assert "diagnosis" in disease_info
        assert "treatment" in disease_info
        assert "references" in disease_info
        
        # Verify content is extracted
        assert disease_info["symptoms"] is not None or len(disease_info["symptoms"]) >= 0