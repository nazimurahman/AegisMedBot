"""
Document Indexer Module

This module handles the ingestion, processing, and indexing of documents into
the vector database. It supports multiple document formats, chunking strategies,
and batch processing for efficient indexing of large document collections.

The indexer is responsible for:
- Document loading from various sources
- Text preprocessing and cleaning
- Intelligent chunking with overlap
- Embedding generation
- Vector storage insertion
- Metadata extraction and management
"""

import logging
import hashlib
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Iterator
from pathlib import Path
from datetime import datetime
import numpy as np

# Import from local modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_store.schema import (
    VectorDocument, DocumentMetadata, VectorStoreConfig,
    DocumentCategory, DocumentSource
)
from vector_store.qdrant_manager import QdrantManager
from vector_store.embeddings import MedicalEmbeddingGenerator

# Configure logging
logger = logging.getLogger(__name__)


class ChunkingStrategy:
    """
    Enumeration of chunking strategies for document splitting.
    
    Different strategies optimize for different document types and use cases:
    - FIXED_SIZE: Simple fixed-size chunks with overlap
    - SEMANTIC: Split at semantic boundaries (paragraphs, sections)
    - HIERARCHICAL: Multi-level chunking for hierarchical retrieval
    - SENTENCE: Split at sentence boundaries
    - PARAGRAPH: Split at paragraph boundaries
    """
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


class DocumentIndexer:
    """
    Main document indexer for vector database population.
    
    This class handles the complete document indexing pipeline:
    1. Document loading and validation
    2. Text preprocessing and cleaning
    3. Intelligent chunking
    4. Embedding generation
    5. Vector store insertion
    
    Attributes:
        vector_store: QdrantManager for vector operations
        embedding_generator: MedicalEmbeddingGenerator for embeddings
        chunk_size: Default chunk size in tokens/characters
        chunk_overlap: Overlap between consecutive chunks
        chunking_strategy: Strategy for splitting documents
        batch_size: Batch size for embedding generation
        config: Configuration dictionary
    """
    
    def __init__(
        self,
        vector_store: QdrantManager,
        embedding_generator: MedicalEmbeddingGenerator,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the document indexer.
        
        Args:
            vector_store: Initialized QdrantManager instance
            embedding_generator: MedicalEmbeddingGenerator instance
            config: Configuration dictionary with indexing parameters
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.config = config or {}
        
        # Chunking parameters
        self.chunk_size = self.config.get('chunk_size', 512)
        self.chunk_overlap = self.config.get('chunk_overlap', 50)
        self.chunking_strategy = self.config.get(
            'chunking_strategy',
            ChunkingStrategy.SEMANTIC
        )
        self.batch_size = self.config.get('batch_size', 100)
        
        # Processing flags
        self.clean_text = self.config.get('clean_text', True)
        self.remove_stopwords = self.config.get('remove_stopwords', False)
        
        logger.info(f"DocumentIndexer initialized with chunking strategy: {self.chunking_strategy}")
    
    async def index_document(
        self,
        document: VectorDocument,
        generate_embeddings: bool = True
    ) -> List[str]:
        """
        Index a single document into the vector store.
        
        Args:
            document: VectorDocument to index
            generate_embeddings: Whether to generate embeddings
            
        Returns:
            List of document IDs for chunks (or single ID for no chunking)
        """
        try:
            # Clean text if configured
            if self.clean_text:
                document.text = self._clean_text(document.text)
            
            # Split into chunks if needed
            chunks = self._chunk_document(document)
            
            if not chunks:
                logger.warning(f"No chunks generated for document {document.id}")
                return []
            
            # Generate embeddings if requested
            if generate_embeddings:
                texts = [chunk.text for chunk in chunks]
                embeddings = await self.embedding_generator.generate_embeddings(texts)
                
                # Assign embeddings to chunks
                for chunk, embedding in zip(chunks, embeddings):
                    chunk.embedding = embedding.tolist()
            
            # Insert chunks into vector store
            chunk_dicts = [chunk.to_dict() for chunk in chunks]
            embeddings_list = [
                np.array(chunk.embedding) for chunk in chunks if chunk.embedding is not None
            ]
            
            if embeddings_list:
                await self.vector_store.insert_documents(
                    documents=chunk_dicts,
                    embeddings=embeddings_list,
                    batch_size=self.batch_size
                )
            
            chunk_ids = [chunk.id for chunk in chunks]
            logger.info(
                f"Indexed document {document.id} into {len(chunks)} chunks"
            )
            
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error indexing document {document.id}: {str(e)}")
            return []
    
    async def index_documents(
        self,
        documents: List[VectorDocument],
        generate_embeddings: bool = True,
        parallel: bool = True
    ) -> Dict[str, List[str]]:
        """
        Index multiple documents in batch.
        
        Args:
            documents: List of VectorDocument to index
            generate_embeddings: Whether to generate embeddings
            parallel: Whether to process documents in parallel
            
        Returns:
            Dictionary mapping document IDs to chunk IDs
        """
        results = {}
        
        if parallel:
            # Process documents in parallel
            tasks = [
                self.index_document(doc, generate_embeddings)
                for doc in documents
            ]
            chunk_lists = await asyncio.gather(*tasks)
            
            for doc, chunks in zip(documents, chunk_lists):
                results[doc.id] = chunks
        else:
            # Process sequentially
            for doc in documents:
                chunks = await self.index_document(doc, generate_embeddings)
                results[doc.id] = chunks
        
        total_chunks = sum(len(chunks) for chunks in results.values())
        logger.info(
            f"Indexed {len(documents)} documents into {total_chunks} chunks"
        )
        
        return results
    
    def _chunk_document(
        self,
        document: VectorDocument
    ) -> List[VectorDocument]:
        """
        Split document into chunks based on selected strategy.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunk documents
        """
        if self.chunking_strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(document)
        elif self.chunking_strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(document)
        elif self.chunking_strategy == ChunkingStrategy.HIERARCHICAL:
            return self._chunk_hierarchical(document)
        elif self.chunking_strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_by_sentence(document)
        elif self.chunking_strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_by_paragraph(document)
        else:
            # Default to fixed-size chunking
            return self._chunk_fixed_size(document)
    
    def _chunk_fixed_size(
        self,
        document: VectorDocument
    ) -> List[VectorDocument]:
        """
        Split document into fixed-size chunks with overlap.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunks with fixed size
        """
        text = document.text
        chunks = []
        
        # Approximate token count by character count (simple heuristic)
        # In production, use proper tokenization
        words = text.split()
        chunk_size_words = self.chunk_size // 4  # Rough conversion: 4 chars per word
        overlap_words = self.chunk_overlap // 4
        
        for i in range(0, len(words), chunk_size_words - overlap_words):
            chunk_words = words[i:i + chunk_size_words]
            if len(chunk_words) < 50:  # Skip very short chunks
                continue
            
            chunk_text = ' '.join(chunk_words)
            
            chunk = VectorDocument(
                text=chunk_text,
                metadata=document.metadata,
                parent_id=document.id,
                chunk_index=len(chunks)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_semantic(
        self,
        document: VectorDocument
    ) -> List[VectorDocument]:
        """
        Split document at semantic boundaries (paragraphs, sections).
        
        This method attempts to split at natural breaks in the text,
        such as paragraph boundaries or section headings.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of semantically-coherent chunks
        """
        text = document.text
        chunks = []
        
        # Split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph.split())
            
            if current_size + paragraph_size > self.chunk_size:
                # Create chunk from accumulated paragraphs
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk = VectorDocument(
                        text=chunk_text,
                        metadata=document.metadata,
                        parent_id=document.id,
                        chunk_index=len(chunks)
                    )
                    chunks.append(chunk)
                
                # Start new chunk
                current_chunk = [paragraph]
                current_size = paragraph_size
            else:
                current_chunk.append(paragraph)
                current_size += paragraph_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk = VectorDocument(
                text=chunk_text,
                metadata=document.metadata,
                parent_id=document.id,
                chunk_index=len(chunks)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_hierarchical(
        self,
        document: VectorDocument
    ) -> List[VectorDocument]:
        """
        Create hierarchical chunks with parent-child relationships.
        
        This creates multiple levels of chunks:
        - Level 0: Document-level (full document)
        - Level 1: Section-level (large chunks)
        - Level 2: Paragraph-level (small chunks for fine-grained retrieval)
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunks at various granularities
        """
        chunks = []
        
        # Level 1: Section chunks (large chunks)
        sections = self._extract_sections(document.text)
        for idx, section in enumerate(sections):
            chunk = VectorDocument(
                text=section,
                metadata=document.metadata,
                parent_id=document.id,
                chunk_index=idx,
                custom_metadata={'level': 1}
            )
            chunks.append(chunk)
            
            # Level 2: Paragraph chunks within section
            paragraphs = section.split('\n\n')
            for p_idx, paragraph in enumerate(paragraphs):
                if len(paragraph.split()) > 50:  # Only create chunks for substantial paragraphs
                    para_chunk = VectorDocument(
                        text=paragraph,
                        metadata=document.metadata,
                        parent_id=chunk.id,
                        chunk_index=p_idx,
                        custom_metadata={'level': 2}
                    )
                    chunks.append(para_chunk)
        
        return chunks
    
    def _chunk_by_sentence(
        self,
        document: VectorDocument
    ) -> List[VectorDocument]:
        """
        Split document at sentence boundaries.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of sentence-based chunks
        """
        text = document.text
        chunks = []
        
        # Simple sentence splitting (improve with NLP in production)
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_words = len(sentence.split())
            
            if current_size + sentence_words > self.chunk_size:
                if current_chunk:
                    chunk_text = '. '.join(current_chunk) + '.'
                    chunk = VectorDocument(
                        text=chunk_text,
                        metadata=document.metadata,
                        parent_id=document.id,
                        chunk_index=len(chunks)
                    )
                    chunks.append(chunk)
                
                current_chunk = [sentence]
                current_size = sentence_words
            else:
                current_chunk.append(sentence)
                current_size += sentence_words
        
        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunk = VectorDocument(
                text=chunk_text,
                metadata=document.metadata,
                parent_id=document.id,
                chunk_index=len(chunks)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_paragraph(
        self,
        document: VectorDocument
    ) -> List[VectorDocument]:
        """
        Split document by paragraph boundaries.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of paragraph chunks
        """
        text = document.text
        chunks = []
        
        # Split by double newlines
        paragraphs = text.split('\n\n')
        
        for idx, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                chunk = VectorDocument(
                    text=paragraph.strip(),
                    metadata=document.metadata,
                    parent_id=document.id,
                    chunk_index=idx
                )
                chunks.append(chunk)
        
        return chunks
    
    def _extract_sections(self, text: str) -> List[str]:
        """
        Extract sections from document text based on headings.
        
        This method identifies common medical document section headings.
        
        Args:
            text: Document text
            
        Returns:
            List of section texts
        """
        # Common section headings in medical documents
        section_markers = [
            'INTRODUCTION', 'BACKGROUND', 'METHODS', 'RESULTS',
            'DISCUSSION', 'CONCLUSION', 'REFERENCES', 'APPENDIX',
            'ABSTRACT', 'OBJECTIVE', 'MATERIALS', 'ANALYSIS',
            'FINDINGS', 'LIMITATIONS', 'RECOMMENDATIONS'
        ]
        
        sections = []
        current_section = []
        
        lines = text.split('\n')
        
        for line in lines:
            line_upper = line.upper().strip()
            is_heading = any(marker in line_upper for marker in section_markers)
            
            if is_heading and current_section:
                sections.append('\n'.join(current_section))
                current_section = []
            
            current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections if sections else [text]
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text for indexing.
        
        Operations:
        - Remove extra whitespace
        - Normalize line breaks
        - Remove special characters (optional)
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        import re
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    async def index_from_directory(
        self,
        directory_path: str,
        file_pattern: str = "*.txt",
        recursive: bool = True
    ) -> Dict[str, List[str]]:
        """
        Index all documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            file_pattern: Pattern to match files
            recursive: Whether to search subdirectories
            
        Returns:
            Dictionary mapping file paths to chunk IDs
        """
        path = Path(directory_path)
        
        if not path.exists():
            logger.error(f"Directory not found: {directory_path}")
            return {}
        
        # Find all matching files
        if recursive:
            files = list(path.rglob(file_pattern))
        else:
            files = list(path.glob(file_pattern))
        
        logger.info(f"Found {len(files)} files to index")
        
        # Process each file
        results = {}
        for file_path in files:
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create document
                metadata = DocumentMetadata(
                    source=DocumentSource.CUSTOM.value,
                    category=self._determine_category(file_path),
                    author="Unknown",
                    access_date=datetime.now().isoformat()
                )
                
                doc = VectorDocument(
                    text=content,
                    metadata=metadata,
                    id=str(hashlib.md5(str(file_path).encode()).hexdigest())
                )
                
                # Index document
                chunk_ids = await self.index_document(doc)
                results[str(file_path)] = chunk_ids
                
                logger.info(f"Indexed {file_path} -> {len(chunk_ids)} chunks")
                
            except Exception as e:
                logger.error(f"Error indexing {file_path}: {str(e)}")
        
        return results
    
    def _determine_category(self, file_path: Path) -> str:
        """
        Determine document category from file path or name.
        
        Args:
            file_path: Path to document
            
        Returns:
            Category string
        """
        path_str = str(file_path).lower()
        
        if 'guideline' in path_str:
            return DocumentCategory.CLINICAL_GUIDELINE.value
        elif 'policy' in path_str:
            return DocumentCategory.HOSPITAL_POLICY.value
        elif 'drug' in path_str or 'medication' in path_str:
            return DocumentCategory.DRUG_INFORMATION.value
        elif 'literature' in path_str or 'paper' in path_str:
            return DocumentCategory.LITERATURE.value
        elif 'textbook' in path_str:
            return DocumentCategory.MEDICAL_TEXTBOOK.value
        elif 'protocol' in path_str:
            return DocumentCategory.CLINICAL_GUIDELINE.value
        
        return DocumentCategory.LITERATURE.value