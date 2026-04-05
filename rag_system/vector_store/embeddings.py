"""
Medical Embedding Generator Module

This module provides specialized embedding generation for medical text using
transformer models fine-tuned on biomedical literature. It supports multiple
embedding strategies including mean pooling, CLS token pooling, and query
expansion for improved retrieval.

The embedding generator is designed to handle the unique characteristics
of medical text including:
- Medical terminology and abbreviations
- Long clinical notes and documents
- Specialized vocabulary from different medical domains
- Hierarchical relationships in medical concepts

Key Features:
- Caching of embeddings for frequently accessed texts
- Batch processing for efficient embedding generation
- Query expansion using medical ontologies
- Support for multiple pooling strategies
- GPU acceleration when available

Dependencies:
- transformers: For loading pre-trained models
- torch: For tensor operations
- sentence-transformers: For additional embedding options
"""

import torch
import numpy as np
from typing import List, Union, Optional, Dict, Any
from functools import lru_cache
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime
import hashlib

# Configure logging for the module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalEmbeddingGenerator:
    """
    Generates high-quality embeddings for medical text using transformer models.
    
    This class is the core component for converting medical text into vector
    representations that can be used for semantic search and similarity comparison.
    It uses pre-trained models specifically fine-tuned on biomedical literature
    to capture the nuanced meaning of clinical terminology.
    
    The class supports multiple pooling strategies to extract the most relevant
    information from transformer outputs:
    - mean pooling: Average of all token embeddings (good for general text)
    - cls pooling: Use CLS token representation (good for classification tasks)
    - max pooling: Maximum across token dimensions (good for capturing key features)
    
    Architecture:
    - Uses PubMedBERT or similar biomedical language models
    - Supports both transformer and sentence-transformer models
    - Implements caching for performance optimization
    - Provides query expansion for improved retrieval
    
    Attributes:
        device: The computing device (cuda/cpu) for model execution
        batch_size: Number of texts to process in a single batch
        tokenizer: Tokenizer for converting text to model inputs
        model: Pre-trained transformer model for generating embeddings
        sentence_transformer: Alternative embedding model for simpler tasks
        embedding_dimension: Dimension of the generated embeddings
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        device: Optional[str] = None,
        batch_size: int = 32,
        use_cache: bool = True,
        cache_size: int = 10000
    ):
        """
        Initialize the medical embedding generator.
        
        Args:
            model_name: Name or path of the pre-trained transformer model.
                        Default uses PubMedBERT which is optimized for biomedical text.
            device: Computing device to use. Auto-detects CUDA if available.
            batch_size: Number of texts to process per batch for efficiency.
            use_cache: Whether to cache embeddings for repeated texts.
            cache_size: Maximum number of embeddings to keep in cache.
        
        The constructor loads the specified model and tokenizer, sets up the
        computing device, and initializes the embedding cache if enabled.
        It also validates that the model can be loaded correctly.
        """
        # Determine the computing device to use
        # Prefer CUDA if available for faster computation
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Store configuration parameters
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.cache_size = cache_size
        
        # Initialize cache for storing frequently used embeddings
        # This significantly improves performance for repeated queries
        self._embedding_cache = {} if use_cache else None
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"Loading medical embedding model on {self.device}")
        
        try:
            # Load the tokenizer for converting text to model inputs
            # The tokenizer handles medical terminology and special tokens
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load the pre-trained transformer model
            # This model has been fine-tuned on biomedical literature
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            
            # Set model to evaluation mode (disables dropout, etc.)
            self.model.eval()
            
            # Store the embedding dimension for reference
            # This is needed when creating vector databases
            self.embedding_dimension = self.model.config.hidden_size
            
            # Also load a sentence transformer for simpler tasks
            # This provides a lightweight alternative when full transformer is overkill
            self.sentence_transformer = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device=self.device
            )
            
            logger.info(f"Successfully loaded embedding model with dimension {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise RuntimeError(f"Could not initialize embedding model: {e}")
    
    def _get_cache_key(self, text: str, pooling_strategy: str) -> str:
        """
        Generate a unique cache key for a text and pooling strategy.
        
        Args:
            text: Input text to generate embedding for
            pooling_strategy: Pooling method used
        
        Returns:
            A hashed string that uniquely identifies the (text, strategy) pair
        
        This method creates a deterministic hash of the input text and pooling
        strategy to use as a cache key. This allows quick lookup of previously
        computed embeddings.
        """
        # Create a combined string of text and strategy
        combined = f"{text}_{pooling_strategy}"
        
        # Generate MD5 hash for the combined string
        # MD5 is fast and produces a fixed-length key
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _get_cached_embedding(self, text: str, pooling_strategy: str) -> Optional[np.ndarray]:
        """
        Retrieve a cached embedding if available.
        
        Args:
            text: Input text to look up
            pooling_strategy: Pooling strategy used
        
        Returns:
            Cached embedding array or None if not found
        
        This method checks the cache for existing embeddings to avoid
        recomputing the same text multiple times.
        """
        if not self.use_cache:
            return None
        
        cache_key = self._get_cache_key(text, pooling_strategy)
        
        if cache_key in self._embedding_cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit for text: {text[:50]}...")
            return self._embedding_cache[cache_key]
        
        self._cache_misses += 1
        return None
    
    def _cache_embedding(self, text: str, pooling_strategy: str, embedding: np.ndarray):
        """
        Store an embedding in the cache.
        
        Args:
            text: Input text that was embedded
            pooling_strategy: Pooling strategy used
            embedding: Generated embedding array
        
        This method adds a new embedding to the cache and manages cache size
        by removing the oldest entries if the cache exceeds its maximum size.
        """
        if not self.use_cache:
            return
        
        cache_key = self._get_cache_key(text, pooling_strategy)
        
        # Manage cache size by removing oldest entries if needed
        if len(self._embedding_cache) >= self.cache_size:
            # Remove the first (oldest) key-value pair
            # Python 3.7+ preserves insertion order
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
            logger.debug(f"Cache full, removed oldest entry")
        
        # Store the embedding in cache
        self._embedding_cache[cache_key] = embedding
        
    def generate_embeddings(
        self,
        texts: List[str],
        use_cache: bool = True,
        pooling_strategy: str = "mean"
    ) -> List[np.ndarray]:
        """
        Generate embeddings for a list of text documents.
        
        Args:
            texts: List of text strings to generate embeddings for
            use_cache: Whether to use cached embeddings (overrides instance setting)
            pooling_strategy: Strategy for pooling token embeddings:
                              "mean" - Average of all token embeddings
                              "cls" - Use CLS token embedding
                              "max" - Maximum across token dimensions
        
        Returns:
            List of embedding vectors as numpy arrays
        
        This is the main method for generating embeddings. It processes texts
        in batches for efficiency and uses caching when appropriate.
        
        The pooling strategy determines how the sequence of token embeddings
        is converted to a single document embedding:
        - Mean pooling: Good for general similarity tasks
        - CLS pooling: Good for classification-style tasks
        - Max pooling: Good for capturing key features
        """
        if not texts:
            return []
        
        # Validate pooling strategy
        if pooling_strategy not in ["mean", "cls", "max"]:
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}. Must be 'mean', 'cls', or 'max'")
        
        # Determine whether to use caching for this call
        use_caching = use_cache and self.use_cache
        
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text if caching is enabled
        if use_caching:
            for i, text in enumerate(texts):
                cached_embedding = self._get_cached_embedding(text, pooling_strategy)
                if cached_embedding is not None:
                    embeddings.append((i, cached_embedding))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            # Process all texts without caching
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            logger.info(f"Generating embeddings for {len(uncached_texts)} texts")
            new_embeddings = self._generate_batch_embeddings(
                uncached_texts,
                pooling_strategy
            )
            
            # Cache the new embeddings if caching is enabled
            if use_caching:
                for text, emb in zip(uncached_texts, new_embeddings):
                    self._cache_embedding(text, pooling_strategy, emb)
            
            # Add to results list
            for idx, emb in zip(uncached_indices, new_embeddings):
                embeddings.append((idx, emb))
        
        # Sort by original index to maintain order
        embeddings.sort(key=lambda x: x[0])
        
        # Log cache statistics
        if use_caching:
            logger.debug(f"Cache stats - Hits: {self._cache_hits}, Misses: {self._cache_misses}")
        
        return [emb for _, emb in embeddings]
    
    def _generate_batch_embeddings(
        self,
        texts: List[str],
        pooling_strategy: str
    ) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts using the transformer model.
        
        Args:
            texts: List of text strings to process
            pooling_strategy: Strategy for pooling token embeddings
        
        Returns:
            List of embedding vectors
        
        This internal method handles the actual model inference. It processes
        texts in smaller batches to manage memory usage and uses no_grad()
        to disable gradient computation for efficiency.
        """
        all_embeddings = []
        
        # Process in batches to manage memory
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Tokenize the batch
            # This converts text to input IDs and attention masks
            inputs = self.tokenizer(
                batch,
                padding=True,          # Pad shorter sequences
                truncation=True,       # Truncate longer sequences
                max_length=512,        # Maximum sequence length (BERT limitation)
                return_tensors="pt"    # Return PyTorch tensors
            ).to(self.device)
            
            # Generate embeddings with gradient tracking disabled
            # This saves memory and speeds up inference
            with torch.no_grad():
                # Forward pass through the transformer
                outputs = self.model(**inputs)
                
                # Get the token embeddings (last hidden state)
                # Shape: [batch_size, sequence_length, hidden_size]
                token_embeddings = outputs.last_hidden_state
                
                # Apply pooling strategy
                if pooling_strategy == "mean":
                    # Mean pooling: average all token embeddings
                    # Weight by attention mask to ignore padding tokens
                    attention_mask = inputs["attention_mask"]
                    # Expand mask to match embedding dimensions
                    mask_expanded = attention_mask.unsqueeze(-1).expand(
                        token_embeddings.size()
                    ).float()
                    # Sum embeddings weighted by mask
                    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                    # Divide by number of non-padding tokens
                    pooled = sum_embeddings / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    
                elif pooling_strategy == "cls":
                    # CLS pooling: use the first token's embedding
                    # The [CLS] token is designed for classification tasks
                    pooled = token_embeddings[:, 0, :]
                    
                elif pooling_strategy == "max":
                    # Max pooling: take maximum across token dimension
                    # Good for capturing key features
                    pooled = torch.max(token_embeddings, dim=1)[0]
                
                # Normalize embeddings to unit length
                # This improves similarity search performance
                pooled = pooled / torch.norm(pooled, dim=1, keepdim=True)
                
                # Convert to numpy arrays for storage and compatibility
                batch_embeddings = pooled.cpu().numpy()
                all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def generate_query_embedding(
        self,
        query: str,
        use_expansion: bool = True,
        expansion_terms: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Generate embedding for a search query with optional expansion.
        
        Args:
            query: Search query text
            use_expansion: Whether to expand query with related medical terms
            expansion_terms: Custom expansion terms to add
        
        Returns:
            Query embedding vector
        
        This method is specialized for search queries. It can expand the query
        with related medical terms to improve retrieval recall. Query expansion
        helps capture variations of medical terminology.
        """
        if use_expansion:
            # Expand the query with related medical terms
            expanded_query = self._expand_medical_query(query)
            
            # Add custom expansion terms if provided
            if expansion_terms:
                expanded_query += " " + " ".join(expansion_terms)
            
            texts = [expanded_query]
        else:
            texts = [query]
        
        # Generate embedding for the expanded query
        embeddings = self.generate_embeddings(texts, use_cache=True)
        return embeddings[0]
    
    def _expand_medical_query(self, query: str) -> str:
        """
        Expand medical query with related terms and synonyms.
        
        Args:
            query: Original search query
        
        Returns:
            Expanded query with additional medical terms
        
        This method uses a medical term dictionary to add synonyms and
        related terms to the query. This improves recall by capturing
        different ways clinicians might express the same concept.
        
        The expansion is based on a medical thesaurus that includes:
        - Common medical synonyms
        - Abbreviations and their expansions
        - Related clinical concepts
        """
        # Define medical term expansions
        # In production, this would be loaded from a medical ontology
        expansions = {
            "heart attack": "myocardial infarction acute coronary syndrome cardiac arrest",
            "heart failure": "congestive heart failure cardiac decompensation",
            "high blood pressure": "hypertension hypertensive elevated blood pressure",
            "diabetes": "diabetes mellitus hyperglycemia blood sugar",
            "cancer": "malignant tumor carcinoma neoplasm oncology",
            "stroke": "cerebrovascular accident cva brain attack cerebral infarction",
            "infection": "sepsis infectious disease bacterial viral",
            "pneumonia": "lung infection respiratory infection pneumonitis",
            "kidney failure": "renal failure acute kidney injury chronic kidney disease",
            "liver disease": "hepatic disease cirrhosis hepatitis",
            "covid": "coronavirus sars-cov-2 covid-19",
            "ventilator": "mechanical ventilation respiratory support life support",
            "icu": "intensive care unit critical care",
            "er": "emergency room emergency department ed",
            "mri": "magnetic resonance imaging",
            "ct": "computed tomography cat scan",
            "eeg": "electroencephalogram",
            "ecg": "electrocardiogram ekg",
            "blood pressure": "bp systolic diastolic",
            "oxygen": "o2 saturation spo2",
            "temperature": "temp fever afebrile",
            "medication": "drug pharmaceutical prescription",
            "aspirin": "acetylsalicylic acid asa",
            "ibuprofen": "advil motrin nsaid",
            "acetaminophen": "tylenol paracetamol",
            "antibiotic": "antimicrobial antibacterial",
            "surgery": "operation surgical procedure",
            "chemotherapy": "chemo radiation oncology treatment"
        }
        
        # Convert query to lowercase for matching
        query_lower = query.lower()
        expanded_parts = [query]
        
        # Check for each term and add expansions
        for term, expansion in expansions.items():
            if term in query_lower:
                expanded_parts.append(expansion)
                logger.debug(f"Expanded '{term}' with: {expansion[:50]}...")
        
        # Combine all parts
        expanded_query = " ".join(expanded_parts)
        
        # If no expansions were found, return original query
        if len(expanded_parts) == 1:
            logger.debug("No query expansions applied")
        else:
            logger.info(f"Query expanded from '{query}' to '{expanded_query[:100]}...'")
        
        return expanded_query
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the embedding cache.
        
        Returns:
            Dictionary with cache performance metrics
        
        This method provides insight into cache effectiveness, which can be
        used to optimize caching behavior.
        """
        if not self.use_cache:
            return {"enabled": False}
        
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "enabled": True,
            "size": len(self._embedding_cache),
            "max_size": self.cache_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }
    
    def clear_cache(self):
        """
        Clear the embedding cache.
        
        This method empties the cache and resets statistics.
        Useful for memory management when processing large datasets.
        """
        if self._embedding_cache:
            self._embedding_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            logger.info("Embedding cache cleared")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings generated by this model.
        
        Returns:
            Integer representing the embedding vector dimension
        
        This is useful for initializing vector databases that need
        to know the embedding size.
        """
        return self.embedding_dimension
    
    def generate_simple_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings using the lightweight sentence transformer.
        
        Args:
            texts: List of text strings
        
        Returns:
            List of embedding vectors
        
        This method uses a smaller, faster model for simple tasks
        where the full PubMedBERT might be overkill. It's useful for
        quick similarity checks and prototyping.
        """
        embeddings = self.sentence_transformer.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.tolist()