"""
Paper Summarization Engine for Medical Literature
=================================================

This module provides intelligent summarization of medical papers using
transformer models. It supports multiple summarization strategies:

1. Extractive Summarization: Selects important sentences directly from text
2. Abstractive Summarization: Generates new sentences capturing key ideas
3. Hybrid Approach: Combines both methods for optimal results
4. Section-based Summarization: Summarizes specific sections (Methods, Results, etc.)

The module uses fine-tuned biomedical language models for domain-specific
understanding and generates structured summaries suitable for clinical use.

Author: AegisMedBot Team
Version: 1.0.0
"""

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    pipeline
)
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import numpy as np
import logging
from dataclasses import dataclass, field
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

logger = logging.getLogger(__name__)

class SummarizationMethod(Enum):
    """
    Enumeration of available summarization methods.
    
    Each method has different strengths and use cases:
    - EXTRACTIVE: Best for preserving original wording, good for factual content
    - ABSTRACTIVE: Better for generating concise summaries, captures main ideas
    - HYBRID: Combines both, generally produces best results
    - SECTION: Focuses on specific paper sections
    """
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    HYBRID = "hybrid"
    SECTION = "section"

class SummaryLength(Enum):
    """
    Enumeration of summary length options.
    """
    TINY = "tiny"      # 1-2 sentences
    SHORT = "short"    # 3-4 sentences
    MEDIUM = "medium"  # 5-7 sentences
    LONG = "long"      # 8-10 sentences
    DETAILED = "detailed"  # 10+ sentences
    
    def get_sentence_count(self) -> int:
        """Get approximate sentence count for length."""
        counts = {
            SummaryLength.TINY: 2,
            SummaryLength.SHORT: 4,
            SummaryLength.MEDIUM: 6,
            SummaryLength.LONG: 8,
            SummaryLength.DETAILED: 12
        }
        return counts.get(self, 6)
    
    def get_token_ratio(self) -> float:
        """
        Get token ratio relative to original text.
        Used for abstractive summarization.
        """
        ratios = {
            SummaryLength.TINY: 0.1,
            SummaryLength.SHORT: 0.2,
            SummaryLength.MEDIUM: 0.3,
            SummaryLength.LONG: 0.4,
            SummaryLength.DETAILED: 0.5
        }
        return ratios.get(self, 0.3)

@dataclass
class SummaryResult:
    """
    Container for summarization results with metadata.
    
    This class holds the generated summary along with important
    metadata about the summarization process.
    """
    text: str  # The generated summary text
    method: SummarizationMethod  # Method used
    length: SummaryLength  # Target length
    confidence: float  # Confidence score (0-1)
    processing_time: float  # Time taken in seconds
    source_sentences: List[str] = field(default_factory=list)  # For extractive: source sentences used
    metrics: Dict[str, float] = field(default_factory=dict)  # Additional metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'text': self.text,
            'method': self.method.value,
            'length': self.length.value,
            'confidence': self.confidence,
            'processing_time': self.processing_time,
            'source_sentences': self.source_sentences,
            'metrics': self.metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SummaryResult':
        """Create from dictionary."""
        return cls(
            text=data['text'],
            method=SummarizationMethod(data['method']),
            length=SummaryLength(data['length']),
            confidence=data['confidence'],
            processing_time=data['processing_time'],
            source_sentences=data.get('source_sentences', []),
            metrics=data.get('metrics', {})
        )

class PaperSummarizer:
    """
    Main paper summarization engine.
    
    This class provides comprehensive summarization capabilities for medical papers.
    It supports multiple models and summarization strategies, with automatic
    GPU/CPU selection and batch processing for efficiency.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        abstractive_model: str = "facebook/bart-large-cnn",
        device: Optional[str] = None,
        use_gpu: bool = True,
        cache_dir: Optional[str] = None,
        max_length: int = 1024,
        batch_size: int = 8
    ):
        """
        Initialize the paper summarizer.
        
        Args:
            model_name: Name of the biomedical transformer model
            abstractive_model: Name of the abstractive summarization model
            device: Device to run models on ('cuda' or 'cpu')
            use_gpu: Whether to use GPU if available
            cache_dir: Directory to cache models
            max_length: Maximum input token length
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.abstractive_model_name = abstractive_model
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Determine device
        if device:
            self.device = device
        else:
            self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        logger.info("Initializing PaperSummarizer on device: %s", self.device)
        
        # Initialize models (lazy loading - only load when needed)
        self.extractive_model = None
        self.extractive_tokenizer = None
        self.abstractive_model = None
        self.abstractive_tokenizer = None
        self.sentence_classifier = None
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Cache for generated summaries
        self.summary_cache = {}
        
        logger.info("PaperSummarizer initialized successfully")
    
    def _load_extractive_model(self):
        """
        Load model for extractive summarization.
        
        Extractive summarization uses a sentence classifier to select
        the most important sentences from the original text.
        """
        if self.extractive_model is not None:
            return
        
        logger.info("Loading extractive model: %s", self.model_name)
        
        try:
            # Load tokenizer and model
            self.extractive_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self.extractive_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            self.extractive_model.eval()
            logger.info("Extractive model loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load extractive model: %s", str(e))
            raise
    
    def _load_abstractive_model(self):
        """
        Load model for abstractive summarization.
        
        Abstractive summarization uses a sequence-to-sequence model to
        generate novel sentences capturing the key information.
        """
        if self.abstractive_model is not None:
            return
        
        logger.info("Loading abstractive model: %s", self.abstractive_model_name)
        
        try:
            # Load tokenizer and model
            self.abstractive_tokenizer = AutoTokenizer.from_pretrained(
                self.abstractive_model_name,
                cache_dir=self.cache_dir
            )
            
            self.abstractive_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.abstractive_model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            self.abstractive_model.eval()
            logger.info("Abstractive model loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load abstractive model: %s", str(e))
            raise
    
    def _load_sentence_classifier(self):
        """
        Load a sentence classifier for extractive summarization.
        
        This is a simplified version - in production, use a proper
        sentence-level classifier.
        """
        # Reuse the extractive model for now
        self._load_extractive_model()
        self.sentence_classifier = self.extractive_model
    
    async def summarize(
        self,
        text: str,
        method: SummarizationMethod = SummarizationMethod.HYBRID,
        length: SummaryLength = SummaryLength.MEDIUM,
        focus_sections: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        use_cache: bool = True
    ) -> SummaryResult:
        """
        Summarize a paper or medical text.
        
        This is the main entry point for summarization. It selects the
        appropriate method and generates a summary.
        
        Args:
            text: The text to summarize (paper, abstract, etc.)
            method: Summarization method to use
            length: Desired summary length
            focus_sections: For SECTION method, which sections to focus on
            min_confidence: Minimum confidence threshold
            use_cache: Whether to use cached results
            
        Returns:
            SummaryResult containing the generated summary and metadata
        """
        import time
        start_time = time.time()
        
        # Check cache
        cache_key = self._generate_cache_key(text, method, length)
        if use_cache and cache_key in self.summary_cache:
            logger.debug("Cache hit for summarization")
            return self.summary_cache[cache_key]
        
        logger.info("Summarizing text of length %d with method: %s", len(text), method.value)
        
        # Validate input
        if not text or len(text.strip()) == 0:
            raise ValueError("Empty text provided for summarization")
        
        # Preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Route to appropriate method
        if method == SummarizationMethod.EXTRACTIVE:
            result = await self._extractive_summarize(cleaned_text, length)
        elif method == SummarizationMethod.ABSTRACTIVE:
            result = await self._abstractive_summarize(cleaned_text, length)
        elif method == SummarizationMethod.HYBRID:
            result = await self._hybrid_summarize(cleaned_text, length)
        elif method == SummarizationMethod.SECTION:
            result = await self._section_summarize(cleaned_text, focus_sections, length)
        else:
            raise ValueError(f"Unknown summarization method: {method}")
        
        # Add processing time
        result.processing_time = time.time() - start_time
        
        # Cache result
        if use_cache:
            self.summary_cache[cache_key] = result
        
        logger.info("Summarization completed in %.2f seconds", result.processing_time)
        
        return result
    
    async def summarize_batch(
        self,
        texts: List[str],
        method: SummarizationMethod = SummarizationMethod.HYBRID,
        length: SummaryLength = SummaryLength.MEDIUM,
        batch_size: Optional[int] = None
    ) -> List[SummaryResult]:
        """
        Summarize multiple texts in batch for efficiency.
        
        Args:
            texts: List of texts to summarize
            method: Summarization method
            length: Desired summary length
            batch_size: Batch size (defaults to self.batch_size)
            
        Returns:
            List of SummaryResult objects
        """
        batch_size = batch_size or self.batch_size
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_tasks = [self.summarize(t, method, length) for t in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results
    
    async def _extractive_summarize(
        self,
        text: str,
        length: SummaryLength
    ) -> SummaryResult:
        """
        Perform extractive summarization.
        
        This method selects the most important sentences directly from
        the original text based on their relevance scores.
        """
        # Load models if needed
        self._load_sentence_classifier()
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= length.get_sentence_count():
            # Text is already short enough
            return SummaryResult(
                text=text,
                method=SummarizationMethod.EXTRACTIVE,
                length=length,
                confidence=1.0,
                processing_time=0,
                source_sentences=sentences
            )
        
        # Score sentences
        scores = await self._score_sentences(sentences)
        
        # Select top sentences
        num_sentences = min(length.get_sentence_count(), len(sentences))
        top_indices = np.argsort(scores)[-num_sentences:]
        top_indices = sorted(top_indices)  # Keep original order
        
        # Extract selected sentences
        selected_sentences = [sentences[i] for i in top_indices]
        selected_scores = [scores[i] for i in top_indices]
        
        # Calculate confidence (average score of selected sentences)
        confidence = float(np.mean(selected_scores))
        
        # Combine sentences
        summary_text = ' '.join(selected_sentences)
        
        return SummaryResult(
            text=summary_text,
            method=SummarizationMethod.EXTRACTIVE,
            length=length,
            confidence=confidence,
            processing_time=0,
            source_sentences=selected_sentences,
            metrics={'sentence_scores': selected_scores.tolist()}
        )
    
    async def _abstractive_summarize(
        self,
        text: str,
        length: SummaryLength
    ) -> SummaryResult:
        """
        Perform abstractive summarization.
        
        This method generates new sentences that capture the key
        information from the original text.
        """
        # Load models if needed
        self._load_abstractive_model()
        
        # Handle long texts by chunking
        if len(text.split()) > 1000:
            # Summarize chunks and then combine
            chunks = self._chunk_text(text, max_words=800)
            chunk_summaries = []
            
            for chunk in chunks:
                chunk_summary = await self._abstractive_summarize_single(chunk, length)
                chunk_summaries.append(chunk_summary)
            
            # Combine chunk summaries
            combined = ' '.join(chunk_summaries)
            final_summary = await self._abstractive_summarize_single(combined, length)
            
        else:
            # Single chunk
            final_summary = await self._abstractive_summarize_single(text, length)
        
        # Calculate confidence (using model's own confidence scores)
        confidence = 0.85  # Placeholder - in production, get from model
        
        return SummaryResult(
            text=final_summary,
            method=SummarizationMethod.ABSTRACTIVE,
            length=length,
            confidence=confidence,
            processing_time=0
        )
    
    async def _abstractive_summarize_single(
        self,
        text: str,
        length: SummaryLength
    ) -> str:
        """
        Summarize a single text chunk with the abstractive model.
        """
        # Tokenize input
        inputs = self.abstractive_tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Calculate target length
        input_length = inputs['input_ids'].shape[1]
        target_length = int(input_length * length.get_token_ratio())
        min_length = max(20, target_length // 2)
        max_length = max(50, target_length)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.abstractive_model.generate(
                inputs['input_ids'],
                num_beams=4,
                min_length=min_length,
                max_length=max_length,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=2.0
            )
        
        # Decode
        summary = self.abstractive_tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )
        
        return summary
    
    async def _hybrid_summarize(
        self,
        text: str,
        length: SummaryLength
    ) -> SummaryResult:
        """
        Perform hybrid summarization combining extractive and abstractive methods.
        
        Steps:
        1. Use extractive method to select key sentences
        2. Use abstractive method to refine and combine them
        3. This often produces better results than either method alone
        """
        # First, extract key sentences (use more than final target)
        extractive_length = SummaryLength.MEDIUM
        extractive_result = await self._extractive_summarize(text, extractive_length)
        
        # Then, abstractively summarize the extracted text
        extracted_text = extractive_result.text
        abstractive_result = await self._abstractive_summarize(extracted_text, length)
        
        # Combine with metadata
        return SummaryResult(
            text=abstractive_result.text,
            method=SummarizationMethod.HYBRID,
            length=length,
            confidence=extractive_result.confidence * 0.4 + 0.6,  # Weighted confidence
            processing_time=extractive_result.processing_time + abstractive_result.processing_time,
            source_sentences=extractive_result.source_sentences,
            metrics={
                'extractive_confidence': extractive_result.confidence,
                'abstractive_confidence': abstractive_result.confidence
            }
        )
    
    async def _section_summarize(
        self,
        text: str,
        sections: Optional[List[str]],
        length: SummaryLength
    ) -> SummaryResult:
        """
        Summarize specific sections of a paper.
        
        This method identifies and extracts specific sections
        (e.g., Methods, Results) and summarizes them individually.
        """
        # Identify sections in the text
        section_boundaries = self._identify_sections(text)
        
        if not section_boundaries:
            # No clear sections found, fall back to hybrid
            return await self._hybrid_summarize(text, length)
        
        # If sections specified, focus on those
        if sections:
            relevant_sections = {
                name: (start, end) 
                for name, (start, end) in section_boundaries.items()
                if any(s.lower() in name.lower() for s in sections)
            }
        else:
            # Summarize all major sections
            relevant_sections = section_boundaries
        
        # Summarize each relevant section
        section_summaries = []
        for section_name, (start, end) in relevant_sections.items():
            section_text = text[start:end]
            section_summary = await self._hybrid_summarize(
                section_text,
                SummaryLength.SHORT
            )
            section_summaries.append(f"{section_name}:\n{section_summary.text}")
        
        # Combine section summaries
        combined_summary = '\n\n'.join(section_summaries)
        
        return SummaryResult(
            text=combined_summary,
            method=SummarizationMethod.SECTION,
            length=length,
            confidence=0.8,
            processing_time=0,
            metrics={'sections_found': list(relevant_sections.keys())}
        )
    
    async def _score_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Score sentences for extractive summarization.
        
        Uses the sentence classifier to assign importance scores
        to each sentence.
        """
        if not sentences:
            return np.array([])
        
        # Tokenize sentences
        inputs = self.extractive_tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        # Get scores
        with torch.no_grad():
            outputs = self.extractive_model(**inputs)
            scores = F.softmax(outputs.logits, dim=-1)
            
            # For binary classification, use probability of positive class
            if scores.shape[1] == 2:
                scores = scores[:, 1]
            else:
                scores = scores.max(dim=-1)[0]
        
        return scores.cpu().numpy()
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Uses simple heuristics - in production, use a proper
        sentence tokenizer like spaCy or NLTK.
        """
        # Simple sentence splitting on punctuation
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _chunk_text(self, text: str, max_words: int = 500) -> List[str]:
        """
        Split long text into overlapping chunks for processing.
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_words // 2):
            chunk = ' '.join(words[i:i+max_words])
            chunks.append(chunk)
        
        return chunks
    
    def _identify_sections(self, text: str) -> Dict[str, Tuple[int, int]]:
        """
        Identify paper sections by common headings.
        
        Returns dictionary mapping section names to character spans.
        """
        sections = {}
        
        # Common section headings in medical papers
        section_patterns = [
            'ABSTRACT', 'INTRODUCTION', 'METHODS', 'RESULTS',
            'DISCUSSION', 'CONCLUSION', 'REFERENCES', 'MATERIALS AND METHODS',
            'BACKGROUND', 'OBJECTIVE', 'FINDINGS', 'LIMITATIONS'
        ]
        
        text_upper = text.upper()
        
        # Find section boundaries
        for i, heading in enumerate(section_patterns):
            start = text_upper.find(heading)
            if start == -1:
                continue
            
            # Find end (next heading)
            end = len(text)
            for j in range(i+1, len(section_patterns)):
                next_heading = section_patterns[j]
                next_pos = text_upper.find(next_heading, start + len(heading))
                if next_pos != -1:
                    end = next_pos
                    break
            
            sections[heading.title()] = (start, end)
        
        return sections
    
    def _preprocess_text(self, text: str) -> str:
        """
        Clean and normalize text before summarization.
        
        Steps:
        1. Remove extra whitespace
        2. Normalize unicode characters
        3. Handle special characters
        """
        import unicodedata
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _generate_cache_key(
        self,
        text: str,
        method: SummarizationMethod,
        length: SummaryLength
    ) -> str:
        """
        Generate cache key for a summarization request.
        """
        content = f"{text[:100]}_{method.value}_{length.value}"
        return hashlib.md5(content.encode()).hexdigest()

# Export public interface
__all__ = [
    'PaperSummarizer',
    'SummarizationMethod',
    'SummaryLength',
    'SummaryResult'
]