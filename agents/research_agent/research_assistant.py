"""
Research Assistant Agent Module for AegisMedBot

This module implements a specialized AI agent focused on medical research tasks including
literature retrieval, paper summarization, evidence synthesis, and clinical trial analysis.
The agent integrates with external medical databases and uses transformer models for
natural language understanding and generation.
"""

import asyncio
import logging
import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    pipeline
)
import numpy as np
from sentence_transformers import SentenceTransformer
import aiohttp
import aiofiles
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field, validator

# Configure logging for the research agent
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Models and Schemas
# ============================================================================

class ResearchQuery(BaseModel):
    """
    Structured representation of a research query with metadata.
    
    This model defines the expected format for incoming research requests,
    including query parameters, filters, and user context.
    """
    query_text: str = Field(..., description="The main research question or topic")
    query_type: str = Field(
        default="general",
        description="Type of research: literature_review, clinical_trial, drug_info, etc."
    )
    date_range: Optional[Tuple[datetime, datetime]] = Field(
        None,
        description="Optional date range for filtering results"
    )
    sources: List[str] = Field(
        default=["pubmed", "clinicaltrials", "cochrane"],
        description="List of sources to search"
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of results to return"
    )
    relevance_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score for including results"
    )
    user_id: Optional[str] = Field(None, description="ID of the user making the request")
    conversation_id: Optional[str] = Field(None, description="Conversation context ID")
    
    @validator('query_text')
    def validate_query_not_empty(cls, v):
        """Ensure query text is not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError('Query text cannot be empty')
        return v.strip()
    
    @validator('query_type')
    def validate_query_type(cls, v):
        """Validate that the query type is supported."""
        valid_types = ['general', 'literature_review', 'clinical_trial', 
                      'drug_info', 'guideline', 'systematic_review']
        if v not in valid_types:
            raise ValueError(f'Query type must be one of: {valid_types}')
        return v

class ResearchPaper(BaseModel):
    """
    Represents a scientific paper or article with metadata.
    
    This model structures research paper information for consistent handling
    across different sources and formats.
    """
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    abstract: str = Field(..., description="Paper abstract")
    journal: Optional[str] = Field(None, description="Journal name")
    publication_date: Optional[datetime] = Field(None, description="Publication date")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    pmid: Optional[str] = Field(None, description="PubMed ID")
    source: str = Field(..., description="Source database (PubMed, Cochrane, etc.)")
    url: Optional[str] = Field(None, description="URL to access the paper")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    mesh_terms: List[str] = Field(default_factory=list, description="MeSH terms")
    citation_count: int = Field(default=0, description="Number of citations")
    full_text_available: bool = Field(default=False, description="Whether full text is accessible")
    relevance_score: float = Field(default=0.0, description="Relevance to query")
    
    class Config:
        """Pydantic configuration for the model."""
        arbitrary_types_allowed = True

class ResearchSummary(BaseModel):
    """
    Summarized research findings with key insights and metadata.
    
    This model structures the output of the research agent, providing
    synthesized information from multiple sources.
    """
    query: str = Field(..., description="Original research query")
    summary: str = Field(..., description="Synthesized summary of findings")
    key_findings: List[str] = Field(default_factory=list, description="Key findings")
    evidence_level: str = Field(
        default="moderate",
        description="Level of evidence (high, moderate, low)"
    )
    sources: List[ResearchPaper] = Field(default_factory=list, description="Source papers")
    limitations: List[str] = Field(default_factory=list, description="Limitations of the evidence")
    recommendations: List[str] = Field(default_factory=list, description="Clinical recommendations")
    generated_at: datetime = Field(default_factory=datetime.now)
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")
    confidence_score: float = Field(default=0.0, description="Confidence in the summary")

class ClinicalTrial(BaseModel):
    """
    Represents a clinical trial with detailed information.
    
    This model structures clinical trial data for analysis and comparison.
    """
    trial_id: str = Field(..., description="Trial identifier")
    title: str = Field(..., description="Trial title")
    phase: str = Field(..., description="Trial phase")
    status: str = Field(..., description="Trial status")
    conditions: List[str] = Field(default_factory=list, description="Medical conditions studied")
    interventions: List[str] = Field(default_factory=list, description="Interventions tested")
    locations: List[str] = Field(default_factory=list, description="Trial locations")
    enrollment: int = Field(default=0, description="Number of participants")
    start_date: Optional[datetime] = Field(None, description="Trial start date")
    completion_date: Optional[datetime] = Field(None, description="Trial completion date")
    results: Optional[str] = Field(None, description="Summary of results if available")
    sponsor: Optional[str] = Field(None, description="Trial sponsor")
    source: str = Field(..., description="Source database")

# ============================================================================
# Core Research Assistant Agent
# ============================================================================

class ResearchAssistantAgent:
    """
    Specialized AI agent for medical research assistance.
    
    This agent provides comprehensive research support including:
    - Literature search and retrieval from multiple medical databases
    - Automatic summarization of research papers
    - Evidence synthesis and grading
    - Clinical trial analysis
    - Drug information retrieval
    - Guideline extraction
    
    The agent uses transformer models for natural language understanding and
    implements sophisticated retrieval and ranking algorithms.
    """
    
    def __init__(
        self,
        model_cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_gpu: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the research assistant agent with required models and connections.
        
        Args:
            model_cache_dir: Directory to cache downloaded models
            device: Device to run models on (cuda/cpu)
            use_gpu: Whether to use GPU if available
            config: Additional configuration parameters
        """
        logger.info("Initializing Research Assistant Agent")
        
        # Set up device for model inference
        if device is None:
            self.device = torch.device(
                "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Set up model cache directory
        self.model_cache_dir = model_cache_dir or str(
            Path.home() / ".cache" / "aegismedbot" / "research_models"
        )
        Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = config or self._load_default_config()
        
        # Initialize models (lazy loading to save memory)
        self._summarizer_model = None
        self._embedding_model = None
        self._relevance_model = None
        self._tokenizer = None
        
        # Initialize connection pools for external APIs
        self.session = None
        self.api_keys = self._load_api_keys()
        
        # Initialize caches
        self.query_cache = {}
        self.paper_cache = {}
        self.embedding_cache = {}
        
        # Statistics and metrics
        self.metrics = defaultdict(list)
        
        logger.info("Research Assistant Agent initialized successfully")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """
        Load default configuration settings for the agent.
        
        Returns:
            Dictionary containing default configuration values
        """
        return {
            "summarization_model": "facebook/bart-large-cnn",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "relevance_model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            "max_input_length": 1024,
            "max_summary_length": 250,
            "min_summary_length": 50,
            "batch_size": 8,
            "cache_ttl": 3600,  # 1 hour
            "max_concurrent_requests": 5,
            "request_timeout": 30,
            "retry_attempts": 3,
            "relevance_threshold": 0.6,
            "enable_caching": True,
            "enable_logging": True,
            "api_endpoints": {
                "pubmed": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                "pubmed_summary": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                "clinicaltrials": "https://clinicaltrials.gov/api/query/study_fields",
                "cochrane": "https://www.cochranelibrary.com/cdsr/reviews"
            }
        }
    
    def _load_api_keys(self) -> Dict[str, str]:
        """
        Load API keys for external services from environment or config file.
        
        Returns:
            Dictionary mapping service names to API keys
        """
        import os
        api_keys = {}
        
        # Try to load from environment variables
        pubmed_key = os.getenv("PUBMED_API_KEY")
        if pubmed_key:
            api_keys["pubmed"] = pubmed_key
        
        clinicaltrials_key = os.getenv("CLINICALTRIALS_API_KEY")
        if clinicaltrials_key:
            api_keys["clinicaltrials"] = clinicaltrials_key
        
        return api_keys
    
    async def _ensure_session(self):
        """
        Ensure an aiohttp session exists for making HTTP requests.
        Creates a new session if one doesn't exist.
        """
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(
                total=self.config["request_timeout"],
                connect=10
            )
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def _load_summarizer_model(self):
        """
        Lazy load the summarization model.
        This saves memory by only loading when needed.
        """
        if self._summarizer_model is None:
            logger.info("Loading summarization model...")
            model_name = self.config["summarization_model"]
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.model_cache_dir
            )
            
            # Load model
            self._summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=self.model_cache_dir
            ).to(self.device)
            
            # Set to evaluation mode
            self._summarizer_model.eval()
            
            logger.info("Summarization model loaded successfully")
    
    async def _load_embedding_model(self):
        """
        Lazy load the embedding model for semantic search.
        """
        if self._embedding_model is None:
            logger.info("Loading embedding model...")
            model_name = self.config["embedding_model"]
            
            self._embedding_model = SentenceTransformer(
                model_name,
                device=str(self.device),
                cache_folder=self.model_cache_dir
            )
            
            logger.info("Embedding model loaded successfully")
    
    async def _load_relevance_model(self):
        """
        Lazy load the relevance scoring model.
        """
        if self._relevance_model is None:
            logger.info("Loading relevance model...")
            model_name = self.config["relevance_model"]
            
            self._relevance_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.model_cache_dir
            )
            
            self._relevance_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=self.model_cache_dir
            ).to(self.device)
            
            self._relevance_model.eval()
            
            logger.info("Relevance model loaded successfully")
    
    def _get_cache_key(self, query: str, query_type: str) -> str:
        """
        Generate a cache key for a query.
        
        Args:
            query: The query string
            query_type: Type of query
            
        Returns:
            SHA256 hash of the normalized query
        """
        normalized = f"{query_type}:{query.lower().strip()}"
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[Any]:
        """
        Check if a query result is in the cache and still valid.
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            Cached value if valid, None otherwise
        """
        if not self.config["enable_caching"]:
            return None
        
        cached = self.query_cache.get(cache_key)
        if cached:
            timestamp, value = cached
            age = (datetime.now() - timestamp).total_seconds()
            if age < self.config["cache_ttl"]:
                return value
        
        return None
    
    def _update_cache(self, cache_key: str, value: Any):
        """
        Update the cache with a new value.
        
        Args:
            cache_key: Cache key
            value: Value to cache
        """
        if self.config["enable_caching"]:
            self.query_cache[cache_key] = (datetime.now(), value)
    
    async def process_query(
        self,
        query: ResearchQuery
    ) -> ResearchSummary:
        """
        Main entry point for processing research queries.
        
        This method orchestrates the entire research process:
        1. Check cache for existing results
        2. Search relevant databases based on query type
        3. Retrieve and rank papers
        4. Generate summary and extract key findings
        5. Grade evidence and formulate recommendations
        
        Args:
            query: Structured research query
            
        Returns:
            Comprehensive research summary with findings and recommendations
        """
        start_time = datetime.now()
        logger.info(f"Processing research query: {query.query_text[:100]}...")
        
        # Check cache first
        cache_key = self._get_cache_key(query.query_text, query.query_type)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            logger.info("Returning cached result")
            cached_result.processing_time_ms = 0
            return cached_result
        
        try:
            # Route to appropriate handler based on query type
            if query.query_type == "clinical_trial":
                papers = await self._search_clinical_trials(query)
            elif query.query_type == "drug_info":
                papers = await self._search_drug_information(query)
            elif query.query_type == "guideline":
                papers = await self._search_guidelines(query)
            else:
                papers = await self._search_literature(query)
            
            # Rank papers by relevance
            ranked_papers = await self._rank_papers_by_relevance(
                papers,
                query.query_text
            )
            
            # Filter by relevance threshold
            relevant_papers = [
                p for p in ranked_papers
                if p.relevance_score >= query.relevance_threshold
            ]
            
            # Generate summary and extract findings
            if relevant_papers:
                summary = await self._generate_research_summary(
                    relevant_papers[:5],  # Use top 5 papers for summary
                    query.query_text
                )
                
                key_findings = await self._extract_key_findings(relevant_papers)
                evidence_level = self._grade_evidence(relevant_papers)
                recommendations = await self._generate_recommendations(
                    relevant_papers,
                    query.query_text
                )
                limitations = self._identify_limitations(relevant_papers)
            else:
                # Handle case with no relevant papers
                summary = f"No relevant research found for: {query.query_text}"
                key_findings = []
                evidence_level = "insufficient"
                recommendations = ["Consider broadening search terms"]
                limitations = ["No papers met the relevance threshold"]
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create research summary
            result = ResearchSummary(
                query=query.query_text,
                summary=summary,
                key_findings=key_findings[:10],  # Limit to top 10 findings
                evidence_level=evidence_level,
                sources=relevant_papers[:query.max_results],
                limitations=limitations,
                recommendations=recommendations,
                generated_at=datetime.now(),
                processing_time_ms=processing_time,
                confidence_score=self._calculate_confidence(relevant_papers)
            )
            
            # Cache the result
            self._update_cache(cache_key, result)
            
            # Update metrics
            self.metrics["queries_processed"].append(1)
            self.metrics["processing_times"].append(processing_time)
            
            logger.info(f"Query processed in {processing_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _search_literature(
        self,
        query: ResearchQuery
    ) -> List[ResearchPaper]:
        """
        Search medical literature databases for relevant papers.
        
        Args:
            query: Research query with search parameters
            
        Returns:
            List of research papers from various sources
        """
        await self._ensure_session()
        
        papers = []
        search_tasks = []
        
        # Search each requested source in parallel
        for source in query.sources:
            if source == "pubmed":
                task = self._search_pubmed(query)
            elif source == "clinicaltrials":
                task = self._search_clinical_trials_db(query)
            elif source == "cochrane":
                task = self._search_cochrane(query)
            else:
                logger.warning(f"Unknown source: {source}")
                continue
            
            search_tasks.append(task)
        
        # Execute all searches concurrently
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Search failed: {str(result)}")
            elif isinstance(result, list):
                papers.extend(result)
        
        # Remove duplicates based on DOI or PMID
        papers = self._deduplicate_papers(papers)
        
        logger.info(f"Found {len(papers)} unique papers")
        return papers
    
    async def _search_pubmed(self, query: ResearchQuery) -> List[ResearchPaper]:
        """
        Search PubMed database using E-utilities API.
        
        Args:
            query: Research query parameters
            
        Returns:
            List of papers from PubMed
        """
        logger.info(f"Searching PubMed for: {query.query_text}")
        
        try:
            # Build search parameters
            params = {
                "db": "pubmed",
                "term": query.query_text,
                "retmax": query.max_results * 2,  # Get more for ranking
                "retmode": "json",
                "usehistory": "y",
                "sort": "relevance"
            }
            
            # Add API key if available
            if "pubmed" in self.api_keys:
                params["api_key"] = self.api_keys["pubmed"]
            
            # Search for IDs
            search_url = self.config["api_endpoints"]["pubmed"]
            async with self.session.get(search_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"PubMed search failed: {response.status}")
                    return []
                
                search_data = await response.json()
                
            # Extract IDs
            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []
            
            # Fetch summaries
            summary_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "json"
            }
            
            summary_url = self.config["api_endpoints"]["pubmed_summary"]
            async with self.session.get(summary_url, params=summary_params) as response:
                if response.status != 200:
                    logger.error(f"PubMed summary fetch failed: {response.status}")
                    return []
                
                summary_data = await response.json()
            
            # Parse results
            papers = []
            uid_list = summary_data.get("result", {}).get("uids", [])
            
            for uid in uid_list:
                article = summary_data["result"].get(uid, {})
                
                # Extract authors
                authors = []
                for author in article.get("authors", []):
                    name = author.get("name", "")
                    if name:
                        authors.append(name)
                
                # Create paper object
                paper = ResearchPaper(
                    title=article.get("title", "No title"),
                    authors=authors,
                    abstract=article.get("abstract", "No abstract available"),
                    journal=article.get("fulljournalname", ""),
                    publication_date=self._parse_pubmed_date(article.get("pubdate", "")),
                    doi=article.get("elocationid", "").replace("doi: ", ""),
                    pmid=uid,
                    source="PubMed",
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                    keywords=article.get("keywords", []),
                    mesh_terms=article.get("meshterms", []),
                    citation_count=int(article.get("pmcrefcount", 0))
                )
                
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {str(e)}")
            return []
    
    async def _search_clinical_trials_db(
        self,
        query: ResearchQuery
    ) -> List[ClinicalTrial]:
        """
        Search ClinicalTrials.gov database.
        
        Args:
            query: Research query parameters
            
        Returns:
            List of clinical trials matching the query
        """
        logger.info(f"Searching ClinicalTrials.gov for: {query.query_text}")
        
        try:
            # Build search parameters
            params = {
                "expr": query.query_text,
                "fmt": "json",
                "max_rnk": query.max_results * 2,
                "fields": [
                    "NCTId",
                    "BriefTitle",
                    "OfficialTitle",
                    "Phase",
                    "OverallStatus",
                    "Condition",
                    "InterventionName",
                    "LocationFacility",
                    "EnrollmentCount",
                    "StartDate",
                    "CompletionDate",
                    "ResultsReference"
                ]
            }
            
            url = self.config["api_endpoints"]["clinicaltrials"]
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"ClinicalTrials search failed: {response.status}")
                    return []
                
                data = await response.json()
            
            # Parse results
            trials = []
            studies = data.get("StudyFieldsResponse", {}).get("StudyFields", [])
            
            for study in studies:
                trial = ClinicalTrial(
                    trial_id=study.get("NCTId", [""])[0],
                    title=study.get("BriefTitle", [""])[0],
                    phase=study.get("Phase", [""])[0],
                    status=study.get("OverallStatus", [""])[0],
                    conditions=study.get("Condition", []),
                    interventions=study.get("InterventionName", []),
                    locations=study.get("LocationFacility", []),
                    enrollment=int(study.get("EnrollmentCount", [0])[0]),
                    start_date=self._parse_date(study.get("StartDate", [""])[0]),
                    completion_date=self._parse_date(study.get("CompletionDate", [""])[0]),
                    source="ClinicalTrials.gov"
                )
                trials.append(trial)
            
            return trials
            
        except Exception as e:
            logger.error(f"Error searching ClinicalTrials: {str(e)}")
            return []
    
    async def _search_cochrane(self, query: ResearchQuery) -> List[ResearchPaper]:
        """
        Search Cochrane Library for systematic reviews.
        
        Args:
            query: Research query parameters
            
        Returns:
            List of Cochrane reviews
        """
        # Simplified implementation - in production, use proper API
        logger.info(f"Searching Cochrane Library for: {query.query_text}")
        
        # Placeholder for actual Cochrane API integration
        # Would implement similar to PubMed search
        return []
    
    async def _search_drug_information(
        self,
        query: ResearchQuery
    ) -> List[ResearchPaper]:
        """
        Specialized search for drug information from multiple sources.
        
        Args:
            query: Research query focused on drug information
            
        Returns:
            List of papers with drug information
        """
        # Enhance query with drug-specific terms
        enhanced_query = f"{query.query_text} drug medication pharmacology"
        drug_query = ResearchQuery(
            query_text=enhanced_query,
            query_type="literature_review",
            sources=["pubmed"],
            max_results=query.max_results
        )
        
        return await self._search_literature(drug_query)
    
    async def _search_guidelines(self, query: ResearchQuery) -> List[ResearchPaper]:
        """
        Specialized search for clinical guidelines.
        
        Args:
            query: Research query for guidelines
            
        Returns:
            List of clinical guidelines
        """
        # Enhance query with guideline-specific terms
        enhanced_query = f"{query.query_text} clinical practice guideline recommendation"
        guideline_query = ResearchQuery(
            query_text=enhanced_query,
            query_type="literature_review",
            sources=["pubmed"],
            max_results=query.max_results
        )
        
        return await self._search_literature(guideline_query)
    
    async def _rank_papers_by_relevance(
        self,
        papers: List[ResearchPaper],
        query: str
    ) -> List[ResearchPaper]:
        """
        Rank papers by relevance to the query using semantic similarity.
        
        Args:
            papers: List of papers to rank
            query: Original query text
            
        Returns:
            Papers sorted by relevance score
        """
        if not papers:
            return []
        
        # Load embedding model if needed
        await self._load_embedding_model()
        
        # Generate query embedding
        query_embedding = self._embedding_model.encode(query)
        
        # Generate paper embeddings (use cached if available)
        paper_embeddings = []
        for paper in papers:
            # Combine title and abstract for embedding
            text = f"{paper.title} {paper.abstract}"
            
            # Check cache
            cache_key = hashlib.sha256(text.encode()).hexdigest()
            if cache_key in self.embedding_cache:
                embedding = self.embedding_cache[cache_key]
            else:
                embedding = self._embedding_model.encode(text)
                self.embedding_cache[cache_key] = embedding
            
            paper_embeddings.append(embedding)
        
        # Convert to numpy arrays for efficient computation
        query_embedding = np.array(query_embedding)
        paper_embeddings = np.array(paper_embeddings)
        
        # Calculate cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        paper_norms = np.linalg.norm(paper_embeddings, axis=1)
        
        similarities = np.dot(paper_embeddings, query_embedding) / (paper_norms * query_norm)
        
        # Update papers with relevance scores
        for paper, score in zip(papers, similarities):
            paper.relevance_score = float(score)
        
        # Sort by relevance
        ranked_papers = sorted(papers, key=lambda x: x.relevance_score, reverse=True)
        
        return ranked_papers
    
    async def _generate_research_summary(
        self,
        papers: List[ResearchPaper],
        query: str
    ) -> str:
        """
        Generate a comprehensive summary of research findings.
        
        Args:
            papers: List of relevant papers
            query: Original query
            
        Returns:
            Synthesized summary text
        """
        if not papers:
            return "No papers available for summarization."
        
        # Load summarization model
        await self._load_summarizer_model()
        
        # Prepare input text by combining paper abstracts
        combined_text = f"Query: {query}\n\n"
        for i, paper in enumerate(papers[:3], 1):  # Use top 3 papers
            combined_text += f"Paper {i}: {paper.title}\n"
            combined_text += f"Abstract: {paper.abstract}\n\n"
        
        # Tokenize input
        inputs = self._tokenizer(
            combined_text,
            max_length=self.config["max_input_length"],
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self._summarizer_model.generate(
                inputs["input_ids"],
                max_length=self.config["max_summary_length"],
                min_length=self.config["min_summary_length"],
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        # Decode summary
        summary = self._tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return summary
    
    async def _extract_key_findings(self, papers: List[ResearchPaper]) -> List[str]:
        """
        Extract key findings from research papers using NLP techniques.
        
        Args:
            papers: List of research papers
            
        Returns:
            List of key findings as strings
        """
        findings = []
        
        # Simple rule-based extraction (in production, use NER and relation extraction)
        finding_patterns = [
            r"found that (.*?)[\.;]",
            r"demonstrated that (.*?)[\.;]",
            r"showed that (.*?)[\.;]",
            r"revealed that (.*?)[\.;]",
            r"concluded that (.*?)[\.;]",
            r"significant association (.*?)[\.;]",
            r"increased risk of (.*?)[\.;]",
            r"reduced risk of (.*?)[\.;]",
            r"effectiveness of (.*?)[\.;]"
        ]
        
        for paper in papers:
            text = f"{paper.title} {paper.abstract}"
            
            for pattern in finding_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    finding = match.strip()
                    if len(finding) > 20 and finding not in findings:
                        findings.append(finding)
        
        # Deduplicate and limit
        unique_findings = list(dict.fromkeys(findings))
        return unique_findings[:15]  # Return top 15 findings
    
    def _grade_evidence(self, papers: List[ResearchPaper]) -> str:
        """
        Grade the level of evidence based on paper characteristics.
        
        Args:
            papers: List of research papers
            
        Returns:
            Evidence level: high, moderate, low, or insufficient
        """
        if not papers:
            return "insufficient"
        
        # Calculate evidence score based on multiple factors
        scores = []
        
        for paper in papers:
            score = 0.0
            
            # Source quality
            if paper.source == "Cochrane":
                score += 0.3
            elif paper.source == "PubMed":
                score += 0.2
            
            # Journal reputation (simplified)
            high_impact_journals = [
                "NEJM", "Lancet", "JAMA", "BMJ", "Nature Medicine"
            ]
            if paper.journal and any(j in paper.journal for j in high_impact_journals):
                score += 0.2
            
            # Citation count
            if paper.citation_count > 100:
                score += 0.2
            elif paper.citation_count > 50:
                score += 0.1
            
            # Recency
            if paper.publication_date:
                years_old = (datetime.now() - paper.publication_date).days / 365
                if years_old < 3:
                    score += 0.2
                elif years_old < 5:
                    score += 0.1
            
            scores.append(min(score, 1.0))
        
        # Calculate average evidence score
        avg_score = sum(scores) / len(scores)
        
        # Determine evidence level
        if avg_score >= 0.7:
            return "high"
        elif avg_score >= 0.4:
            return "moderate"
        else:
            return "low"
    
    async def _generate_recommendations(
        self,
        papers: List[ResearchPaper],
        query: str
    ) -> List[str]:
        """
        Generate clinical recommendations based on research findings.
        
        Args:
            papers: List of research papers
            query: Original query
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Extract actionable statements
        recommendation_patterns = [
            r"recommend (.*?)[\.;]",
            r"should be considered (.*?)[\.;]",
            r"may benefit from (.*?)[\.;]",
            r"suggest that (.*?)[\.;]",
            r"indicated for (.*?)[\.;]"
        ]
        
        for paper in papers:
            text = f"{paper.title} {paper.abstract}"
            
            for pattern in recommendation_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    recommendation = f"Based on {paper.source}: {match.strip()}"
                    if len(recommendation) > 30 and recommendation not in recommendations:
                        recommendations.append(recommendation)
        
        # If no specific recommendations found, generate general ones
        if not recommendations and papers:
            recommendations.append(
                f"Consider reviewing the provided literature for "
                f"detailed recommendations on {query}"
            )
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _identify_limitations(self, papers: List[ResearchPaper]) -> List[str]:
        """
        Identify limitations in the available evidence.
        
        Args:
            papers: List of research papers
            
        Returns:
            List of limitations
        """
        limitations = []
        
        if not papers:
            limitations.append("No relevant papers found")
            return limitations
        
        # Check sample sizes
        # (Would need actual sample size data from papers)
        
        # Check publication dates
        old_papers = 0
        for paper in papers:
            if paper.publication_date:
                years_old = (datetime.now() - paper.publication_date).days / 365
                if years_old > 5:
                    old_papers += 1
        
        if old_papers > len(papers) / 2:
            limitations.append("Evidence may be outdated (>5 years old)")
        
        # Check source diversity
        sources = set(p.source for p in papers)
        if len(sources) < 2:
            limitations.append(f"Evidence primarily from single source: {sources.pop()}")
        
        # General limitation
        limitations.append("This summary is based on available abstracts and may not reflect full papers")
        
        return limitations
    
    def _calculate_confidence(self, papers: List[ResearchPaper]) -> float:
        """
        Calculate confidence score for the research summary.
        
        Args:
            papers: List of papers used
            
        Returns:
            Confidence score between 0 and 1
        """
        if not papers:
            return 0.0
        
        # Factors affecting confidence
        num_papers = len(papers)
        avg_relevance = sum(p.relevance_score for p in papers) / num_papers
        evidence_level_score = {
            "high": 1.0,
            "moderate": 0.6,
            "low": 0.3,
            "insufficient": 0.0
        }
        
        evidence_score = evidence_level_score.get(self._grade_evidence(papers), 0.0)
        
        # Calculate weighted confidence
        confidence = (
            0.3 * min(num_papers / 10, 1.0) +  # Paper quantity
            0.4 * avg_relevance +                # Paper relevance
            0.3 * evidence_score                  # Evidence quality
        )
        
        return min(confidence, 1.0)
    
    def _deduplicate_papers(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """
        Remove duplicate papers based on DOI or PMID.
        
        Args:
            papers: List of papers possibly containing duplicates
            
        Returns:
            Deduplicated list of papers
        """
        unique_papers = {}
        
        for paper in papers:
            # Use DOI or PMID as unique identifier
            if paper.doi:
                key = f"doi:{paper.doi}"
            elif paper.pmid:
                key = f"pmid:{paper.pmid}"
            else:
                # Fallback to title hash
                key = f"title:{hashlib.md5(paper.title.encode()).hexdigest()}"
            
            if key not in unique_papers:
                unique_papers[key] = paper
        
        return list(unique_papers.values())
    
    def _parse_pubmed_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse PubMed date format into datetime object.
        
        Args:
            date_str: Date string from PubMed
            
        Returns:
            Parsed datetime or None
        """
        try:
            # Common PubMed date formats
            formats = [
                "%Y %b %d",
                "%Y %b",
                "%Y",
                "%b %d, %Y",
                "%Y-%m-%d"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            return None
            
        except Exception:
            return None
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse various date formats into datetime.
        
        Args:
            date_str: Date string
            
        Returns:
            Parsed datetime or None
        """
        try:
            # Try common formats
            for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%b %d, %Y", "%d %b %Y"]:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            return None
            
        except Exception:
            return None
    
    async def close(self):
        """
        Clean up resources used by the agent.
        Should be called when the agent is no longer needed.
        """
        logger.info("Closing Research Assistant Agent")
        
        if self.session and not self.session.closed:
            await self.session.close()
        
        # Clear caches
        self.query_cache.clear()
        self.paper_cache.clear()
        self.embedding_cache.clear()
        
        logger.info("Research Assistant Agent closed")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the agent.
        
        Returns:
            Dictionary with performance statistics
        """
        metrics = {
            "queries_processed": len(self.metrics["queries_processed"]),
            "avg_processing_time_ms": np.mean(self.metrics["processing_times"]) if self.metrics["processing_times"] else 0,
            "cache_size": len(self.query_cache),
            "embedding_cache_size": len(self.embedding_cache),
            "papers_cached": len(self.paper_cache)
        }
        
        return metrics
    
    def __del__(self):
        """
        Destructor to ensure resources are cleaned up.
        """
        if hasattr(self, 'session') and self.session and not self.session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except:
                pass

# ============================================================================
# Utility Functions
# ============================================================================

def create_research_agent(
    config_path: Optional[str] = None,
    use_gpu: bool = True
) -> ResearchAssistantAgent:
    """
    Factory function to create and configure a research assistant agent.
    
    Args:
        config_path: Path to configuration file (JSON)
        use_gpu: Whether to use GPU if available
        
    Returns:
        Configured ResearchAssistantAgent instance
    """
    config = None
    
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
    
    return ResearchAssistantAgent(
        use_gpu=use_gpu,
        config=config
    )

# ============================================================================
# Example Usage
# ============================================================================

async def example_usage():
    """
    Example demonstrating how to use the research assistant agent.
    """
    # Create agent
    agent = ResearchAssistantAgent(use_gpu=False)
    
    try:
        # Create a research query
        query = ResearchQuery(
            query_text="Efficacy of mRNA vaccines for COVID-19 in elderly patients",
            query_type="literature_review",
            sources=["pubmed"],
            max_results=5,
            relevance_threshold=0.6
        )
        
        # Process query
        result = await agent.process_query(query)
        
        # Print results
        print(f"Query: {result.query}")
        print(f"Summary: {result.summary}")
        print(f"Evidence Level: {result.evidence_level}")
        print(f"Confidence: {result.confidence_score:.2f}")
        
        print("\nKey Findings:")
        for finding in result.key_findings:
            print(f"  - {finding}")
        
        print("\nRecommendations:")
        for rec in result.recommendations:
            print(f"  - {rec}")
        
        print(f"\nProcessing Time: {result.processing_time_ms:.2f}ms")
        
    finally:
        # Clean up
        await agent.close()

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())