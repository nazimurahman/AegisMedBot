"""
Medical Retriever Tool
This module provides functionality for retrieving medical information from various sources
including PubMed, clinical guidelines databases, and internal knowledge bases.
It implements RAG (Retrieval-Augmented Generation) for evidence-based responses.
"""

import asyncio  # For concurrent API calls
from typing import List, Dict, Any, Optional, Tuple
import aiohttp  # Asynchronous HTTP client for API calls
import json  # For parsing JSON responses
import logging  # For structured logging
from datetime import datetime, timedelta  # For date handling and caching
import hashlib  # For generating cache keys
import pickle  # For serializing cache data

# Import vector store for embeddings and similarity search
from ...rag_system.vector_store.qdrant_manager import QdrantManager
from ...rag_system.vector_store.embeddings import MedicalEmbeddingGenerator

logger = logging.getLogger(__name__)


class MedicalRetriever:
    """
    Retrieves medical information from various sources using RAG architecture.
    
    This tool integrates multiple medical knowledge sources:
    - PubMed/Medline for peer-reviewed literature
    - Clinical guidelines from professional societies
    - Hospital-specific protocols and policies
    - Medical textbooks and reference materials
    - Drug databases for pharmaceutical information
    
    It uses vector embeddings for semantic search and maintains a cache
    for frequently accessed information.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the medical retriever with configuration and connections.
        
        Args:
            config: Configuration dictionary containing API keys, endpoints,
                   and cache settings
        """
        self.config = config or {}
        
        # Initialize vector database connection for semantic search
        # Qdrant is used for storing and retrieving document embeddings
        self.vector_db = QdrantManager(
            host=self.config.get("qdrant_host", "localhost"),
            port=self.config.get("qdrant_port", 6333),
            collection_name=self.config.get("collection_name", "medical_knowledge"),
            vector_size=self.config.get("embedding_dim", 768)
        )
        
        # Initialize embedding generator for converting text to vectors
        self.embedding_generator = MedicalEmbeddingGenerator(
            model_name=self.config.get("embedding_model", 
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"),
            device=self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Define available data sources with their query methods
        # Each source has a function that implements the actual API call
        self.sources = {
            "pubmed": self._query_pubmed,           # PubMed/MEDLINE
            "guidelines": self._query_guidelines,    # Clinical guidelines
            "internal": self._query_internal_kb,      # Internal knowledge base
            "drugbank": self._query_drugbank,         # Drug information
            "uptodate": self._query_uptodate          # UpToDate clinical reference
        }
        
        # Initialize cache for reducing redundant API calls
        # Cache structure: {query_hash: (timestamp, results)}
        self.cache = {}
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # 1 hour default
        
        # Set up API credentials if provided
        self.pubmed_api_key = self.config.get("pubmed_api_key")
        self.drugbank_api_key = self.config.get("drugbank_api_key")
        
        # Create a session for making HTTP requests
        self.session = None
        
        logger.info(f"MedicalRetriever initialized with sources: {list(self.sources.keys())}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create an aiohttp session for making HTTP requests.
        Sessions are reused for connection pooling and performance.
        
        Returns:
            aiohttp.ClientSession instance
        """
        if self.session is None or self.session.closed:
            # Configure timeout and headers for all requests
            timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
            headers = {
                "User-Agent": "AegisMedBot/1.0 (medical research bot)",
                "Accept": "application/json"
            }
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self.session
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant medical information from multiple sources.
        
        This method:
        1. Checks cache for recent identical queries
        2. Generates query embedding for semantic search
        3. Queries all configured sources in parallel
        4. Combines and deduplicates results
        5. Ranks by relevance score
        
        Args:
            query: The search query string
            top_k: Maximum number of results to return
            filter_conditions: Optional filters (e.g., {"source": "pubmed", "year": 2023})
            use_cache: Whether to use cached results
            
        Returns:
            List of document dictionaries with content and metadata
        """
        # Generate cache key if caching is enabled
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(query, filter_conditions)
            cached_result = self._check_cache(cache_key)
            if cached_result:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_result[:top_k]
        
        # Generate query embedding for semantic search
        # This is used for vector similarity matching
        query_embedding = await self.embedding_generator.generate_query_embedding(
            query,
            use_expansion=True  # Expand query with medical terms for better retrieval
        )
        
        # First, search the vector database for similar documents
        vector_results = await self.vector_db.search(
            query_vector=query_embedding,
            top_k=top_k * 2,  # Get more results for re-ranking
            filter_conditions=filter_conditions,
            score_threshold=0.6  # Minimum similarity threshold
        )
        
        # Prepare to query external sources in parallel
        tasks = []
        for source_name, source_func in self.sources.items():
            # Skip if filter_conditions specify a particular source
            if filter_conditions and filter_conditions.get("source") and source_name != filter_conditions["source"]:
                continue
                
            # Add query task with timeout
            tasks.append(
                self._query_with_timeout(
                    source_func, 
                    query, 
                    top_k=top_k,
                    timeout=10  # 10 second timeout per source
                )
            )
        
        # Execute all source queries concurrently
        source_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results from all sources
        all_results = []
        
        # Add vector database results first (they have embeddings)
        all_results.extend(vector_results)
        
        # Add external source results
        for result in source_results:
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error querying source: {str(result)}")
        
        # Deduplicate results based on content similarity
        unique_results = self._deduplicate_results(all_results)
        
        # Re-rank results combining vector similarity and keyword relevance
        reranked_results = self._rerank_results(unique_results, query)
        
        # Cache the results if caching is enabled
        if use_cache and cache_key:
            self._cache_results(cache_key, reranked_results)
        
        # Return top_k results
        return reranked_results[:top_k]
    
    def _generate_cache_key(self, query: str, filters: Optional[Dict]) -> str:
        """
        Generate a unique cache key for a query and its filters.
        
        Args:
            query: The search query
            filters: Filter conditions
            
        Returns:
            MD5 hash string for cache lookup
        """
        # Create a string representation of the query and filters
        key_parts = [query]
        if filters:
            # Sort filters for consistent ordering
            key_parts.append(json.dumps(filters, sort_keys=True))
        
        key_string = "||".join(key_parts)
        
        # Generate MD5 hash for fixed-length key
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[List[Dict]]:
        """
        Check if cached results exist and are still valid.
        
        Args:
            cache_key: Cache lookup key
            
        Returns:
            Cached results if valid, None otherwise
        """
        if cache_key in self.cache:
            timestamp, results = self.cache[cache_key]
            # Check if cache entry is still fresh
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return results
            else:
                # Remove expired entry
                del self.cache[cache_key]
        return None
    
    def _cache_results(self, cache_key: str, results: List[Dict]):
        """
        Store results in cache with current timestamp.
        
        Args:
            cache_key: Cache key
            results: Results to cache
        """
        # Limit cache size to prevent memory issues
        if len(self.cache) > 1000:
            # Remove oldest entry (simple FIFO)
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k][0])
            del self.cache[oldest_key]
        
        self.cache[cache_key] = (datetime.now(), results)
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """
        Remove duplicate results based on content similarity.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Deduplicated list
        """
        if not results:
            return []
        
        unique = []
        seen_content = set()
        
        for result in results:
            # Create a signature for comparison
            content = result.get("content", "")[:200]  # First 200 chars
            title = result.get("title", "")
            source = result.get("source", "")
            
            signature = f"{title}|{source}|{content}"
            
            if signature not in seen_content:
                seen_content.add(signature)
                unique.append(result)
        
        logger.debug(f"Deduplicated {len(results)} results to {len(unique)} unique")
        return unique
    
    def _rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Re-rank results combining vector similarity and keyword relevance.
        
        Args:
            results: List of result dictionaries
            query: Original query for keyword matching
            
        Returns:
            Re-ranked results
        """
        if not results:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for result in results:
            # Start with existing score
            base_score = result.get("score", 0.5)
            
            # Boost score based on title relevance
            title = result.get("title", "").lower()
            title_matches = sum(1 for word in query_words if word in title)
            title_boost = min(0.2 * title_matches, 0.4)  # Max 0.4 boost
            
            # Boost score based on recency
            year = result.get("year")
            if year:
                current_year = datetime.now().year
                years_old = current_year - year
                recency_boost = max(0, 0.3 * (1 - years_old / 10))  # Linear decay over 10 years
            else:
                recency_boost = 0
            
            # Boost score based on source authority
            source = result.get("source", "").lower()
            authority_boost = {
                "pubmed": 0.2,
                "guidelines": 0.3,
                "uptodate": 0.25,
                "internal": 0.1,
                "drugbank": 0.2
            }.get(source, 0)
            
            # Calculate final score (cap at 1.0)
            final_score = min(base_score + title_boost + recency_boost + authority_boost, 1.0)
            result["score"] = final_score
        
        # Sort by final score
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)
    
    async def _query_with_timeout(self, func, *args, timeout: int = 10, **kwargs):
        """
        Execute a query function with a timeout to prevent hanging.
        
        Args:
            func: Async function to call
            timeout: Maximum time to wait in seconds
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Function result or empty list on timeout
        """
        try:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Query to {func.__name__} timed out after {timeout}s")
            return []
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return []
    
    async def _query_pubmed(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Query PubMed/MEDLINE for medical literature.
        
        Uses the Entrez Programming Utilities (E-utilities) API.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of article dictionaries
        """
        session = await self._get_session()
        
        # Base URL for PubMed E-utilities
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        # Step 1: Search for article IDs
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": top_k,
            "retmode": "json",
            "sort": "relevance"
        }
        
        # Add API key if available (for higher rate limits)
        if self.pubmed_api_key:
            search_params["api_key"] = self.pubmed_api_key
        
        try:
            # Perform search
            async with session.get(f"{base_url}esearch.fcgi", params=search_params) as response:
                if response.status != 200:
                    logger.error(f"PubMed search failed: {response.status}")
                    return []
                
                search_data = await response.json()
                ids = search_data.get("esearchresult", {}).get("idlist", [])
                
                if not ids:
                    return []
                
                # Step 2: Fetch article details
                fetch_params = {
                    "db": "pubmed",
                    "id": ",".join(ids),
                    "retmode": "xml",
                    "rettype": "abstract"
                }
                
                async with session.get(f"{base_url}efetch.fcgi", params=fetch_params) as fetch_response:
                    if fetch_response.status != 200:
                        return []
                    
                    # Parse XML response (simplified - in production use XML parser)
                    xml_text = await fetch_response.text()
                    
                    # This is a simplified parsing - in production, use proper XML parsing
                    results = []
                    for i, article_id in enumerate(ids):
                        results.append({
                            "source": "pubmed",
                            "id": article_id,
                            "title": f"Article {i+1} about {query[:30]}...",
                            "content": f"Abstract placeholder for article {article_id}",
                            "authors": ["Author names would appear here"],
                            "year": datetime.now().year,
                            "journal": "Medical Journal",
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/",
                            "score": 0.8 - (i * 0.05)  # Decreasing scores
                        })
                    
                    return results
                    
        except Exception as e:
            logger.error(f"PubMed query error: {str(e)}")
            return []
    
    async def _query_guidelines(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Query clinical guidelines database.
        
        This would integrate with sources like:
        - National Guideline Clearinghouse
        - Professional society guidelines (ACC, AHA, ASCO, etc.)
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of guideline dictionaries
        """
        # This is a simplified implementation
        # In production, this would call actual guideline APIs
        
        # Simulate API delay
        await asyncio.sleep(0.1)
        
        results = []
        condition = query.split()[-1] if query else "condition"
        
        for i in range(min(top_k, 3)):  # Return up to 3 guidelines
            results.append({
                "source": "guidelines",
                "title": f"Clinical Practice Guideline for {condition.title()}",
                "content": f"Recommendations for management of {condition} include diagnostic criteria, treatment algorithms, and follow-up schedules. "
                          f"First-line treatment typically involves lifestyle modifications and pharmacotherapy as indicated.",
                "organization": ["American College of Physicians", "American Heart Association", "WHO"][i % 3],
                "year": 2023 - i,
                "guideline_id": f"GUIDE-{i+1:04d}",
                "recommendations": [
                    "Perform diagnostic evaluation including relevant labs",
                    "Initiate treatment based on severity",
                    "Monitor response and adjust therapy"
                ],
                "evidence_level": "Strong",
                "score": 0.85 - (i * 0.1)
            })
        
        return results
    
    async def _query_internal_kb(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Query internal hospital knowledge base.
        
        This includes hospital-specific protocols, policies, and formularies.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of internal document dictionaries
        """
        # This would query a vector database of internal documents
        # For now, return simulated results
        
        await asyncio.sleep(0.05)  # Simulate fast internal query
        
        results = []
        protocols = [
            "Sepsis Management Protocol",
            "Acute Stroke Response",
            "Chest Pain Pathway",
            "Diabetic Ketoacidosis Protocol",
            "Post-operative Care Guidelines"
        ]
        
        for i, protocol in enumerate(protocols[:top_k]):
            if any(word in protocol.lower() for word in query.lower().split()):
                results.append({
                    "source": "internal",
                    "title": protocol,
                    "content": f"Hospital-approved protocol for {protocol}. Includes step-by-step procedures, medication dosing, and escalation criteria.",
                    "department": "Internal Medicine",
                    "version": "2.1",
                    "last_reviewed": "2024-01-15",
                    "approved_by": "Medical Executive Committee",
                    "score": 0.9 - (i * 0.05)
                })
        
        return results
    
    async def _query_drugbank(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Query DrugBank for pharmaceutical information.
        
        Args:
            query: Search query (drug name or class)
            top_k: Number of results to return
            
        Returns:
            List of drug information dictionaries
        """
        # This would call the DrugBank API
        # For now, return simulated results
        
        await asyncio.sleep(0.1)
        
        # Extract drug name from query
        drug_name = None
        common_drugs = ["metformin", "lisinopril", "atorvastatin", "amlodipine", "omeprazole"]
        for drug in common_drugs:
            if drug in query.lower():
                drug_name = drug
                break
        
        if not drug_name:
            return []
        
        # Simulate drug information
        return [{
            "source": "drugbank",
            "drug_name": drug_name,
            "drug_class": "Antihypertensive" if drug_name == "lisinopril" else "Various",
            "indications": ["Hypertension", "Heart failure"],
            "contraindications": ["Pregnancy", "Renal impairment"],
            "side_effects": ["Cough", "Dizziness", "Headache"],
            "interactions": ["NSAIDs", "Potassium supplements"],
            "dosing": "Individualized based on response",
            "score": 0.88
        }]
    
    async def _query_uptodate(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Query UpToDate clinical reference.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of clinical topic summaries
        """
        # Simulate UpToDate-like responses
        await asyncio.sleep(0.1)
        
        return [{
            "source": "uptodate",
            "title": f"{query.title()} - Clinical Overview",
            "content": f"This topic provides an overview of {query} including epidemiology, etiology, clinical manifestations, diagnosis, and management.",
            "last_updated": "2024-02-01",
            "contributors": ["Dr. John Smith", "Dr. Jane Doe"],
            "sections": ["Background", "Diagnosis", "Treatment", "Prognosis"],
            "score": 0.82
        }]
    
    async def retrieve_guidelines(
        self,
        condition: str,
        patient_context: Optional[Dict[str, Any]] = None,
        include_recent_only: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve clinical guidelines for a specific condition.
        
        This method specializes in finding and synthesizing clinical guidelines
        from authoritative sources.
        
        Args:
            condition: Medical condition to find guidelines for
            patient_context: Optional patient data for personalization
            include_recent_only: Whether to filter for recent guidelines only
            
        Returns:
            Dictionary with synthesized guideline information
        """
        # Build search query
        query = f"clinical guidelines {condition}"
        if include_recent_only:
            query += " 2024 2023"
        
        # Retrieve relevant documents
        results = await self.retrieve(
            query,
            top_k=10,
            filter_conditions={"source": "guidelines"} if include_recent_only else None
        )
        
        if not results:
            return {
                "condition": condition,
                "summary": f"No specific guidelines found for {condition}. Please consult specialty society guidelines.",
                "recommendations": [],
                "sources": [],
                "confidence": 0.3,
                "year": datetime.now().year
            }
        
        # Extract and synthesize recommendations
        all_recommendations = []
        sources = []
        max_year = 0
        
        for result in results:
            # Collect recommendations
            recs = result.get("recommendations", [])
            if isinstance(recs, list):
                all_recommendations.extend(recs)
            
            # Track sources
            sources.append({
                "organization": result.get("organization", "Unknown"),
                "year": result.get("year", datetime.now().year),
                "title": result.get("title", "")
            })
            
            # Track latest year
            year = result.get("year", 0)
            if year > max_year:
                max_year = year
        
        # Deduplicate recommendations
        unique_recommendations = []
        seen = set()
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        # Generate summary
        summary = f"Based on guidelines from {', '.join([s['organization'] for s in sources[:3]])}: "
        summary += f"Management of {condition} involves "
        if unique_recommendations:
            summary += unique_recommendations[0][:100] + "..."
        else:
            summary += "individualized care based on severity and patient factors."
        
        # Personalize if patient context provided
        if patient_context:
            age = patient_context.get("age")
            if age and age > 65:
                summary += " Special considerations for elderly patients should be followed."
        
        return {
            "condition": condition,
            "summary": summary,
            "recommendations": unique_recommendations[:5],  # Top 5 recommendations
            "sources": sources,
            "confidence": results[0].get("score", 0.7),
            "year": max_year,
            "evidence_level": "Based on current guidelines"
        }
    
    async def retrieve_disease_info(self, disease: str) -> Dict[str, Any]:
        """
        Retrieve comprehensive information about a disease.
        
        This method aggregates information about symptoms, causes,
        diagnosis, treatment, and prognosis from multiple sources.
        
        Args:
            disease: Name of the disease
            
        Returns:
            Dictionary with structured disease information
        """
        # Retrieve information from multiple sources
        overview_results = await self.retrieve(f"{disease} overview", top_k=3)
        symptoms_results = await self.retrieve(f"{disease} symptoms clinical presentation", top_k=3)
        diagnosis_results = await self.retrieve(f"{disease} diagnosis diagnostic criteria", top_k=3)
        treatment_results = await self.retrieve(f"{disease} treatment management", top_k=3)
        prognosis_results = await self.retrieve(f"{disease} prognosis outcomes", top_k=2)
        
        # Extract and synthesize information
        disease_info = {
            "disease": disease,
            "overview": self._synthesize_overview(overview_results, disease),
            "symptoms": self._extract_symptoms(symptoms_results),
            "causes": self._extract_causes(overview_results),
            "diagnosis": self._extract_diagnostic_criteria(diagnosis_results),
            "treatment": self._extract_treatment_options(treatment_results),
            "prognosis": self._extract_prognosis(prognosis_results),
            "references": self._extract_references([
                *overview_results, *symptoms_results, 
                *diagnosis_results, *treatment_results
            ]),
            "confidence": 0.85
        }
        
        return disease_info
    
    def _synthesize_overview(self, results: List[Dict], disease: str) -> str:
        """Synthesize disease overview from multiple sources."""
        if not results:
            return f"{disease} is a medical condition that requires proper diagnosis and treatment."
        
        # Take the most comprehensive overview
        best_result = max(results, key=lambda x: len(x.get("content", "")))
        return best_result.get("content", f"Overview of {disease} not available.")
    
    def _extract_symptoms(self, results: List[Dict]) -> List[str]:
        """Extract symptoms from search results."""
        symptoms = set()
        
        for result in results:
            content = result.get("content", "").lower()
            
            # Look for symptom lists
            if "symptoms" in content or "presentation" in content:
                # Simple extraction - in production use NLP
                lines = content.split("\n")
                for line in lines:
                    if "symptom" in line or "sign" in line:
                        # Extract bullet points
                        if "•" in line or "-" in line:
                            parts = line.split("•" if "•" in line else "-")
                            for part in parts:
                                if part.strip() and len(part) < 100:
                                    symptoms.add(part.strip())
        
        # If no symptoms found, provide generic list
        if not symptoms:
            symptoms = [
                "Symptoms vary based on disease severity and individual factors",
                "Consult healthcare provider for proper evaluation"
            ]
        
        return list(symptoms)[:8]  # Limit to 8 symptoms
    
    def _extract_causes(self, results: List[Dict]) -> str:
        """Extract disease causes from results."""
        for result in results:
            content = result.get("content", "")
            if "caused by" in content.lower() or "etiology" in content.lower():
                # Return first 200 characters
                return content[:200] + "..."
        
        return "Causes may include genetic, environmental, and lifestyle factors."
    
    def _extract_diagnostic_criteria(self, results: List[Dict]) -> List[str]:
        """Extract diagnostic criteria from results."""
        criteria = set()
        
        for result in results:
            content = result.get("content", "")
            if "diagnosis" in content.lower() or "criteria" in content.lower():
                # Look for numbered or bulleted lists
                lines = content.split("\n")
                for line in lines:
                    if any(str(i) in line for i in range(1, 6)) or "•" in line:
                        criteria.add(line.strip())
        
        if not criteria:
            criteria = [
                "Clinical evaluation by healthcare provider",
                "Laboratory tests as indicated",
                "Imaging studies when appropriate"
            ]
        
        return list(criteria)[:5]
    
    def _extract_treatment_options(self, results: List[Dict]) -> List[str]:
        """Extract treatment options from results."""
        treatments = set()
        
        for result in results:
            content = result.get("content", "")
            if "treatment" in content.lower() or "management" in content.lower():
                # Look for treatment mentions
                lines = content.split("\n")
                for line in lines:
                    if any(term in line.lower() for term in ["medication", "therapy", "surgery", "lifestyle"]):
                        treatments.add(line.strip())
        
        if not treatments:
            treatments = [
                "Treatment depends on disease severity and patient factors",
                "Consult specialist for personalized treatment plan"
            ]
        
        return list(treatments)[:5]
    
    def _extract_prognosis(self, results: List[Dict]) -> str:
        """Extract prognosis information from results."""
        for result in results:
            content = result.get("content", "")
            if "prognosis" in content.lower() or "outcome" in content.lower():
                return content[:200] + "..."
        
        return "Prognosis varies based on early diagnosis, treatment adherence, and individual factors."
    
    def _extract_references(self, results: List[Dict]) -> List[str]:
        """Extract references from results."""
        references = set()
        
        for result in results:
            source = result.get("source", "Unknown")
            title = result.get("title", "")
            year = result.get("year", "")
            
            if source and title:
                references.add(f"{title} ({source}, {year})")
        
        return list(references)[:3]
    
    async def close(self):
        """Clean up resources when done."""
        if self.session and not self.session.closed:
            await self.session.close()