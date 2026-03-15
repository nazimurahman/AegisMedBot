"""
Literature Retrieval Engine for Medical Research
================================================

This module handles searching and retrieving medical literature from various sources:
- PubMed/MEDLINE via Entrez API
- arXiv for preprints
- Internal document repositories
- CrossRef for citation data
- Semantic Scholar for enhanced metadata

The retriever implements caching, rate limiting, and intelligent query expansion
to provide comprehensive search results while respecting API limits.

Author: AegisMedBot Team
Version: 1.0.0
"""

import asyncio
import aiohttp
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from urllib.parse import quote_plus, urlencode

import numpy as np
from redis import asyncio as aioredis
import pickle

logger = logging.getLogger(__name__)

class SearchSource(Enum):
    """
    Enumeration of supported literature sources.
    
    Each source has specific API requirements and rate limits.
    """
    PUBMED = "pubmed"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    CROSSREF = "crossref"
    INTERNAL = "internal"
    ALL = "all"
    
    def get_api_url(self) -> str:
        """Get base API URL for the source."""
        urls = {
            SearchSource.PUBMED: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            SearchSource.ARXIV: "http://export.arxiv.org/api",
            SearchSource.SEMANTIC_SCHOLAR: "https://api.semanticscholar.org/v1",
            SearchSource.CROSSREF: "https://api.crossref.org",
            SearchSource.INTERNAL: None,  # Internal database
        }
        return urls.get(self, "")

class SearchResult:
    """
    Represents a single paper or article from search results.
    
    This class standardizes results from different sources into a common format
    for consistent processing by downstream components.
    """
    
    def __init__(
        self,
        title: str,
        abstract: str,
        authors: List[str],
        journal: str,
        year: int,
        doi: Optional[str] = None,
        pmid: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        url: Optional[str] = None,
        citations: int = 0,
        keywords: List[str] = None,
        full_text: Optional[str] = None,
        source: SearchSource = SearchSource.PUBMED,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize a search result with paper metadata.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            authors: List of author names
            journal: Journal or publication venue
            year: Publication year
            doi: Digital Object Identifier
            pmid: PubMed ID
            arxiv_id: arXiv identifier
            url: URL to paper
            citations: Citation count
            keywords: List of keywords or MeSH terms
            full_text: Full paper text (if available)
            source: Source database
            metadata: Additional source-specific metadata
        """
        self.title = title
        self.abstract = abstract
        self.authors = authors
        self.journal = journal
        self.year = year
        self.doi = doi
        self.pmid = pmid
        self.arxiv_id = arxiv_id
        self.url = url
        self.citations = citations
        self.keywords = keywords or []
        self.full_text = full_text
        self.source = source
        self.metadata = metadata or {}
        
        # Generate unique ID for this paper
        self.id = self._generate_id()
        
        # Timestamp for caching
        self.retrieved_at = datetime.now()
        
    def _generate_id(self) -> str:
        """
        Generate a unique identifier for the paper.
        
        Uses DOI if available, otherwise creates a hash from title and authors.
        """
        if self.doi:
            return f"doi:{self.doi}"
        elif self.pmid:
            return f"pmid:{self.pmid}"
        elif self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        else:
            # Create hash from title and first author
            content = f"{self.title}_{self.authors[0] if self.authors else ''}"
            return f"hash:{hashlib.md5(content.encode()).hexdigest()[:12]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'title': self.title,
            'abstract': self.abstract,
            'authors': self.authors,
            'journal': self.journal,
            'year': self.year,
            'doi': self.doi,
            'pmid': self.pmid,
            'arxiv_id': self.arxiv_id,
            'url': self.url,
            'citations': self.citations,
            'keywords': self.keywords,
            'source': self.source.value if self.source else None,
            'metadata': self.metadata,
            'retrieved_at': self.retrieved_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        """Create instance from dictionary."""
        result = cls(
            title=data['title'],
            abstract=data['abstract'],
            authors=data['authors'],
            journal=data['journal'],
            year=data['year'],
            doi=data.get('doi'),
            pmid=data.get('pmid'),
            arxiv_id=data.get('arxiv_id'),
            url=data.get('url'),
            citations=data.get('citations', 0),
            keywords=data.get('keywords', []),
            source=SearchSource(data['source']) if data.get('source') else None,
            metadata=data.get('metadata', {})
        )
        if 'retrieved_at' in data:
            result.retrieved_at = datetime.fromisoformat(data['retrieved_at'])
        return result
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"SearchResult(title='{self.title[:50]}...', year={self.year}, citations={self.citations})"

class RateLimiter:
    """
    Rate limiter for API calls to prevent exceeding service limits.
    
    Implements token bucket algorithm for smooth rate limiting across
    multiple API sources.
    """
    
    def __init__(self, max_calls: int, period: float = 60.0):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds (default: 60 seconds)
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []  # List of timestamps of recent calls
        
    async def acquire(self) -> float:
        """
        Acquire permission to make an API call.
        
        Returns:
            Wait time in seconds before call can be made (0 if ready)
        """
        now = time.time()
        
        # Remove old calls outside the period
        self.calls = [t for t in self.calls if now - t < self.period]
        
        if len(self.calls) < self.max_calls:
            # We can make the call now
            self.calls.append(now)
            return 0.0
        else:
            # Need to wait
            wait_time = self.calls[0] + self.period - now
            return max(0.0, wait_time)
    
    async def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        wait_time = await self.acquire()
        if wait_time > 0:
            logger.debug("Rate limiting: waiting %.2f seconds", wait_time)
            await asyncio.sleep(wait_time)

class LiteratureRetriever:
    """
    Main literature retrieval engine.
    
    This class handles searching multiple medical literature databases,
    caching results, and managing API rate limits.
    """
    
    def __init__(
        self,
        cache_ttl: int = 3600,  # Cache results for 1 hour by default
        max_papers_per_query: int = 50,
        use_cache: bool = True,
        redis_url: Optional[str] = None,
        api_keys: Optional[Dict[str, str]] = None
    ):
        """
        Initialize literature retriever.
        
        Args:
            cache_ttl: Time-to-live for cached results in seconds
            max_papers_per_query: Maximum papers to return per query
            use_cache: Whether to use Redis caching
            redis_url: Redis connection URL (required if use_cache=True)
            api_keys: Dictionary of API keys for external services
        """
        self.cache_ttl = cache_ttl
        self.max_papers_per_query = max_papers_per_query
        self.use_cache = use_cache
        self.api_keys = api_keys or {}
        
        # Initialize Redis client for caching
        self.redis = None
        if use_cache and redis_url:
            try:
                self.redis = aioredis.from_url(
                    redis_url,
                    decode_responses=False,  # We'll handle encoding ourselves
                    encoding=None
                )
                logger.info("Connected to Redis cache at %s", redis_url)
            except Exception as e:
                logger.error("Failed to connect to Redis: %s", str(e))
                self.redis = None
                self.use_cache = False
        
        # Rate limiters for different sources
        self.rate_limiters = {
            SearchSource.PUBMED: RateLimiter(max_calls=3, period=1.0),  # 3 per second
            SearchSource.ARXIV: RateLimiter(max_calls=10, period=1.0),  # 10 per second
            SearchSource.SEMANTIC_SCHOLAR: RateLimiter(max_calls=5, period=1.0),  # 5 per second
            SearchSource.CROSSREF: RateLimiter(max_calls=10, period=1.0),  # 10 per second
        }
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session exists and return it."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    'User-Agent': 'AegisMedBot/1.0 (mailto:contact@example.com)'
                }
            )
        return self.session
    
    async def close(self):
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def search(
        self,
        query: str,
        sources: List[SearchSource] = None,
        max_results: int = 20,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        authors: Optional[List[str]] = None,
        journals: Optional[List[str]] = None,
        sort_by: str = "relevance",  # relevance, date, citations
        use_cache: bool = True
    ) -> List[SearchResult]:
        """
        Search for medical literature across specified sources.
        
        This is the main entry point for literature search. It handles:
        1. Query expansion for better results
        2. Caching to avoid repeated API calls
        3. Parallel searching across multiple sources
        4. Result deduplication and ranking
        
        Args:
            query: Search query string
            sources: List of sources to search (default: all)
            max_results: Maximum number of results to return
            year_from: Filter results from this year
            year_to: Filter results to this year
            authors: Filter by author names
            journals: Filter by journal names
            sort_by: Sorting method
            use_cache: Whether to use cached results
            
        Returns:
            List of SearchResult objects, sorted by relevance
        """
        start_time = time.time()
        
        # Default to all sources if none specified
        if sources is None:
            sources = [src for src in SearchSource if src != SearchSource.ALL]
        
        # Expand query for better results
        expanded_query = await self._expand_query(query)
        logger.info("Searching for: '%s' (expanded from: '%s')", expanded_query, query)
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            expanded_query, sources, year_from, year_to, authors, journals
        )
        
        # Try to get from cache
        if use_cache and self.use_cache and self.redis:
            cached = await self._get_from_cache(cache_key)
            if cached:
                logger.info("Cache hit for query: %s", query)
                return self._filter_and_sort(
                    cached, max_results, year_from, year_to, authors, journals, sort_by
                )
        
        # Search all sources in parallel
        search_tasks = []
        for source in sources:
            task = self._search_source(
                source, expanded_query, max_results // len(sources) + 10
            )
            search_tasks.append(task)
        
        # Wait for all searches to complete
        results_lists = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Combine and deduplicate results
        all_results = []
        seen_ids = set()
        
        for results in results_lists:
            if isinstance(results, Exception):
                logger.error("Error searching source: %s", str(results))
                continue
            
            for result in results:
                if result.id not in seen_ids:
                    seen_ids.add(result.id)
                    all_results.append(result)
        
        # Apply filters
        filtered_results = self._filter_results(
            all_results, year_from, year_to, authors, journals
        )
        
        # Sort results
        sorted_results = self._sort_results(filtered_results, sort_by)
        
        # Cache results
        if use_cache and self.use_cache and self.redis:
            await self._save_to_cache(cache_key, sorted_results)
        
        elapsed = time.time() - start_time
        logger.info("Search completed in %.2f seconds, found %d results", 
                   elapsed, len(sorted_results))
        
        # Return limited number of results
        return sorted_results[:max_results]
    
    async def _search_source(
        self,
        source: SearchSource,
        query: str,
        max_results: int
    ) -> List[SearchResult]:
        """
        Search a single source.
        
        This dispatches to source-specific search methods.
        """
        # Apply rate limiting
        limiter = self.rate_limiters.get(source)
        if limiter:
            await limiter.wait_if_needed()
        
        # Dispatch to appropriate search method
        if source == SearchSource.PUBMED:
            return await self._search_pubmed(query, max_results)
        elif source == SearchSource.ARXIV:
            return await self._search_arxiv(query, max_results)
        elif source == SearchSource.SEMANTIC_SCHOLAR:
            return await self._search_semantic_scholar(query, max_results)
        elif source == SearchSource.CROSSREF:
            return await self._search_crossref(query, max_results)
        elif source == SearchSource.INTERNAL:
            return await self._search_internal(query, max_results)
        else:
            logger.warning("Unsupported source: %s", source)
            return []
    
    async def _search_pubmed(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Search PubMed via NCBI E-utilities API.
        
        PubMed is the primary source for biomedical literature with over
        30 million citations. This implementation uses the Entrez API.
        """
        session = await self._ensure_session()
        results = []
        
        try:
            # Step 1: Search for IDs
            base_url = SearchSource.PUBMED.get_api_url()
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'sort': 'relevance',
                'api_key': self.api_keys.get('pubmed', '')
            }
            
            # Remove empty API key
            if not search_params['api_key']:
                del search_params['api_key']
            
            search_url = f"{base_url}/esearch.fcgi"
            async with session.get(search_url, params=search_params) as response:
                if response.status != 200:
                    logger.error("PubMed search failed: %d", response.status)
                    return []
                
                data = await response.json()
                id_list = data.get('esearchresult', {}).get('idlist', [])
                
                if not id_list:
                    return []
            
            # Step 2: Fetch details for each ID
            # We fetch in batches to avoid URL length limits
            batch_size = 50
            for i in range(0, len(id_list), batch_size):
                batch_ids = id_list[i:i+batch_size]
                
                fetch_params = {
                    'db': 'pubmed',
                    'id': ','.join(batch_ids),
                    'retmode': 'xml',
                    'api_key': self.api_keys.get('pubmed', '')
                }
                
                if not fetch_params['api_key']:
                    del fetch_params['api_key']
                
                fetch_url = f"{base_url}/efetch.fcgi"
                async with session.get(fetch_url, params=fetch_params) as response:
                    if response.status != 200:
                        continue
                    
                    # Parse XML response
                    xml_text = await response.text()
                    batch_results = self._parse_pubmed_xml(xml_text)
                    results.extend(batch_results)
        
        except Exception as e:
            logger.error("Error searching PubMed: %s", str(e))
        
        return results
    
    def _parse_pubmed_xml(self, xml_text: str) -> List[SearchResult]:
        """
        Parse PubMed XML response into SearchResult objects.
        
        This is a simplified parser - in production, use a proper XML library
        like lxml or xml.etree.ElementTree.
        """
        results = []
        
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_text)
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    # Extract basic metadata
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else "No title"
                    
                    abstract_elem = article.find('.//AbstractText')
                    abstract = abstract_elem.text if abstract_elem is not None else ""
                    
                    # Extract authors
                    authors = []
                    for author in article.findall('.//Author'):
                        last = author.find('LastName')
                        fore = author.find('ForeName')
                        if last is not None and fore is not None:
                            authors.append(f"{fore.text} {last.text}")
                        elif last is not None:
                            authors.append(last.text)
                    
                    # Extract journal
                    journal_elem = article.find('.//Title')
                    journal = journal_elem.text if journal_elem is not None else "Unknown"
                    
                    # Extract year
                    year_elem = article.find('.//PubDate/Year')
                    year = int(year_elem.text) if year_elem is not None else 0
                    
                    # Extract PMID
                    pmid_elem = article.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else None
                    
                    # Extract DOI
                    doi_elem = article.find(".//ELocationID[@EIdType='doi']")
                    doi = doi_elem.text if doi_elem is not None else None
                    
                    # Create result
                    result = SearchResult(
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        journal=journal,
                        year=year,
                        doi=doi,
                        pmid=pmid,
                        source=SearchSource.PUBMED
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.debug("Error parsing PubMed article: %s", str(e))
                    continue
                    
        except Exception as e:
            logger.error("Error parsing PubMed XML: %s", str(e))
        
        return results
    
    async def _search_arxiv(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Search arXiv for preprints.
        
        arXiv is important for accessing cutting-edge research before
        formal publication.
        """
        session = await self._ensure_session()
        results = []
        
        try:
            base_url = SearchSource.ARXIV.get_api_url()
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            url = f"{base_url}/query"
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error("arXiv search failed: %d", response.status)
                    return []
                
                # Parse Atom feed
                import xml.etree.ElementTree as ET
                text = await response.text()
                root = ET.fromstring(text)
                
                # Define namespaces
                ns = {
                    'atom': 'http://www.w3.org/2005/Atom',
                    'arxiv': 'http://arxiv.org/schemas/atom'
                }
                
                for entry in root.findall('.//atom:entry', ns):
                    try:
                        title = entry.find('atom:title', ns).text
                        summary = entry.find('atom:summary', ns).text
                        
                        # Extract authors
                        authors = []
                        for author in entry.findall('atom:author', ns):
                            name = author.find('atom:name', ns)
                            if name is not None:
                                authors.append(name.text)
                        
                        # Extract arXiv ID
                        id_elem = entry.find('atom:id', ns)
                        arxiv_id = id_elem.text.split('/')[-1] if id_elem is not None else None
                        
                        # Extract DOI if present
                        doi_elem = entry.find('arxiv:doi', ns)
                        doi = doi_elem.text if doi_elem is not None else None
                        
                        # Extract published date
                        published = entry.find('atom:published', ns)
                        year = int(published.text[:4]) if published is not None else 0
                        
                        result = SearchResult(
                            title=title,
                            abstract=summary,
                            authors=authors,
                            journal="arXiv Preprint",
                            year=year,
                            doi=doi,
                            arxiv_id=arxiv_id,
                            url=f"https://arxiv.org/abs/{arxiv_id}",
                            source=SearchSource.ARXIV
                        )
                        results.append(result)
                        
                    except Exception as e:
                        logger.debug("Error parsing arXiv entry: %s", str(e))
                        continue
                        
        except Exception as e:
            logger.error("Error searching arXiv: %s", str(e))
        
        return results
    
    async def _search_semantic_scholar(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Search Semantic Scholar for enhanced paper metadata including citations.
        
        Semantic Scholar provides citation counts and influence metrics.
        """
        session = await self._ensure_session()
        results = []
        
        try:
            base_url = SearchSource.SEMANTIC_SCHOLAR.get_api_url()
            params = {
                'query': query,
                'limit': max_results,
                'fields': 'title,abstract,authors,venue,year,doi,url,citationCount'
            }
            
            url = f"{base_url}/paper/search"
            headers = {}
            
            # Add API key if available
            api_key = self.api_keys.get('semantic_scholar')
            if api_key:
                headers['x-api-key'] = api_key
            
            async with session.get(url, params=params, headers=headers) as response:
                if response.status != 200:
                    logger.error("Semantic Scholar search failed: %d", response.status)
                    return []
                
                data = await response.json()
                
                for paper in data.get('data', []):
                    try:
                        # Extract authors
                        authors = [a.get('name', '') for a in paper.get('authors', [])]
                        
                        result = SearchResult(
                            title=paper.get('title', 'No title'),
                            abstract=paper.get('abstract', ''),
                            authors=authors,
                            journal=paper.get('venue', 'Unknown'),
                            year=paper.get('year', 0),
                            doi=paper.get('doi'),
                            url=paper.get('url'),
                            citations=paper.get('citationCount', 0),
                            source=SearchSource.SEMANTIC_SCHOLAR
                        )
                        results.append(result)
                        
                    except Exception as e:
                        logger.debug("Error parsing Semantic Scholar paper: %s", str(e))
                        continue
                        
        except Exception as e:
            logger.error("Error searching Semantic Scholar: %s", str(e))
        
        return results
    
    async def _search_crossref(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Search CrossRef for comprehensive DOI-based metadata.
        
        CrossRef is excellent for finding papers by DOI and getting
        accurate citation information.
        """
        session = await self._ensure_session()
        results = []
        
        try:
            base_url = SearchSource.CROSSREF.get_api_url()
            params = {
                'query': query,
                'rows': max_results,
                'sort': 'relevance'
            }
            
            url = f"{base_url}/works"
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error("CrossRef search failed: %d", response.status)
                    return []
                
                data = await response.json()
                items = data.get('message', {}).get('items', [])
                
                for item in items:
                    try:
                        # Extract authors
                        authors = []
                        for author in item.get('author', []):
                            given = author.get('given', '')
                            family = author.get('family', '')
                            if given and family:
                                authors.append(f"{given} {family}")
                            elif family:
                                authors.append(family)
                        
                        # Extract year from published date
                        year = 0
                        published = item.get('published-print', {}).get('date-parts', [[]])[0]
                        if published:
                            year = published[0]
                        
                        result = SearchResult(
                            title=item.get('title', ['No title'])[0],
                            abstract=item.get('abstract', ''),
                            authors=authors,
                            journal=item.get('container-title', ['Unknown'])[0],
                            year=year,
                            doi=item.get('DOI'),
                            url=item.get('URL'),
                            source=SearchSource.CROSSREF
                        )
                        results.append(result)
                        
                    except Exception as e:
                        logger.debug("Error parsing CrossRef item: %s", str(e))
                        continue
                        
        except Exception as e:
            logger.error("Error searching CrossRef: %s", str(e))
        
        return results
    
    async def _search_internal(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Search internal document database.
        
        This could be a local Elasticsearch instance or vector database
        containing hospital-specific documents and protocols.
        """
        # Placeholder for internal search implementation
        # In production, this would query your internal document store
        return []
    
    async def _expand_query(self, query: str) -> str:
        """
        Expand medical query with synonyms and related terms.
        
        Uses medical ontologies like UMLS or MeSH to improve search recall.
        """
        # Simple expansion - in production, use medical thesaurus
        expansions = {
            'cancer': 'cancer tumor malignant carcinoma neoplasm',
            'heart attack': 'heart attack myocardial infarction MI',
            'diabetes': 'diabetes mellitus diabetic',
            'hypertension': 'hypertension high blood pressure',
            'pneumonia': 'pneumonia lung infection',
            'stroke': 'stroke cerebrovascular accident CVA',
            'alzheimer': 'alzheimer disease dementia',
            'arthritis': 'arthritis joint inflammation',
            'depression': 'depression major depressive disorder MDD',
            'anxiety': 'anxiety anxious disorder'
        }
        
        query_lower = query.lower()
        for term, expansion in expansions.items():
            if term in query_lower:
                # Replace the term with expanded version
                query = query.replace(term, expansion)
                break
        
        return query
    
    def _generate_cache_key(
        self,
        query: str,
        sources: List[SearchSource],
        year_from: Optional[int],
        year_to: Optional[int],
        authors: Optional[List[str]],
        journals: Optional[List[str]]
    ) -> str:
        """
        Generate a unique cache key for a search query.
        
        The key incorporates all search parameters to ensure cache uniqueness.
        """
        components = [
            query,
            ','.join(sorted([s.value for s in sources])),
            str(year_from or ''),
            str(year_to or ''),
            ','.join(sorted(authors or [])),
            ','.join(sorted(journals or []))
        ]
        
        key_string = '|'.join(components)
        hash_obj = hashlib.sha256(key_string.encode())
        return f"search:{hash_obj.hexdigest()}"
    
    async def _get_from_cache(self, cache_key: str) -> Optional[List[SearchResult]]:
        """
        Retrieve search results from Redis cache.
        """
        if not self.redis:
            return None
        
        try:
            data = await self.redis.get(cache_key)
            if data:
                # Deserialize pickled data
                results_data = pickle.loads(data)
                return [SearchResult.from_dict(r) for r in results_data]
        except Exception as e:
            logger.error("Error reading from cache: %s", str(e))
        
        return None
    
    async def _save_to_cache(self, cache_key: str, results: List[SearchResult]):
        """
        Save search results to Redis cache.
        """
        if not self.redis:
            return
        
        try:
            # Convert to serializable format
            results_data = [r.to_dict() for r in results]
            # Pickle for efficient storage
            data = pickle.dumps(results_data)
            await self.redis.setex(cache_key, self.cache_ttl, data)
            logger.debug("Cached %d results with key: %s", len(results), cache_key)
        except Exception as e:
            logger.error("Error saving to cache: %s", str(e))
    
    def _filter_results(
        self,
        results: List[SearchResult],
        year_from: Optional[int],
        year_to: Optional[int],
        authors: Optional[List[str]],
        journals: Optional[List[str]]
    ) -> List[SearchResult]:
        """
        Apply filters to search results.
        """
        filtered = []
        
        for result in results:
            # Year filter
            if year_from and result.year < year_from:
                continue
            if year_to and result.year > year_to:
                continue
            
            # Author filter
            if authors:
                author_match = False
                for author in authors:
                    if any(author.lower() in a.lower() for a in result.authors):
                        author_match = True
                        break
                if not author_match:
                    continue
            
            # Journal filter
            if journals:
                journal_match = False
                for journal in journals:
                    if journal.lower() in result.journal.lower():
                        journal_match = True
                        break
                if not journal_match:
                    continue
            
            filtered.append(result)
        
        return filtered
    
    def _sort_results(
        self,
        results: List[SearchResult],
        sort_by: str
    ) -> List[SearchResult]:
        """
        Sort results according to specified criterion.
        """
        if sort_by == "date":
            return sorted(results, key=lambda x: x.year, reverse=True)
        elif sort_by == "citations":
            return sorted(results, key=lambda x: x.citations, reverse=True)
        else:  # relevance - keep original order (assumed to be relevance-sorted from APIs)
            return results
    
    def _filter_and_sort(
        self,
        results: List[SearchResult],
        max_results: int,
        year_from: Optional[int],
        year_to: Optional[int],
        authors: Optional[List[str]],
        journals: Optional[List[str]],
        sort_by: str
    ) -> List[SearchResult]:
        """
        Combined filtering and sorting for cached results.
        """
        filtered = self._filter_results(results, year_from, year_to, authors, journals)
        sorted_results = self._sort_results(filtered, sort_by)
        return sorted_results[:max_results]

# Export public interface
__all__ = [
    'LiteratureRetriever',
    'SearchSource',
    'SearchResult',
    'RateLimiter'
]