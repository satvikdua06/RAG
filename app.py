import os
import json
import logging
import requests
from bs4 import BeautifulSoup
import uuid
import time
import asyncio
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any, Union
from contextlib import contextmanager
from functools import wraps, lru_cache
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pymupdf  # PyMuPDF
import faiss
import numpy as np
import pandas as pd
import re
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== Enhanced Configuration ==========
@dataclass
class Config:
    # API Configuration
    GOOGLE_API_KEY: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    EMBEDDING_MODEL: str = 'models/embedding-001'
    GENERATIVE_MODEL: str = 'gemini-1.5-flash'
    
    # RAG Parameters
    TOP_K: int = 10
    CHUNK_WORDS: int = 150  # Increased for better context
    CHUNK_OVERLAP: int = 30  # Increased overlap
    CHAT_HISTORY_TURNS: int = 10  # Increased memory
    MODEL_DIMENSION: int = 768
    
    # Performance Settings
    MAX_WORKERS: int = 4
    CACHE_TTL: int = 3600  # 1 hour cache TTL
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    
    # Flask Settings
    SECRET_KEY: str = field(default_factory=lambda: os.getenv('SECRET_KEY', 'enhanced-secret-key'))
    UPLOAD_FOLDER: str = 'uploads'
    MAX_CONTENT_LENGTH: int = 256 * 1024 * 1024  # 256MB
    
    # File Processing
    SUPPORTED_TEXT_EXTENSIONS: tuple = ('.txt', '.md', '.rtf')
    SUPPORTED_DATA_EXTENSIONS: tuple = ('.csv', '.xlsx', '.xls')
    SUPPORTED_PDF_EXTENSIONS: tuple = ('.pdf',)
    
    def __post_init__(self):
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY must be set in environment variables")

config = Config()

# Configure Gemini API with retry mechanism
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def configure_gemini_api():
    try:
        genai.configure(api_key=config.GOOGLE_API_KEY)
        # Test the configuration
        test_model = genai.GenerativeModel(config.GENERATIVE_MODEL)
        logger.info("Google Gemini API configured and tested successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")
        raise

configure_gemini_api()

# ========== Enhanced Data Structures ==========
@dataclass
class ProcessingMetrics:
    start_time: float
    end_time: Optional[float] = None
    chunks_created: int = 0
    embedding_time: float = 0
    indexing_time: float = 0
    
    @property
    def total_time(self) -> float:
        return (self.end_time or time.time()) - self.start_time

@dataclass
class DataSource:
    file_name: str
    data_type: str
    index: faiss.Index
    chunks: List[str]
    dataset_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    file_size: Optional[int] = None
    metrics: Optional[ProcessingMetrics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'file_name': self.file_name,
            'data_type': self.data_type,
            'chunks_count': len(self.chunks),
            'created_at': self.created_at.isoformat(),
            'file_size': self.file_size,
            'dataset_id': self.dataset_id,
            'metadata': self.metadata
        }

@dataclass
class RAGSession:
    session_id: str
    tagged_sources: Dict[str, List[DataSource]] = field(default_factory=dict)
    chat_history: List[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    def update_activity(self):
        self.last_activity = datetime.now()
    
    def get_summary(self) -> Dict[str, Any]:
        total_sources = sum(len(sources) for sources in self.tagged_sources.values())
        return {
            'session_id': self.session_id,
            'total_tags': len(self.tagged_sources),
            'total_sources': total_sources,
            'chat_turns': len(self.chat_history),
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat()
        }

# ========== Enhanced Flask App Configuration ==========
app = Flask(__name__)
app.config.update(
    SECRET_KEY=config.SECRET_KEY,
    UPLOAD_FOLDER=config.UPLOAD_FOLDER,
    MAX_CONTENT_LENGTH=config.MAX_CONTENT_LENGTH,
    JSON_SORT_KEYS=False
)
CORS(app, resources={r"/api/*": {"origins": "*"}})

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ========== Enhanced Global Stores with Thread Safety ==========
class ThreadSafeDict(dict):
    def __init__(self):
        super().__init__()
        self._lock = threading.RLock()
    
    def __setitem__(self, key, value):
        with self._lock:
            super().__setitem__(key, value)
    
    def __getitem__(self, key):
        with self._lock:
            return super().__getitem__(key)
    
    def __delitem__(self, key):
        with self._lock:
            super().__delitem__(key)
    
    def get(self, key, default=None):
        with self._lock:
            return super().get(key, default)
    
    def pop(self, key, default=None):
        with self._lock:
            return super().pop(key, default)

sessions = ThreadSafeDict()
datasets = ThreadSafeDict()

# Thread pool for concurrent processing
executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

# ========== Enhanced Utility Functions ==========
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{func.__name__} completed in {end_time - start_time:.3f}s")
            return result
        except Exception as e:
            end_time = time.time()
            logger.error(f"{func.__name__} failed after {end_time - start_time:.3f}s: {e}")
            raise
    return wrapper

@contextmanager
def error_handler(operation_name: str):
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {operation_name}: {e}", exc_info=True)
        raise

def validate_file_type(filename: str) -> Tuple[bool, str]:
    """Enhanced file type validation"""
    if not filename:
        return False, "Empty filename"
    
    ext = os.path.splitext(filename.lower())[1]
    
    if ext in config.SUPPORTED_TEXT_EXTENSIONS:
        return True, "text"
    elif ext in config.SUPPORTED_DATA_EXTENSIONS:
        return True, "data"
    elif ext in config.SUPPORTED_PDF_EXTENSIONS:
        return True, "pdf"
    else:
        return False, f"Unsupported file type: {ext}"

def clean_and_validate_text(text: str, min_length: int = 10) -> str:
    """Enhanced text cleaning and validation"""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace and normalize
    cleaned = re.sub(r'\s+', ' ', text.strip())
    
    # Remove control characters except newlines and tabs
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)
    
    return cleaned if len(cleaned) >= min_length else ""

# ========== Enhanced Text Processing ==========
@timing_decorator
def scrape_and_process_url(url: str) -> Optional[str]:
    """Scrapes a URL and extracts the main text content."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
            
        # Get text, using a space as a separator and stripping whitespace
        text = soup.get_text(separator=' ', strip=True)
        
        return clean_and_validate_text(text, min_length=100)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
        return None

@timing_decorator
def enhanced_chunk_text(content: str, chunk_words: int = None, overlap: int = None) -> List[str]:
    """Enhanced text chunking with better sentence boundary detection"""
    chunk_words = chunk_words or config.CHUNK_WORDS
    overlap = overlap or config.CHUNK_OVERLAP
    
    if not content or not content.strip():
        return []
    
    # Split into sentences first for better chunk boundaries
    sentences = re.split(r'(?<=[.!?])\s+', content)
    
    words = []
    for sentence in sentences:
        sentence_words = sentence.split()
        words.extend(sentence_words)
    
    if not words:
        return []
    
    chunks = []
    for i in range(0, len(words), chunk_words - overlap):
        chunk_words_list = words[i:i + chunk_words]
        if len(chunk_words_list) < 10:  # Skip very short chunks
            continue
        chunk_text = ' '.join(chunk_words_list)
        cleaned_chunk = clean_and_validate_text(chunk_text)
        if cleaned_chunk:
            chunks.append(cleaned_chunk)
    
    logger.info(f"Created {len(chunks)} chunks from {len(words)} words")
    return chunks

@timing_decorator
def enhanced_dataframe_chunking(df: pd.DataFrame) -> List[str]:
    """Enhanced DataFrame to text chunks conversion"""
    if df.empty:
        return []
    
    chunks = []
    
    # Create column summary
    col_info = f"Dataset contains columns: {', '.join(df.columns)}"
    chunks.append(col_info)
    
    # Create row-wise chunks with better formatting
    for idx, row in df.iterrows():
        row_data = []
        for col, val in row.items():
            if pd.notna(val) and str(val).strip():
                # Handle different data types appropriately
                if isinstance(val, (int, float)):
                    row_data.append(f"{col}: {val}")
                else:
                    # Clean string values
                    clean_val = clean_and_validate_text(str(val))
                    if clean_val:
                        row_data.append(f"{col}: {clean_val}")
        
        if row_data:
            chunk_text = f"Row {idx + 1} - " + ", ".join(row_data)
            chunks.append(chunk_text)
    
    logger.info(f"Created {len(chunks)} chunks from DataFrame with {len(df)} rows")
    return chunks

# ========== Enhanced PDF Processing ==========
@timing_decorator
def enhanced_pdf_processing(filepath: str) -> Tuple[List[pd.DataFrame], str, Dict[str, Any]]:
    """Enhanced PDF processing with corrected table and header extraction."""
    metadata = {
        'total_pages': 0,
        'tables_extracted': 0,
        'text_pages': 0,
        'processing_errors': []
    }
    
    try:
        doc = pymupdf.open(filepath)
        metadata['total_pages'] = len(doc)
        
        all_dfs = []
        non_table_text = ""
        
        for page_num, page in enumerate(doc):
            try:
                table_objects = page.find_tables()
                table_bboxes = [pymupdf.Rect(t.bbox) for t in table_objects]
                
                for i, table in enumerate(table_objects):
                    try:
                        # CORRECTED: Use table.extract() for all data, then separate the header
                        table_data = table.extract()
                        
                        if table_data and len(table_data) > 1:
                            header = table_data[0]
                            body = table_data[1:]
                            
                            # Clean headers to remove None values
                            cleaned_headers = [f"Unnamed_Column_{j}" if h is None else str(h) for j, h in enumerate(header)]
                            
                            df = pd.DataFrame(body, columns=cleaned_headers)
                            df = df.dropna(how='all', axis=1).dropna(how='all', axis=0)
                            
                            if not df.empty:
                                all_dfs.append(df)
                                metadata['tables_extracted'] += 1
                    
                    except Exception as e:
                        error_msg = f"Error extracting table {i} on page {page_num}: {e}"
                        metadata['processing_errors'].append(error_msg)
                        logger.warning(error_msg)
                
                # Extract text blocks not covered by tables
                text_blocks = page.get_text("blocks")
                page_text = ""
                for block in text_blocks:
                    block_bbox = pymupdf.Rect(block[:4])
                    is_table_text = any(table_bbox.intersects(block_bbox) for table_bbox in table_bboxes)
                    
                    if not is_table_text:
                        block_text = clean_and_validate_text(block[4])
                        if block_text:
                            page_text += block_text + " "
                
                if page_text.strip():
                    non_table_text += page_text
                    metadata['text_pages'] += 1
            
            except Exception as e:
                error_msg = f"Error processing page {page_num}: {e}"
                metadata['processing_errors'].append(error_msg)
                logger.warning(error_msg)
        
        doc.close()
        
        logger.info(f"PDF processing complete: {metadata['tables_extracted']} tables, "
                    f"{metadata['text_pages']} text pages from {metadata['total_pages']} total pages")
        
        return all_dfs, non_table_text.strip(), metadata
    
    except Exception as e:
        logger.error(f"Fatal error processing PDF {filepath}: {e}")
        raise


# ========== Enhanced Embedding Functions ==========
@lru_cache(maxsize=1000)
def cached_embedding(text_hash: str, text: str, task: str = "RETRIEVAL_DOCUMENT") -> Optional[np.ndarray]:
    """Cached embedding generation to avoid recomputing identical texts"""
    return get_embeddings_batch_impl([text], task)

@retry(
    stop=stop_after_attempt(config.MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,))
)
def get_embeddings_batch_impl(texts: List[str], task: str = "RETRIEVAL_DOCUMENT") -> Optional[np.ndarray]:
    """Implementation with retry logic"""
    if not texts:
        return None
    
    try:
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return None
        
        result = genai.embed_content(
            model=config.EMBEDDING_MODEL,
            content=valid_texts,
            task_type=task
        )
        
        embeddings = np.array(result['embedding']).astype('float32')
        
        # Ensure proper shape for single text
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        logger.debug(f"Generated embeddings for {len(valid_texts)} texts, shape: {embeddings.shape}")
        return embeddings
    
    except Exception as e:
        logger.error(f"Error creating Gemini embeddings: {e}")
        raise

@timing_decorator
def get_embeddings_batch(texts: List[str], task: str = "RETRIEVAL_DOCUMENT") -> Optional[np.ndarray]:
    """Enhanced batch embedding with caching and error handling"""
    if not texts:
        return None
    
    # Process in smaller batches to avoid API limits
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            batch_embeddings = get_embeddings_batch_impl(batch, task)
            if batch_embeddings is not None:
                all_embeddings.append(batch_embeddings)
            else:
                logger.warning(f"Failed to get embeddings for batch {i//batch_size + 1}")
        except Exception as e:
            logger.error(f"Error processing embedding batch {i//batch_size + 1}: {e}")
            # Continue with other batches
            continue
    
    if not all_embeddings:
        return None
    
    return np.vstack(all_embeddings)

# ========== Enhanced Generation Functions ==========
@retry(
    stop=stop_after_attempt(config.MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,))
)
def generate_with_gemini(prompt: str, temperature: float = 0.0, max_output_tokens: int = 2048) -> str:
    """Enhanced Gemini generation with better error handling and configuration"""
    try:
        model = genai.GenerativeModel(config.GENERATIVE_MODEL)
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=0.8,
            top_k=40
        )
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        if not response.parts:
            logger.warning("Gemini response was blocked or empty")
            return "I apologize, but I couldn't generate a response due to safety restrictions. Please try rephrasing your question."
        
        return response.text.strip()
    
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        raise

# ========== Enhanced RAG Functions ==========
def intelligent_tag_extraction(query: str, available_tags: List[str]) -> Optional[str]:
    """Enhanced tag extraction using fuzzy matching and context analysis"""
    if not available_tags:
        return None
    
    query_lower = query.lower()
    
    # Direct quote matching (highest priority)
    for tag in available_tags:
        tag_lower = tag.lower()
        patterns = [
            f"'{tag_lower}'", f'"{tag_lower}"',
            f"tag {tag_lower}", f"file {tag_lower}",
            f"document {tag_lower}", f"data {tag_lower}"
        ]
        for pattern in patterns:
            if pattern in query_lower:
                logger.info(f"Found tag '{tag}' via direct pattern matching")
                return tag
    
    # Word boundary matching
    query_words = set(re.findall(r'\w+', query_lower))
    for tag in available_tags:
        tag_words = set(re.findall(r'\w+', tag.lower()))
        if tag_words.intersection(query_words):
            logger.info(f"Found tag '{tag}' via word matching")
            return tag
    
    # If only one tag available, use it
    if len(available_tags) == 1:
        logger.info(f"Using only available tag: '{available_tags[0]}'")
        return available_tags[0]
    
    return None


@timing_decorator
def enhanced_vector_search(query: str, sources: List[DataSource]) -> Optional[DataSource]:
    """Enhanced vector search with better ranking and fallback strategies"""
    if not sources:
        return None
    
    if len(sources) == 1:
        return sources[0]
    
    logger.info(f"Performing enhanced vector search across {len(sources)} sources")
    
    query_embedding = get_embeddings_batch([query], task="RETRIEVAL_QUERY")
    if query_embedding is None:
        logger.warning("Could not generate query embedding, using first source")
        return sources[0]
    
    scored_sources = []
    
    for source in sources:
        try:
            if source.index.ntotal == 0:
                continue
            
            # Get top-k results for better ranking
            k = min(5, source.index.ntotal)
            distances, indices = source.index.search(query_embedding, k)
            
            if distances.size > 0:
                # Use average of top results for more robust scoring
                avg_distance = np.mean(distances[0][:min(3, len(distances[0]))])
                scored_sources.append((source, avg_distance))
        
        except Exception as e:
            logger.warning(f"Error during vector search for source {source.file_name}: {e}")
            continue
    
    if not scored_sources:
        logger.warning("No valid vector search results, using first source")
        return sources[0]
    
    # Sort by distance (lower is better)
    scored_sources.sort(key=lambda x: x[1])
    best_source = scored_sources[0][0]
    
    logger.info(f"Vector search selected '{best_source.file_name}' "
                f"(distance: {scored_sources[0][1]:.4f})")
    
    return best_source


@timing_decorator
def intelligent_agent_routing(query: str, chat_history: List[Dict]) -> str:
    """Enhanced agent routing with better decision logic and bias toward pandas for data analysis"""
    
    # Strong indicators for pandas analysis
    strong_analysis_keywords = [
        'calculate', 'sum', 'average', 'mean', 'median', 'mode', 'count', 'total', 
        'maximum', 'minimum', 'max', 'min', 'statistics', 'stats', 'distribution', 
        'correlation', 'group', 'groupby', 'filter', 'sort', 'rank', 'top', 'bottom',
        'percentage', 'ratio', 'trend', 'comparison', 'compare', 'aggregate',
        'pivot', 'crosstab', 'unique', 'duplicates', 'null', 'missing',
        'histogram', 'plot', 'chart', 'visualize', 'graph'
    ]
    
    # Mathematical and statistical operations
    math_keywords = [
        'multiply', 'divide', 'subtract', 'add', 'plus', 'minus', 'times',
        'variance', 'std', 'deviation', 'quartile', 'percentile', 'quantile',
        'frequency', 'distribution', 'bins', 'range', 'spread'
    ]
    
    # Data manipulation keywords
    data_manipulation_keywords = [
        'merge', 'join', 'concat', 'combine', 'append', 'stack', 'unstack',
        'melt', 'pivot', 'transpose', 'reshape', 'transform', 'apply',
        'map', 'replace', 'drop', 'remove', 'select', 'subset'
    ]
    
    # Conditional and logical operations
    conditional_keywords = [
        'where', 'condition', 'if', 'greater', 'less', 'equal', 'between',
        'contains', 'startswith', 'endswith', 'match', 'regex', 'pattern'
    ]
    
    # Time series keywords
    time_keywords = [
        'date', 'time', 'year', 'month', 'day', 'hour', 'minute', 'second',
        'datetime', 'timestamp', 'period', 'frequency', 'resample', 'rolling'
    ]
    
    # Keywords that suggest simple information retrieval
    retrieval_keywords = [
        'what is', 'who is', 'where is', 'when was', 'how is', 'why is',
        'explain', 'describe', 'tell me about', 'information about',
        'details about', 'show me', 'find', 'search', 'lookup', 'definition',
        'meaning', 'purpose', 'background', 'history', 'overview', 'summary'
    ]
    
    # Content exploration keywords (more suitable for RAG)
    content_keywords = [
        'content', 'text', 'document', 'paragraph', 'section', 'chapter',
        'quote', 'mention', 'discuss', 'topic', 'subject', 'theme'
    ]
    
    query_lower = query.lower()
    
    # Calculate scores with different weights
    strong_analysis_score = sum(2 for keyword in strong_analysis_keywords if keyword in query_lower)
    math_score = sum(2 for keyword in math_keywords if keyword in query_lower)
    manipulation_score = sum(1.5 for keyword in data_manipulation_keywords if keyword in query_lower)
    conditional_score = sum(1.5 for keyword in conditional_keywords if keyword in query_lower)
    time_score = sum(1 for keyword in time_keywords if keyword in query_lower)
    
    retrieval_score = sum(1 for keyword in retrieval_keywords if keyword in query_lower)
    content_score = sum(1.5 for keyword in content_keywords if keyword in query_lower)
    
    # Look for numerical patterns that suggest calculations
    number_patterns = len(re.findall(r'\b\d+(?:\.\d+)?%?\b', query_lower))
    comparison_patterns = len(re.findall(r'\b(?:>|<|>=|<=|!=|==)\b', query_lower))
    
    # Aggregate analysis score
    total_analysis_score = (strong_analysis_score + math_score + manipulation_score + 
                           conditional_score + time_score + (number_patterns * 0.5) + 
                           (comparison_patterns * 1))
    
    # Aggregate retrieval score
    total_retrieval_score = retrieval_score + content_score
    
    # Consider recent chat history for context (with decay)
    if chat_history:
        recent_messages = chat_history[-6:]  # Last 6 messages
        for i, msg in enumerate(recent_messages):
            if msg.get('role') == 'user':
                msg_lower = msg.get('content', '').lower()
                # Apply decay factor (more recent messages have higher weight)
                decay_factor = (i + 1) / len(recent_messages) * 0.3
                
                total_analysis_score += decay_factor * sum(1 for keyword in strong_analysis_keywords + math_keywords if keyword in msg_lower)
                total_retrieval_score += decay_factor * sum(1 for keyword in retrieval_keywords + content_keywords if keyword in msg_lower)
    
    # Enhanced decision logic with lower threshold for pandas
    decision_threshold = 0.5  # Lower threshold makes it easier to choose pandas
    
    # Special cases that should definitely use pandas
    definite_pandas_phrases = [
        'how many', 'count of', 'number of', 'total of', 'sum of', 'average of',
        'calculate the', 'compute the', 'find the total', 'find the sum',
        'show me the count', 'what is the average', 'what is the total',
        'group by', 'grouped by', 'per category', 'by category', 'breakdown',
        'highest', 'lowest', 'most', 'least', 'largest', 'smallest'
    ]
    
    has_definite_pandas = any(phrase in query_lower for phrase in definite_pandas_phrases)
    
    # Special cases that should use RAG
    definite_rag_phrases = [
        'what does', 'what is the meaning', 'explain the concept', 'describe the',
        'tell me about', 'what are the details', 'give me information',
        'find information about', 'search for', 'look up'
    ]
    
    has_definite_rag = any(phrase in query_lower for phrase in definite_rag_phrases)
    
    # Decision logic
    if has_definite_pandas and not has_definite_rag:
        decision = 'pandas'
        reason = 'definite pandas phrase detected'
    elif has_definite_rag and not has_definite_pandas:
        decision = 'rag'
        reason = 'definite RAG phrase detected'
    elif total_analysis_score > total_retrieval_score + decision_threshold:
        decision = 'pandas'
        reason = f'analysis score ({total_analysis_score:.1f}) > retrieval score ({total_retrieval_score:.1f}) + threshold'
    elif total_retrieval_score > total_analysis_score + decision_threshold:
        decision = 'rag'
        reason = f'retrieval score ({total_retrieval_score:.1f}) > analysis score ({total_analysis_score:.1f}) + threshold'
    elif total_analysis_score > 0:  # If there's any analysis intent, prefer pandas
        decision = 'pandas'
        reason = 'has analysis keywords, defaulting to pandas'
    else:
        decision = 'rag'
        reason = 'no clear analysis intent, defaulting to RAG'
    
    print(f"Agent routing decision: {decision}")
    print(f"Reason: {reason}")
    print(f"Scores - Analysis: {total_analysis_score:.1f}, Retrieval: {total_retrieval_score:.1f}")
    print(f"Query: '{query[:100]}...'")
    
    return decision


@timing_decorator
def enhanced_pandas_agent(query: str, df, chat_history: List[Dict]) -> str:
    """Enhanced pandas agent with better code generation, error handling, and query understanding"""
    import pandas as pd
    import numpy as np
    import json
    from datetime import datetime
    
    # Analyze the query to understand intent better
    query_analysis = analyze_query_intent(query)
    
    # Get comprehensive dataset info
    dataset_info = get_comprehensive_dataset_info(df)
    
    # Build context-aware prompt
    code_gen_prompt = build_enhanced_prompt(query, query_analysis, dataset_info, chat_history)
    
    try:
        # Generate code with higher temperature for more creative solutions
        generated_code = generate_with_gemini(code_gen_prompt, temperature=0.15)
        
        # Clean and validate the generated code
        cleaned_code = clean_generated_code(generated_code)
        
        print(f"Generated pandas code:\n{cleaned_code}")
        
        # Execute with enhanced safety and error handling
        result = execute_pandas_code_safely(cleaned_code, df)
        
        # Format result for better presentation
        formatted_result = format_pandas_result(result, query_analysis)
        
        return formatted_result
        
    except Exception as e:
        print(f"Error in enhanced pandas agent: {e}")
        return handle_pandas_error(e, query)


def analyze_query_intent(query: str) -> Dict:
    """Analyze the query to understand what type of analysis is needed"""
    query_lower = query.lower()
    
    intent = {
        'type': 'general',
        'aggregation': False,
        'filtering': False,
        'grouping': False,
        'sorting': False,
        'mathematical': False,
        'statistical': False,
        'comparison': False,
        'temporal': False
    }
    
    # Detect different types of operations
    if any(word in query_lower for word in ['sum', 'total', 'count', 'average', 'mean', 'max', 'min']):
        intent['aggregation'] = True
        intent['type'] = 'aggregation'
    
    if any(word in query_lower for word in ['where', 'filter', 'only', 'excluding', 'including']):
        intent['filtering'] = True
    
    if any(word in query_lower for word in ['group', 'by category', 'per', 'each', 'breakdown']):
        intent['grouping'] = True
        intent['type'] = 'grouping'
    
    if any(word in query_lower for word in ['sort', 'order', 'rank', 'top', 'bottom', 'highest', 'lowest']):
        intent['sorting'] = True
    
    if any(word in query_lower for word in ['calculate', 'compute', '+', '-', '*', '/', 'multiply', 'divide']):
        intent['mathematical'] = True
    
    if any(word in query_lower for word in ['correlation', 'variance', 'std', 'percentile', 'distribution']):
        intent['statistical'] = True
        intent['type'] = 'statistical'
    
    if any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus', 'between']):
        intent['comparison'] = True
        intent['type'] = 'comparison'
    
    if any(word in query_lower for word in ['date', 'time', 'year', 'month', 'day', 'period']):
        intent['temporal'] = True
    
    return intent


def get_comprehensive_dataset_info(df) -> Dict:
    """Get comprehensive information about the dataset"""
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'null_counts': df.isnull().sum().to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'sample_values': {}
    }
    
    # Get sample values for each column
    for col in df.columns:
        try:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 10:
                info['sample_values'][col] = unique_vals.tolist()
            else:
                info['sample_values'][col] = unique_vals[:5].tolist() + ['...']
        except:
            info['sample_values'][col] = ['<complex_data>']
    
    # Get basic statistics for numeric columns
    if info['numeric_columns']:
        try:
            info['numeric_stats'] = df[info['numeric_columns']].describe().to_dict()
        except:
            info['numeric_stats'] = {}
    
    return info


def build_enhanced_prompt(query: str, query_analysis: Dict, dataset_info: Dict, chat_history: List[Dict]) -> str:
    """Build a comprehensive prompt for better code generation"""
    
    # Build context from chat history
    history_context = ""
    if chat_history:
        recent_history = chat_history[-4:]
        for msg in recent_history:
            if msg.get('role') == 'user':
                history_context += f"Previous question: {msg.get('content', '')[:100]}\n"
    
    # Build column guidance
    column_guidance = ""
    if dataset_info['numeric_columns']:
        column_guidance += f"Numeric columns (good for calculations): {', '.join(dataset_info['numeric_columns'])}\n"
    if dataset_info['categorical_columns']:
        column_guidance += f"Categorical columns (good for grouping): {', '.join(dataset_info['categorical_columns'])}\n"
    if dataset_info['datetime_columns']:
        column_guidance += f"DateTime columns (good for time analysis): {', '.join(dataset_info['datetime_columns'])}\n"
    
    # Build operation-specific guidance
    operation_guidance = ""
    if query_analysis['aggregation']:
        operation_guidance += "- Use aggregation functions like sum(), mean(), count(), max(), min()\n"
    if query_analysis['grouping']:
        operation_guidance += "- Use groupby() for categorical analysis\n"
    if query_analysis['filtering']:
        operation_guidance += "- Use boolean indexing or query() for filtering\n"
    if query_analysis['sorting']:
        operation_guidance += "- Use sort_values() for ordering results\n"
    
    prompt = f"""You are an expert Python data analyst. Analyze the dataset and answer the user's query with clean, efficient pandas code. Do not forget to use indentation.

DATASET INFORMATION:
- Shape: {dataset_info['shape']} (rows, columns)
- Columns: {', '.join(dataset_info['columns'])}
- Data types: {dataset_info['dtypes']}
- Missing values: {dataset_info['null_counts']}

{column_guidance}

SAMPLE DATA (first few unique values per column):
{json.dumps(dataset_info['sample_values'], indent=2, default=str)}

{history_context}

QUERY ANALYSIS:
- Type: {query_analysis['type']}
- Operations needed: {operation_guidance}

INSTRUCTIONS:
1. Use the pre-loaded DataFrame `df` (DO NOT redefine or reload it)
2. Write clean, efficient pandas code
3. Store your final answer in a variable called `result`
4. Handle missing values appropriately (mention if you're excluding them)
5. For large results, show only the most relevant parts (top 10-20 rows)
6. Add comments to explain complex operations
7. Use appropriate data types and avoid unnecessary conversions
8. For calculations, ensure you're using the right columns

CURRENT QUERY: "{query}"

Provide ONLY the Python code (no markdown formatting, no explanations outside comments):
"""
    
    return prompt


def clean_generated_code(generated_code: str) -> str:
    """Clean and validate generated code"""
    # Remove markdown formatting
    cleaned = generated_code.strip()
    for pattern in ['```python', '```', '`']:
        cleaned = cleaned.replace(pattern, '')
    
    # Remove common problematic patterns
    lines = cleaned.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines and comments at the start
        if not line or (line.startswith('#') and not cleaned_lines):
            continue
        # Skip problematic imports or redefinitions
        if any(skip_pattern in line.lower() for skip_pattern in [
            'import pandas', 'pd.read_', 'df = pd.', 'df=pd.'
        ]):
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def execute_pandas_code_safely(code: str, df) -> any:
    """Execute pandas code with safety measures and proper error handling"""
    import pandas as pd
    import numpy as np
    import re
    from datetime import datetime
    import signal
    import sys
    from io import StringIO
    
    # Create safe execution environment
    safe_globals = {
        'pd': pd,
        'np': np,
        're': re,
        'datetime': datetime,
        '__builtins__': {}  # Restrict built-ins for safety
    }
    
    # Create local scope with a copy of the dataframe
    local_scope = {'df': df.copy()}
    
    # Capture output for debugging
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        # Set execution timeout
        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out after 15 seconds")
        
        if hasattr(signal, 'SIGALRM'):  # Unix systems
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(15)
        
        # Execute the code
        exec(code, safe_globals, local_scope)
        
        # Get the result
        result = local_scope.get('result', 'No result variable found.')
        
        return result
        
    except TimeoutError:
        raise TimeoutError("Analysis took too long to complete")
    except Exception as e:
        # Provide more specific error information
        raise Exception(f"Execution error: {str(e)}")
    finally:
        # Cleanup
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        sys.stdout = old_stdout


def format_pandas_result(result, query_analysis: Dict) -> str:
    """Format the result for better presentation"""
    import pandas as pd
    
    if result is None:
        return "The analysis completed but returned no result."
    
    try:
        if isinstance(result, pd.DataFrame):
            if len(result) == 0:
                return "The analysis returned an empty result."
            elif len(result) > 20:
                summary = f"Analysis Result (showing first 20 of {len(result)} rows):\n"
                summary += result.head(20).to_string(max_colwidth=50)
                if len(result) > 20:
                    summary += f"\n\n... and {len(result) - 20} more rows."
            else:
                summary = f"Analysis Result:\n{result.to_string(max_colwidth=50)}"
            return summary
            
        elif isinstance(result, pd.Series):
            if len(result) > 20:
                summary = f"Analysis Result (showing first 20 of {len(result)} items):\n"
                summary += result.head(20).to_string()
                if len(result) > 20:
                    summary += f"\n\n... and {len(result) - 20} more items."
            else:
                summary = f"Analysis Result:\n{result.to_string()}"
            return summary
            
        elif isinstance(result, (list, tuple)) and len(result) > 20:
            return f"Analysis Result (showing first 20 of {len(result)} items):\n{result[:20]}\n\n... and {len(result) - 20} more items."
            
        elif isinstance(result, dict) and len(result) > 10:
            items = list(result.items())[:10]
            summary = f"Analysis Result (showing first 10 of {len(result)} items):\n"
            for k, v in items:
                summary += f"{k}: {v}\n"
            if len(result) > 10:
                summary += f"... and {len(result) - 10} more items."
            return summary
        else:
            return f"Analysis Result:\n{result}"
            
    except Exception as e:
        return f"Analysis completed. Result: {str(result)[:500]}..."


def handle_pandas_error(error, query: str) -> str:
    """Handle pandas errors with helpful suggestions"""
    error_str = str(error).lower()
    
    if 'keyerror' in error_str or 'column' in error_str:
        return ("I encountered a column-related error. Please check that the column names in your query "
                "match the actual column names in the dataset. You can ask 'what are the column names?' to verify.")
    
    elif 'timeout' in error_str:
        return ("The analysis took too long to complete. Please try a simpler query or filter the data first to reduce the dataset size.")
    
    elif 'memory' in error_str:
        return ("The analysis requires too much memory. Please try working with a subset of the data or use more efficient operations.")
    
    elif 'syntax' in error_str or 'invalid' in error_str:
        return ("There was a syntax error in the generated code. Please rephrase your query more clearly or try a different approach.")
    
    else:
        return f"I encountered an error while analyzing the data: {str(error)[:200]}. Please try rephrasing your query or ask for help with a specific operation."


@timing_decorator
def enhanced_rag_pipeline(query: str, source: DataSource, chat_history: List[dict]) -> str:
    """Enhanced RAG pipeline with better context management and response generation"""
    
    # Retrieve relevant chunks
    retrieved_chunks = []
    
    try:
        if source.index.ntotal == 0:
            return f"No indexed content available in '{source.file_name}'."
        
        query_embedding = get_embeddings_batch([query], task="RETRIEVAL_QUERY")
        if query_embedding is None:
            return "I couldn't process your query. Please try again."
        
        # Get more chunks for better context
        k = min(config.TOP_K * 2, source.index.ntotal)
        distances, indices = source.index.search(query_embedding, k)
        
        # Filter chunks by relevance threshold and diversity
        used_chunks = set()
        for dist, idx in zip(distances[0], indices[0]):
            if dist < 1.0:  # Relevance threshold
                chunk = source.chunks[idx]
                # Simple diversity check - avoid very similar chunks
                chunk_words = set(chunk.lower().split())
                is_diverse = all(
                    len(chunk_words.intersection(set(used_chunk.lower().split()))) < len(chunk_words) * 0.8
                    for used_chunk in used_chunks
                )
                
                if is_diverse or len(retrieved_chunks) < config.TOP_K // 2:
                    retrieved_chunks.append((chunk, float(dist)))
                    used_chunks.add(chunk)
                
                if len(retrieved_chunks) >= config.TOP_K:
                    break
        
        if not retrieved_chunks:
            return f"I couldn't find relevant information in '{source.file_name}' for your query."
        
        # Create enhanced context
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved_chunks[:config.TOP_K], 1):
            context_parts.append(f"[{i}] {chunk}")
        
        context = '\n'.join(context_parts)
        
        # Consider chat history for context
        history_context = ""
        if chat_history:
            recent_history = chat_history[-4:]  # Last 2 exchanges
            history_parts = []
            for msg in recent_history:
                role = msg.get('role', '')
                content = msg.get('content', '')[:200]  # Limit length
                if role and content:
                    history_parts.append(f"{role.capitalize()}: {content}")
            if history_parts:
                history_context = f"\n\nRecent conversation context:\n" + "\n".join(history_parts)
        
        # Enhanced system prompt
        system_prompt = f"""You are a helpful AI assistant analyzing content from '{source.file_name}'. 

Use the following information to provide a comprehensive and accurate answer:

RELEVANT CONTENT:
{context}
{history_context}

INSTRUCTIONS:
1. Answer the user's question using ONLY the provided content
2. If the content doesn't fully address the question, clearly state what information is missing
3. Provide specific details and examples from the content when possible
4. If you reference specific information, you can mention it's from section [1], [2], etc.
5. Be conversational and helpful in your response
6. If the question requires analysis or calculation that isn't directly in the content, explain what you can determine and what would need additional information

USER QUESTION: {query}

ANSWER:"""
        
        response = generate_with_gemini(system_prompt, temperature=0.2, max_output_tokens=1024)
        
        logger.info(f"RAG pipeline completed for query in '{source.file_name}'")
        return response
        
    except Exception as e:
        logger.error(f"Error in enhanced RAG pipeline: {e}", exc_info=True)
        return f"I encountered an error while searching '{source.file_name}'. Please try again."

# ========== Session Management ==========
def cleanup_expired_sessions():
    """Remove expired sessions to free memory"""
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, rag_session in sessions.items():
        if current_time - rag_session.last_activity > timedelta(hours=2):
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        try:
            rag_session = sessions.pop(session_id, None)
            if rag_session:
                # Clean up associated datasets
                for tag_sources in rag_session.tagged_sources.values():
                    for source in tag_sources:
                        if source.dataset_id:
                            datasets.pop(source.dataset_id, None)
                logger.info(f"Cleaned up expired session: {session_id}")
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")

# ========== Enhanced Flask Routes ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test Gemini API
        test_response = generate_with_gemini("Test", temperature=0)
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'active_sessions': len(sessions),
            'cached_datasets': len(datasets),
            'gemini_api': 'operational' if test_response else 'error'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/session/create', methods=['POST'])
def create_session():
    """Enhanced session creation with cleanup"""
    try:
        # Clean up expired sessions first
        cleanup_expired_sessions()
        
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        
        rag_session = RAGSession(session_id=session_id)
        sessions[session_id] = rag_session
        
        logger.info(f"Created new session: {session_id}")
        
        return jsonify({
            'session_id': session_id,
            'message': 'Session created successfully.',
            'timestamp': rag_session.created_at.isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return jsonify({'error': 'Failed to create session'}), 500

@app.route('/api/session/info', methods=['GET'])
def session_info():
    """Get current session information"""
    session_id = session.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'No active session'}), 400
    
    rag_session = sessions[session_id]
    rag_session.update_activity()
    
    # Get detailed source information
    sources_info = {}
    for tag, sources in rag_session.tagged_sources.items():
        sources_info[tag] = [source.to_dict() for source in sources]
    
    return jsonify({
        'session': rag_session.get_summary(),
        'sources': sources_info
    })

@app.route('/api/ingest/url', methods=['POST'])
def ingest_url():
    """Ingests content from a publicly accessible URL."""
    data = request.get_json()
    if not data or 'url' not in data or 'tag' not in data:
        return jsonify({'error': 'Request must include a JSON body with "url" and "tag" keys'}), 400
        
    url = data.get('url')
    tag = data.get('tag', '').strip()

    if not tag:
        return jsonify({'error': 'Tag cannot be empty'}), 400

    session_id = session.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'No active session'}), 400
    
    rag_session = sessions[session_id]
    rag_session.update_activity()
    
    try:
        logger.info(f"Attempting to scrape URL: {url}")
        scraped_content = scrape_and_process_url(url)
        
        if not scraped_content:
            return jsonify({'error': f"Failed to scrape or find sufficient content at URL: {url}"}), 400

        chunks = enhanced_chunk_text(scraped_content)
        if not chunks:
            return jsonify({'error': 'Could not extract sufficient text chunks from the website.'}), 400

        embeddings = get_embeddings_batch(chunks)
        if embeddings is None:
            return jsonify({'error': 'Failed to create embeddings for the web content.'}), 500

        index = faiss.IndexFlatL2(config.MODEL_DIMENSION)
        index.add(embeddings)
        
        # Use the URL as the "filename" for the DataSource
        source = DataSource(
            file_name=url, 
            data_type='text', 
            index=index, 
            chunks=chunks,
            file_size=len(scraped_content),
            metadata={'source_type': 'url'}
        )
        
        if tag not in rag_session.tagged_sources:
            rag_session.tagged_sources[tag] = []
        rag_session.tagged_sources[tag].append(source)
        
        logger.info(f"Successfully ingested content from URL '{url}' under tag '{tag}'.")
        return jsonify({
            'message': "Successfully ingested content from URL.", 
            'source': source.to_dict()
        })

    except Exception as e:
        logger.error(f"Error during URL ingestion: {e}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred: {e}'}), 500

@app.route('/api/ingest', methods=['POST'])
def enhanced_ingest_data():
    """Enhanced data ingestion with better error handling and metrics"""
    
    # Validate request
    if 'file' not in request.files or 'tag' not in request.form:
        return jsonify({'error': 'Request must include a file and a tag'}), 400
    
    file = request.files['file']
    tag = request.form['tag'].strip()
    
    if not tag or not file.filename:
        return jsonify({'error': 'File and tag cannot be empty'}), 400
    
    # Validate session
    session_id = session.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'No active session'}), 400
    
    rag_session = sessions[session_id]
    rag_session.update_activity()
    
    # Validate file type
    is_valid, file_type = validate_file_type(file.filename)
    if not is_valid:
        return jsonify({'error': file_type}), 400
    
    try:
        metrics = ProcessingMetrics(start_time=time.time())
        file_name = secure_filename(file.filename)
        file_extension = os.path.splitext(file_name)[1].lower()
        
        # Initialize tag if needed
        if tag not in rag_session.tagged_sources:
            rag_session.tagged_sources[tag] = []
        
        # Get file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        logger.info(f"Processing file '{file_name}' ({file_size} bytes) with tag '{tag}'")
        
        with error_handler(f"ingesting {file_name}"):
            if file_extension == '.pdf':
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{tag}_{file_name}")
                file.save(filepath)
                
                # Enhanced PDF processing
                extracted_dfs, text_content, pdf_metadata = enhanced_pdf_processing(filepath)
                
                # Process extracted tables
                for i, df in enumerate(extracted_dfs):
                    if df.empty:
                        continue
                    
                    # Save table as Excel file
                    base_name = os.path.splitext(file_name)[0]
                    table_name = f"{base_name}_table_{i+1}.xlsx"
                    excel_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(table_name))
                    df.to_excel(excel_path, index=False)
                    
                    # Create data source
                    dataset_id = str(uuid.uuid4())
                    datasets[dataset_id] = df
                    
                    chunks = enhanced_dataframe_chunking(df)
                    metrics.chunks_created += len(chunks)
                    
                    if chunks:
                        # Create embeddings and index
                        embeddings_start = time.time()
                        embeddings = get_embeddings_batch(chunks)
                        metrics.embedding_time += time.time() - embeddings_start
                        
                        if embeddings is not None:
                            index_start = time.time()
                            index = faiss.IndexFlatL2(config.MODEL_DIMENSION)
                            index.add(embeddings)
                            metrics.indexing_time += time.time() - index_start
                            
                            source = DataSource(
                                file_name=table_name,
                                data_type='excel',
                                index=index,
                                chunks=chunks,
                                dataset_id=dataset_id,
                                file_size=len(df) * len(df.columns),  # Approximate size
                                metadata={'extracted_from_pdf': file_name, 'table_index': i}
                            )
                            
                            rag_session.tagged_sources[tag].append(source)
                            logger.info(f"Successfully processed table {i+1} from PDF")
                
                # Process remaining text content
                if text_content:
                    text_chunks = enhanced_chunk_text(text_content)
                    metrics.chunks_created += len(text_chunks)
                    
                    if text_chunks:
                        embeddings_start = time.time()
                        text_embeddings = get_embeddings_batch(text_chunks)
                        metrics.embedding_time += time.time() - embeddings_start
                        
                        if text_embeddings is not None:
                            index_start = time.time()
                            text_index = faiss.IndexFlatL2(config.MODEL_DIMENSION)
                            text_index.add(text_embeddings)
                            metrics.indexing_time += time.time() - index_start
                            
                            text_source = DataSource(
                                file_name=file_name,
                                data_type='text',
                                index=text_index,
                                chunks=text_chunks,
                                file_size=len(text_content),
                                metadata=pdf_metadata
                            )
                            
                            rag_session.tagged_sources[tag].append(text_source)
                            logger.info(f"Successfully processed text content from PDF")
            
            elif file_extension in ['.xlsx', '.xls', '.csv']:
                # Enhanced data file processing
                try:
                    if file_extension == '.csv':
                        df = pd.read_csv(file, encoding='utf-8')
                    else:
                        # Handle multiple sheets
                        excel_data = pd.read_excel(file, sheet_name=None)
                        if len(excel_data) == 1:
                            df = list(excel_data.values())[0]
                        else:
                            # Combine all sheets
                            df = pd.concat(excel_data.values(), ignore_index=True)
                    
                    # Enhanced data cleaning
                    original_shape = df.shape
                    df = df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
                    
                    if df.empty:
                        return jsonify({'error': 'File contains no valid data after cleaning'}), 400
                    
                    logger.info(f"Data cleaned: {original_shape} -> {df.shape}")
                    
                    # Store dataset and create source
                    dataset_id = str(uuid.uuid4())
                    datasets[dataset_id] = df
                    
                    chunks = enhanced_dataframe_chunking(df)
                    metrics.chunks_created = len(chunks)
                    
                    if chunks:
                        embeddings_start = time.time()
                        embeddings = get_embeddings_batch(chunks)
                        metrics.embedding_time = time.time() - embeddings_start
                        
                        if embeddings is not None:
                            index_start = time.time()
                            index = faiss.IndexFlatL2(config.MODEL_DIMENSION)
                            index.add(embeddings)
                            metrics.indexing_time = time.time() - index_start
                            
                            source = DataSource(
                                file_name=file_name,
                                data_type='excel',
                                index=index,
                                chunks=chunks,
                                dataset_id=dataset_id,
                                file_size=file_size,
                                metadata={'original_shape': original_shape, 'cleaned_shape': df.shape}
                            )
                            
                            rag_session.tagged_sources[tag].append(source)
                
                except Exception as e:
                    logger.error(f"Error processing data file: {e}")
                    return jsonify({'error': f'Error processing data file: {str(e)[:100]}'}), 400
            
            else:  # Text files
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{tag}_{file_name}")
                file.save(filepath)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    cleaned_content = clean_and_validate_text(content, min_length=50)
                    if not cleaned_content:
                        return jsonify({'error': 'File contains insufficient readable text'}), 400
                    
                    chunks = enhanced_chunk_text(cleaned_content)
                    metrics.chunks_created = len(chunks)
                    
                    if chunks:
                        embeddings_start = time.time()
                        embeddings = get_embeddings_batch(chunks)
                        metrics.embedding_time = time.time() - embeddings_start
                        
                        if embeddings is not None:
                            index_start = time.time()
                            index = faiss.IndexFlatL2(config.MODEL_DIMENSION)
                            index.add(embeddings)
                            metrics.indexing_time = time.time() - index_start
                            
                            source = DataSource(
                                file_name=file_name,
                                data_type='text',
                                index=index,
                                chunks=chunks,
                                file_size=file_size,
                                metadata={'content_length': len(cleaned_content)}
                            )
                            
                            rag_session.tagged_sources[tag].append(source)
                
                except UnicodeDecodeError:
                    return jsonify({'error': 'Unable to read file. Please ensure it\'s a valid text file with UTF-8 encoding.'}), 400
        
        # Finalize metrics
        metrics.end_time = time.time()
        
        # Update source metrics
        for source in rag_session.tagged_sources[tag]:
            if not hasattr(source, 'metrics') or source.metrics is None:
                source.metrics = metrics
                break
        
        success_message = f"Successfully processed '{file_name}' under tag '{tag}'"
        logger.info(f"{success_message} - {metrics.chunks_created} chunks in {metrics.total_time:.2f}s")
        
        return jsonify({
            'message': success_message,
            'processing_stats': {
                'chunks_created': metrics.chunks_created,
                'processing_time': round(metrics.total_time, 2),
                'embedding_time': round(metrics.embedding_time, 2),
                'indexing_time': round(metrics.indexing_time, 2)
            }
        })
    
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        return jsonify({'error': f'Processing failed: {str(e)[:200]}'}), 500

@app.route('/api/chat', methods=['POST'])
def enhanced_chat():
    """Enhanced chat endpoint with better routing and error handling"""
    
    # Validate request
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request must contain JSON data'}), 400
    
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    # Validate session
    session_id = session.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'No active session. Please create a session first.'}), 400
    
    rag_session = sessions[session_id]
    rag_session.update_activity()
    
    # Check if any sources are available
    available_tags = list(rag_session.tagged_sources.keys())
    if not available_tags:
        return jsonify({
            'response': "No documents have been uploaded yet. Please upload some documents first using the /api/ingest endpoint."
        })
    
    try:
        start_time = time.time()
        
        # Enhanced tag extraction
        selected_tag = intelligent_tag_extraction(query, available_tags)
        
        if not selected_tag:
            return jsonify({
                'response': f"I couldn't determine which document to use. Please specify one of the available tags: {', '.join(available_tags)}. "
                           f"For example, you can say 'using tag \"{available_tags[0]}\"' or include the tag name in your question."
            })
        
        # Clean query by removing routing instructions
        cleaned_query = query
        for tag in available_tags:
            patterns = [
                rf'\b(using|from|in|with|on|for)\s+(tag\s+)?[\'"]?{re.escape(tag)}[\'"]?\b',
                rf'\b{re.escape(tag)}\s+(tag|document|file|data)\b'
            ]
            for pattern in patterns:
                cleaned_query = re.sub(pattern, '', cleaned_query, count=1, flags=re.IGNORECASE)
        
        cleaned_query = re.sub(r'^[,\s\-:]+', '', cleaned_query.strip())
        
        if not cleaned_query:
            cleaned_query = query  # Fallback to original query
        
        logger.info(f"Query processing: tag='{selected_tag}', original='{query}', cleaned='{cleaned_query}'")
        
        # Get sources for the selected tag
        sources_for_tag = rag_session.tagged_sources[selected_tag]
        if not sources_for_tag:
            return jsonify({
                'error': f'No sources found for tag "{selected_tag}"'
            }), 500
        
        # Enhanced source selection
        target_source = enhanced_vector_search(cleaned_query, sources_for_tag)
        if not target_source:
            return jsonify({
                'error': f'Could not select a relevant source for tag "{selected_tag}"'
            }), 500
        
        # Route to appropriate agent
        response_text = ""
        processing_info = {
            'selected_tag': selected_tag,
            'selected_source': target_source.file_name,
            'agent_used': '',
            'processing_time': 0
        }
        
        if target_source.data_type == 'excel' and target_source.dataset_id:
            # Route between pandas agent and RAG
            agent_choice = intelligent_agent_routing(cleaned_query, rag_session.chat_history)
            processing_info['agent_used'] = agent_choice
            
            if agent_choice == 'pandas':
                df = datasets.get(target_source.dataset_id)
                if df is not None:
                    response_text = enhanced_pandas_agent(cleaned_query, df, rag_session.chat_history)
                else:
                    logger.warning(f"Dataset {target_source.dataset_id} not found, falling back to RAG")
                    response_text = enhanced_rag_pipeline(cleaned_query, target_source, rag_session.chat_history)
                    processing_info['agent_used'] = 'rag_fallback'
            else:
                response_text = enhanced_rag_pipeline(cleaned_query, target_source, rag_session.chat_history)
        else:
            # Use RAG for text sources
            processing_info['agent_used'] = 'rag'
            response_text = enhanced_rag_pipeline(cleaned_query, target_source, rag_session.chat_history)
        
        processing_info['processing_time'] = round(time.time() - start_time, 2)
        
        # Update chat history (limit to recent turns)
        rag_session.chat_history.append({"role": "user", "content": query})
        rag_session.chat_history.append({"role": "assistant", "content": response_text})
        
        # Keep only recent history
        if len(rag_session.chat_history) > config.CHAT_HISTORY_TURNS * 2:
            rag_session.chat_history = rag_session.chat_history[-(config.CHAT_HISTORY_TURNS * 2):]
        
        logger.info(f"Chat completed in {processing_info['processing_time']}s using {processing_info['agent_used']} agent")
        
        return jsonify({
            'response': response_text,
            'processing_info': processing_info
        })
    
    except Exception as e:
        logger.error(f"Error in enhanced chat endpoint: {e}", exc_info=True)
        return jsonify({
            'error': 'An error occurred while processing your request. Please try again.',
            'details': str(e)[:200] if app.debug else None
        }), 500

@app.route('/api/session/clear', methods=['POST'])
def clear_session():
    """Enhanced session clearing"""
    session_id = session.get('session_id')
    
    if session_id and session_id in sessions:
        try:
            rag_session = sessions[session_id]
            
            # Clean up datasets
            datasets_removed = 0
            for tag_sources in rag_session.tagged_sources.values():
                for source in tag_sources:
                    if source.dataset_id and source.dataset_id in datasets:
                        datasets.pop(source.dataset_id, None)
                        datasets_removed += 1
            
            # Remove session
            sessions.pop(session_id, None)
            session.clear()
            
            logger.info(f"Cleared session {session_id}: removed {datasets_removed} datasets")
            
            return jsonify({
                'message': 'Session cleared successfully.',
                'datasets_removed': datasets_removed
            })
        
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return jsonify({'error': 'Error clearing session'}), 500
    else:
        return jsonify({'message': 'No active session to clear.'})

@app.route('/api/stats', methods=['GET'])
def system_stats():
    """System statistics endpoint"""
    try:
        total_sources = sum(
            len(sources) for session_data in sessions.values() 
            for sources in session_data.tagged_sources.values()
        )
        
        return jsonify({
            'active_sessions': len(sessions),
            'total_datasets': len(datasets),
            'total_sources': total_sources,
            'memory_usage': {
                'sessions': len(sessions),
                'datasets': len(datasets)
            },
            'uptime': time.time(),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return jsonify({'error': 'Unable to get system stats'}), 500

# ========== Error Handlers ==========
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 256MB.'}), 413

# ========== Application Startup ==========
if __name__ == '__main__':
    logger.info("Starting Enhanced RAG System...")
    logger.info(f"Configuration: TOP_K={config.TOP_K}, CHUNK_WORDS={config.CHUNK_WORDS}, "
                f"MAX_WORKERS={config.MAX_WORKERS}")
    
    try:
        app.run(
        debug=True,  # Set to True for development and auto-reloading
        host='0.0.0.0',
        port=5000
    )
    except KeyboardInterrupt:
        logger.info("Shutting down Enhanced RAG System...")
        executor.shutdown(wait=True)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise