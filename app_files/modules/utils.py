from __future__ import annotations

"""General utility functions for the application.

This module provides utility functions and classes for:
- File operations (reading, encoding detection)
- Text extraction (PDF, DOCX, HTML)
- HTTP session management and file downloads
- Safe filename generation
- Path handling and validation
"""

# Standard library imports
import asyncio
import io
import logging
import os
import re
from asyncio import Lock, Semaphore
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache, wraps, partial
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    AsyncContextManager,
    AsyncGenerator,
    Callable,
    Dict,
    Final,
    Iterable,
    List,
    Literal,
    Optional,
    ParamSpec,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeAlias,
    TypedDict,
    TypeGuard,
    TypeVar,
    Union,
    cast
)

# Third-party imports
import aiohttp
import numpy as np
import pandas as pd
from aiohttp import ClientSession
from numpy.typing import NDArray
from pandas import DataFrame, Series

# Local imports
from .core_types import (
    DataFrameType,
    PathLike,
    JsonDict,
    BytesContent,
    TextContent,
    ClientSessionAlias,
    ArrayLike,
    SeriesType,
    StatusEntry
)

# Initialize logger
logger = logging.getLogger(__name__)

# Type variables for generic operations
T = TypeVar('T')
DType = TypeVar('DType')
E = TypeVar('E', bound=Exception)

# Type aliases for complex types
ClientSessionAlias: TypeAlias = ClientSession
DataFrameType: TypeAlias = DataFrame
SeriesType = pd.Series
PathLike: TypeAlias = Union[str, Path]
JsonDict: TypeAlias = Dict[str, Any]
ArrayLike: TypeAlias = Union[NDArray[Any], pd.Series, Sequence[Any]]
BytesContent: TypeAlias = bytes
TextContent: TypeAlias = str

# Type parameters for decorator typing
P = ParamSpec('P')
R_co = TypeVar('R_co', covariant=True)

# Constants
MAX_CONCURRENT_DOWNLOADS: Final[int] = 5
MAX_RETRIES: Final[int] = 3
DOWNLOAD_TIMEOUT: Final[int] = 30
MAX_ATTACHMENT_SIZE: Final[int] = 10 * 1024 * 1024  # 10MB
CACHE_SIZE: Final[int] = 1000
MAX_DATAFRAME_ROWS: Final[int] = 1_000_000
CATEGORY_THRESHOLD: Final[float] = 0.5

ALLOWED_TYPES: Final[Dict[str, str]] = {
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.txt': 'text/plain',
    '.csv': 'text/csv'
}

# Optional imports with fallbacks
try:
    import chardet  # type: ignore
except ImportError:
    chardet = None  # type: ignore
    logger.warning("chardet not installed. Encoding detection will fall back to utf-8")

try:
    import docx  # type: ignore
except ImportError:
    docx = None  # type: ignore
    logger.warning("python-docx not installed. DOCX processing will be unavailable")

try:
    from bs4 import BeautifulSoup  # type: ignore
    BeautifulSoup = BeautifulSoup  # type: ignore
except ImportError:
    BeautifulSoup = None  # type: ignore
    logger.warning("beautifulsoup4 not installed. HTML processing will be limited")

try:
    from pypdf import PdfReader  # type: ignore
    PdfReader = PdfReader  # type: ignore
except ImportError:
    PdfReader = None  # type: ignore
    logger.warning("pypdf not installed. PDF processing will be unavailable")

class AsyncFunction(Protocol[P, R_co]):
    """Protocol for async functions.
    
    This protocol defines the signature for async functions that can be used
    with decorators and other higher-order functions in the module.
    
    Type Parameters:
        P: Parameter specification for the function
        R_co: Covariant return type
        
    Note:
        Used for type-safe decorator implementations
    """
    async def __call__(self, *args, **kwargs) -> Any: ...

def handle_error(error: E, logger: logging.Logger) -> str:
    """Handle exceptions with consistent logging.
    
    Args:
        error: Exception to handle
        logger: Logger instance
        
    Returns:
        Formatted error message
    """
    error_msg = f"{error.__class__.__name__}: {str(error)}"
    logger.error(error_msg)
    return error_msg

class HttpSessionManager:
    """Manages HTTP client sessions for the application.
    
    This class provides a centralized way to manage aiohttp client sessions,
    including connection pooling, concurrent download limits, and proper
    resource cleanup.
    
    Attributes:
        _session: The current aiohttp client session
        _lock: Lock for thread-safe session management
        _download_semaphore: Semaphore for limiting concurrent downloads
        
    Note:
        Thread-safe and handles session lifecycle
    """
    
    def __init__(self) -> None:
        """Initialize the session manager with default settings."""
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock: Lock = asyncio.Lock()
        self._download_semaphore: Semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create a shared aiohttp session.
        
        Returns:
            An active aiohttp ClientSession instance
            
        Raises:
            RuntimeError: If aiohttp is not installed
            
        Note:
            Creates new session if current one is closed
        """
        async with self._lock:
            if self._session is None or self._session.closed:
                if aiohttp is None:
                    raise RuntimeError("aiohttp is required but not installed")
                self._session = aiohttp.ClientSession()
            return self._session
    
    async def close(self) -> None:
        """Close the current session if it exists.
        
        This method ensures proper cleanup of resources by closing
        any open client session.
        
        Note:
            Thread-safe and handles errors gracefully
        """
        async with self._lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None
    
    @property
    def download_semaphore(self) -> Semaphore:
        """Get the download semaphore for limiting concurrent downloads.
        
        Returns:
            Semaphore instance controlling concurrent downloads
            
        Note:
            Used to prevent too many concurrent downloads
        """
        return self._download_semaphore

# Global session manager instance
http_session_manager = HttpSessionManager()

def is_url(text: str) -> bool:
    """Check if text is a URL."""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(text))

@asynccontextmanager
async def managed_session() -> AsyncGenerator[ClientSessionAlias, None]:
    """Context manager for aiohttp ClientSession with retry logic."""
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        yield session

def with_retries(
    max_retries: int = MAX_RETRIES
) -> Callable[[AsyncFunction[P, R_co]], AsyncFunction[P, Optional[R_co]]]:
    """Decorator for retrying async functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        
    Returns:
        Decorated async function with retry logic
        
    Note:
        Uses exponential backoff between retries
    """
    def decorator(func: AsyncFunction[P, R_co]) -> AsyncFunction[P, Optional[R_co]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Optional[R_co]:
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    delay = 2 ** attempt
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. "
                        f"Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
            logger.error(f"All {max_retries} attempts failed. Last error: {str(last_error)}")
            return None
        return wrapper
    return decorator

class FileReader:
    """Utility class for reading files with encoding detection."""
    
    @staticmethod
    async def read_with_encoding(file_path: str) -> Tuple[str, str]:
        """Read file content with encoding detection.
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (file content, detected encoding)
            
        Raises:
            IOError: If file cannot be read
            
        Note:
            Uses chardet for encoding detection if available
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                
            if chardet:
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'
            else:
                encoding = 'utf-8'
                
            content = raw_data.decode(encoding)
            return content, encoding
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise IOError(f"Could not read file: {str(e)}")

class TextExtractor:
    """Utility class for extracting text from various file formats."""
    
    @staticmethod
    def from_pdf(content: bytes) -> str:
        """Extract text from PDF content.
        
        Args:
            content: PDF file content as bytes
            
        Returns:
            Extracted text
            
        Note:
            Requires pypdf package
        """
        if not PdfReader:
            logger.error("pypdf not installed")
            return ""
            
        try:
            pdf = PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            return ""
    
    @staticmethod
    def from_docx(content: bytes) -> str:
        """Extract text from DOCX content.
        
        Args:
            content: DOCX file content as bytes
            
        Returns:
            Extracted text
            
        Note:
            Requires python-docx package
        """
        if not docx:
            logger.error("python-docx not installed")
            return ""
            
        try:
            doc = docx.Document(io.BytesIO(content))
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {str(e)}")
            return ""
    
    @staticmethod
    def from_html(html_content: str) -> str:
        """Extract text from HTML content using BeautifulSoup."""
        if not BeautifulSoup:
            logger.error("BeautifulSoup not installed")
            return ""

    @staticmethod
    def from_text(text_content: str) -> str:
        """Process plain text content."""
        return text_content.strip()

@with_retries()
async def download_attachment(
    session: ClientSessionAlias, # type: ignore
    url: str,
    timeout: int = DOWNLOAD_TIMEOUT
) -> Optional[BytesContent]: # type: ignore
    """Download attachment with size limit and timeout.
    
    Args:
        session: aiohttp client session
        url: URL to download from
        timeout: Download timeout in seconds
        
    Returns:
        Downloaded content as bytes or None if failed
        
    Note:
        - Enforces size limit
        - Uses timeout
        - Retries on failure
    """
    try:
        async with session.get(url, timeout=timeout) as response:
            if response.status != 200:
                logger.error(f"Download failed with status {response.status}: {url}")
                return None
                
            content_type = response.headers.get('content-type', '')
            if not any(mime in content_type.lower() for mime in ALLOWED_TYPES.values()):
                logger.warning(f"Unsupported content type: {content_type}")
                return None
                
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > MAX_ATTACHMENT_SIZE:
                logger.warning(f"File too large: {content_length} bytes")
                return None
                
            return await response.read()
                
    except asyncio.TimeoutError:
        logger.error(f"Download timed out after {timeout}s: {url}")
        return None
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return None

def get_safe_filename(filename: str) -> str:
    """Generate safe filename from potentially unsafe input.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
        
    Note:
        - Removes unsafe characters
        - Limits length
        - Adds timestamp if needed
    """
    # Remove unsafe characters
    safe_chars = re.sub(r'[^\w\-_\. ]', '', filename)
    
    # Replace spaces with underscores
    safe_chars = safe_chars.replace(' ', '_')
    
    # Limit length (leaving room for timestamp)
    max_length = 200
    if len(safe_chars) > max_length:
        name, ext = os.path.splitext(safe_chars)
        safe_chars = name[:max_length-len(ext)] + ext
    
    # Add timestamp if empty
    if not safe_chars:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_chars = f"file_{timestamp}"
    
    return safe_chars

def parse_list_input(input_str: str) -> List[str]:
    """Parse comma or newline separated string into list of topics.
    
    Args:
        input_str: Input string containing topics
        
    Returns:
        List of cleaned topic strings
        
    Note:
        - Handles both comma and newline separation
        - Cleans and deduplicates topics
        - Removes empty strings
    """
    if not input_str or not isinstance(input_str, str):
        return []
        
    # Split on commas and newlines
    items = re.split(r'[,\n]', input_str)
    
    # Clean and deduplicate topics
    cleaned = set()
    for item in items:
        item = item.strip()
        if item:  # Only add non-empty items
            cleaned.add(item)
            
    return list(cleaned)

def get_column_names(df: DataFrameType) -> List[str]:
    """Get a list of column names from a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of column names as strings
        
    Raises:
        ValueError: If DataFrame is None or empty
        
    Note:
        Validates DataFrame before returning names
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty")
    return df.columns.tolist()

def validate_dataframe(
    df: DataFrameType,
    required_columns: Optional[List[str]] = None,
    max_rows: int = MAX_DATAFRAME_ROWS
) -> None:
    """Validate DataFrame against common requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: Optional list of required column names
        max_rows: Maximum allowed number of rows
        
    Raises:
        ValueError: If validation fails for any reason
        
    Note:
        - Checks for None/empty DataFrame
        - Validates row count
        - Verifies required columns
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty")
        
    if len(df) > max_rows:
        raise ValueError(f"DataFrame too large: {len(df)} rows (max {max_rows})")
        
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

def optimize_dataframe(
    df: DataFrameType,
    category_threshold: float = CATEGORY_THRESHOLD
) -> DataFrameType:
    """Optimize DataFrame memory usage.
    
    Args:
        df: DataFrame to optimize
        category_threshold: Threshold for converting to categorical
        
    Returns:
        Optimized DataFrame with appropriate data types
        
    Note:
        - Converts object columns to categorical when appropriate
        - Optimizes numeric columns
        - Handles NA values safely
        - Returns original DataFrame if optimization fails
    """
    try:
        df = df.copy()
        
        # Handle numeric columns
        for col in df.select_dtypes(include=['int', 'float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
        # Handle object columns
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < category_threshold:
                df[col] = df[col].astype('category')
                
        return df
        
    except Exception as e:
        logger.error(f"Error optimizing DataFrame: {str(e)}")
        return df

def safe_to_numeric(
    series: pd.Series,
    errors: Literal['raise', 'coerce'] = 'coerce'
) -> pd.Series:
    """Safely convert a series to numeric type."""
    try:
        return pd.to_numeric(series, errors=errors)
    except Exception as e:
        logger.error(f"Error converting to numeric: {str(e)}")
        return pd.Series([], dtype=float)  # Return empty series with correct type

def safe_to_datetime(
    series: pd.Series,
    format: Optional[str] = None,
    errors: Literal['raise', 'coerce'] = 'coerce'
) -> pd.Series:
    """Safely convert a series to datetime type."""
    try:
        return pd.to_datetime(series, format=format, errors=errors)
    except Exception as e:
        logger.error(f"Error converting to datetime: {str(e)}")
        return pd.Series([], dtype='datetime64[ns]')  # Return empty series with correct type

def get_memory_usage(
    df: pd.DataFrame
) -> Dict[str, Union[int, float, Dict[str, int], Dict[str, int]]]:
    """Get memory usage statistics for a DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary containing memory usage statistics
        
    Note:
        Returns details about:
        - Total memory usage
        - Memory usage by column
        - Memory usage by data type
        - Returns empty dict if analysis fails
    """
    try:
        total = df.memory_usage(deep=True).sum()
        by_column = df.memory_usage(deep=True)
        by_dtype = df.memory_usage(deep=True).groupby(df.dtypes).sum()
        
        return {
            'total_bytes': total,
            'total_mb': total / 1024 / 1024,
            'by_column': dict(by_column),
            'by_dtype': dict(by_dtype)
        }
    except Exception as e:
        logger.error(f"Error calculating memory usage: {str(e)}")
        return {
            'total_bytes': 0,
            'total_mb': 0,
            'by_column': {},
            'by_dtype': {}
        }

import base64

def create_data_url(file_path: PathLike) -> str:
    """Create a data URL from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Data URL string containing the file content
        
    Note:
        Handles both HTML and image files appropriately
        Returns empty string on error
    """
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            
        ext = Path(file_path).suffix.lower()
        mime_type = {
            '.html': 'text/html',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg'
        }.get(ext, 'application/octet-stream')
        
        b64_content = base64.b64encode(content).decode('utf-8')
        return f"data:{mime_type};base64,{b64_content}"
        
    except Exception as e:
        logger.error(f"Error creating data URL for {file_path}: {str(e)}")
        return ""

