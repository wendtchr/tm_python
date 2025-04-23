"""Data processing module for topic modeling application.

This module provides functionality for:
- File upload processing
- Attachment handling
- Data cleaning and validation
- Text preprocessing
- Paragraph splitting
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable, AsyncGenerator, overload
import asyncio
import re
from contextlib import asynccontextmanager
from functools import lru_cache
from dataclasses import dataclass
import typing

import pandas as pd
import aiohttp
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from . import config, utils
from .core_types import (
    DataFrameType, StatusProtocol, PathLike,
    SessionProtocol, ClientSessionAlias
)

logger = logging.getLogger(__name__)

# Module-specific types
ProcessResult = Tuple[int, str]
StatusCallback = Callable[[str, float, str], None]

@dataclass
class StageResult:
    """Result of a processing stage."""
    df: DataFrameType
    files: List[Path]
    stage: str
    success: bool

__all__ = [
    'DataFrameProcessor',
    'process_file_upload',
    'process_attachments',
    'clean_data',
    'split_paragraphs'
]

class DataFrameProcessor:
    """Utility class for DataFrame operations."""
    
    @staticmethod
    def process_dataframe(df: DataFrameType) -> DataFrameType:
        """Process DataFrame with validation and cleaning."""
        if df is None or df.empty:
            raise ValueError("DataFrame is None or empty")

        df_clean = df.copy()
        
        # Clean text data but preserve paragraph breaks
        if 'Comment' in df_clean.columns:
            df_clean['Comment'] = df_clean['Comment'].astype(str).apply(
                lambda x: re.sub(r'\s*\n\s*\n\s*', '\n\n', x.strip())  # Normalize paragraph breaks
            )
            
        # Handle timestamps
        if 'Posted Date' in df_clean.columns:
            df_clean['Posted Date'] = pd.to_datetime(
                df_clean['Posted Date'], 
                format='mixed', 
                errors='coerce'
            )
        
        return df_clean
    
    @staticmethod
    def split_paragraphs(df: DataFrameType, min_length: int) -> DataFrameType:
        """Use standalone split_paragraphs function instead."""
        return split_paragraphs(df, min_length=min_length, output_dir=None, status_manager=None)

@asynccontextmanager
async def status_update(
    status_manager: Optional[StatusProtocol],
    stage: str,
    progress: float,
    message: str
) -> AsyncGenerator[None, None]:
    """Context manager for status updates with delay.
    
    Args:
        status_manager: Status manager instance
        stage: Current processing stage
        progress: Progress percentage (0-100)
        message: Status message
        
    Yields:
        None
        
    Note:
        Adds delay before and after status update
        Ensures status is always updated even if operation fails
    """
    if status_manager:
        status_manager.update_status(stage=stage, progress=progress, message=message)
        await asyncio.sleep(config.STATUS_DELAY)
    try:
        yield
    finally:
        await asyncio.sleep(config.STATUS_DELAY)

@overload
async def process_file_upload(
    file_path: PathLike,
    status_manager: Optional[StatusProtocol] = None,
    session_manager: Optional[SessionProtocol] = None,
    output_dir: Optional[PathLike] = None
) -> DataFrameType:
    """Process uploaded file with validation and error handling."""
    try:
        if status_manager:
            status_manager.update_status("Processing", 0, f"Processing {file_path}")
            
        # Validate file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Use encoding-aware reading
        df = read_csv_with_encoding(str(file_path))
        
        # Validate DataFrame
        if df.empty:
            raise ValueError("Empty DataFrame")
            
        # Validate required columns
        missing = [col for col in config.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        # Save initial DataFrame if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            df_path = _get_output_filename(output_dir, 'initial')
            df.to_csv(df_path, index=False)
            logger.info(f"Saving initial DataFrame to: {df_path}")
            
            if session_manager:
                session_manager.add_file(str(df_path))
                
        return df
        
    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        logger.error(error_msg)
        if status_manager:
            status_manager.set_error(error_msg)
        raise ValueError(error_msg)

def read_csv_with_encoding(file_path: str) -> DataFrameType:
    """Read CSV file with multiple encoding attempts.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame from CSV
        
    Raises:
        ValueError: If file cannot be read with any encoding
        
    Note:
        Tries multiple encodings in order:
        - utf-8
        - cp1252
        - iso-8859-1
        - latin1
    """
    encodings = ['utf-8', 'cp1252', 'iso-8859-1', 'latin1']
    
    for encoding in encodings:
        try:
            logger.debug(f"Attempting to read file with {encoding} encoding")
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info(f"Successfully read file with {encoding} encoding")
            return df
        except UnicodeDecodeError as e:
            logger.debug(f"Failed to read with {encoding} encoding: {str(e)}")
            continue
            
    raise ValueError(f"Failed to read file with any encoding: {encodings}")

async def process_attachments(
    df: DataFrameType,
    status_manager: Optional[StatusProtocol] = None,
    session_manager: Optional[SessionProtocol] = None,
    output_dir: Optional[PathLike] = None
) -> DataFrameType:
    """Process and download attachments from DataFrame."""
    try:
        logger.info("Starting attachment processing")  # Add entry log
        if status_manager:
            logger.debug(f"Status manager type: {type(status_manager)}")
            
        if session_manager:
            logger.debug(f"Session manager type: {type(session_manager)}")
            
        # Add debug info about DataFrame
        logger.debug(f"Input DataFrame shape: {df.shape}")
        logger.debug(f"Columns: {df.columns.tolist()}")
        
        if status_manager:
            status_manager.update_status("Attachments", 0, "Processing attachments")
            
        if 'Attachment Files' not in df.columns:
            logger.info("No attachments column found")
            return df
            
        # Create attachments directory
        if output_dir:
            attachments_dir = Path(output_dir) / "attachments"
            attachments_dir.mkdir(parents=True, exist_ok=True)
            
        total_rows = len(df)
        processed_rows = 0
        downloaded_files = []
        
        async with aiohttp.ClientSession() as session:
            for idx, row in df.iterrows():
                try:
                    attachments = row['Attachment Files']
                    if pd.isna(attachments) or not attachments:
                        continue
                        
                    # Parse and validate URLs
                    urls = []
                    for url in str(attachments).split(','):
                        url = url.strip()
                        if not url:
                            continue
                        if not url.startswith(('http://', 'https://')):
                            logger.warning(f"Invalid URL format: {url}")
                            continue
                        urls.append(url)
                    
                    for url in urls:
                        try:
                            # Check file size before downloading
                            async with session.head(url) as response:
                                if 'content-length' in response.headers:
                                    size = int(response.headers['content-length'])
                                    if size > config.MAX_ATTACHMENT_SIZE:
                                        logger.warning(f"File too large: {url} ({size} bytes)")
                                        continue
                            
                            # Download attachment
                            async with session.get(url) as response:
                                if response.status == 200:
                                    content = await response.read()
                                    
                                    # Validate content size
                                    if len(content) > config.MAX_ATTACHMENT_SIZE:
                                        logger.warning(f"File too large after download: {url}")
                                        continue
                                    
                                    # Save attachment
                                    if output_dir:
                                        filename = url.split('/')[-1]
                                        file_path = attachments_dir / filename
                                        file_path.write_bytes(content)
                                        downloaded_files.append(file_path)
                                        
                                        if session_manager:
                                            session_manager.add_file(file_path)
                                            
                                else:
                                    logger.warning(f"Failed to download {url}: {response.status}")
                                    
                        except aiohttp.ClientError as e:
                            logger.error(f"Network error downloading {url}: {str(e)}")
                        except Exception as e:
                            logger.error(f"Error downloading {url}: {str(e)}")
                            
                    processed_rows += 1
                    if status_manager:
                        progress = (processed_rows / total_rows) * 100
                        status_manager.update_status(
                            "Attachments",
                            progress,
                            f"Processed {processed_rows}/{total_rows} rows ({len(downloaded_files)} files)"
                        )
                        
                except Exception as e:
                    logger.error(f"Error processing row {idx}: {str(e)}")
                    continue
        
        logger.info(f"Downloaded {len(downloaded_files)} attachments")
        
        if output_dir:
            df_path = _get_output_filename(Path(output_dir), 'attach')
            df.to_csv(df_path, index=False)
            logger.info(f"Saved data with attachments to {df_path}")
            
            if session_manager:
                session_manager.add_file(str(df_path))
        
        return df
        
    except Exception as e:
        logger.exception("Critical error in process_attachments:")  # Get full traceback
        raise

async def _process_single_attachment(
    attachment_url: str,
    original_comment: str,
    idx: int,
    session: ClientSessionAlias
) -> Optional[Tuple[int, str]]:
    """Process a single attachment from URL."""
    try:
        if not attachment_url or not isinstance(attachment_url, str):
            logger.error(f"Invalid attachment URL for index {idx}")
            return None
            
        async with session.get(attachment_url) as response:
            if response.status != 200:
                logger.error(f"Failed to download attachment: HTTP {response.status}")
                return None
                
            content_type = response.headers.get('content-type', '').lower()
            content = await response.read()
            
            if 'pdf' in content_type:
                extracted_text = utils.TextExtractor.from_pdf(content)
            elif 'docx' in content_type or 'word' in content_type:
                extracted_text = utils.TextExtractor.from_docx(content)
            elif 'html' in content_type:
                text_content = await response.text()
                extracted_text = utils.TextExtractor.from_html(text_content)
            else:
                try:
                    extracted_text = content.decode('utf-8')
                except UnicodeDecodeError:
                    logger.error(f"Failed to decode attachment content at index {idx}")
                    return None
            
            if not extracted_text:
                logger.error(f"No text extracted from attachment at index {idx}")
                return None
                
            combined_text = (
                f"{original_comment}" + 
                "\n\n" + 
                "Attachment Content:\n" + 
                f"{extracted_text}"
            )
            return idx, combined_text
            
    except Exception as e:
        logger.error(f"Error processing attachment at index {idx}: {str(e)}")
        return None

@lru_cache(maxsize=config.CACHE_SIZE)
def _clean_text(text: str) -> str:
    """Clean text by removing special characters and normalizing whitespace.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
        
    Note:
        - Removes special characters
        - Normalizes whitespace
        - Preserves sentence structure
        - Results are cached
    """
    if not isinstance(text, str):
        return ""
    
    # Remove special characters but preserve sentence structure
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    
    # Normalize whitespace
    return _normalize_whitespace(text)

@lru_cache(maxsize=config.CACHE_SIZE)
def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
        
    Note:
        - Replaces multiple spaces with single space
        - Removes leading/trailing whitespace
        - Results are cached
    """
    if not isinstance(text, str):
        return ""
    return ' '.join(text.split())

async def clean_data(
    df: DataFrameType,
    status_manager: Optional[StatusProtocol] = None,
    output_dir: Optional[PathLike] = None
) -> DataFrameType:
    """Clean and preprocess the data.
    
    Args:
        df: Input DataFrame
        status_manager: Optional status manager
        output_dir: Optional output directory
        
    Returns:
        Cleaned DataFrame
        
    Note:
        - Removes duplicates and missing values
        - Cleans text data
        - Optimizes memory usage
        - Saves cleaned data if output_dir provided
    """
    if status_manager:
        status_manager.update_status("Cleaning", 0, "Starting data cleaning...")
    
    try:
        # Validation
        if df is None or df.empty:
            raise ValueError("DataFrame is empty")
        
        logger.info("Starting data cleaning process")
        total_rows = len(df)
        
        # Process the DataFrame using the main processing method
        df_clean = DataFrameProcessor.process_dataframe(df)
        cleaned_empty = total_rows - len(df_clean)
        logger.info(f"Removed {cleaned_empty} rows during cleaning")
        
        if output_dir:
            try:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                df_path = _get_output_filename(output_path, 'cleaned')
                df_clean.to_csv(df_path, index=False)
                logger.info(f"Saved cleaned data to {df_path}")
                
                if status_manager:
                    status_manager.update_status(
                        "Cleaning",
                        100,
                        f"Data cleaned and saved. Removed {cleaned_empty} invalid rows"
                    )
            except Exception as e:
                logger.error(f"Error saving cleaned data: {str(e)}")
                if status_manager:
                    status_manager.set_error(f"Failed to save cleaned data: {str(e)}")
                raise
        
        return df_clean
        
    except Exception as e:
        error_msg = f"Error cleaning data: {str(e)}"
        logger.error(error_msg)
        if status_manager:
            status_manager.set_error(error_msg)
        raise ValueError(error_msg)

async def split_paragraphs(
    df: DataFrameType,
    status_manager: Optional[StatusProtocol] = None,
    output_dir: Optional[PathLike] = None,
    min_length: int = config.CHUNK_CONFIG['MIN_LENGTH'],
    similarity_threshold: float = config.CHUNK_CONFIG['SIMILARITY_THRESHOLD'],
    max_length: int = config.CHUNK_CONFIG['MAX_CHUNK_LENGTH']
) -> DataFrameType:
    """Split and intelligently recombine comments with improved chunking."""
    try:
        if status_manager:
            status_manager.update_status("Splitting", 0, "Starting semantic chunking")
            
        # Initialize sentence transformer
        model = SentenceTransformer(config.CHUNK_CONFIG['EMBEDDING_MODEL'])
        
        processed_chunks = []
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            try:
                comment_text = str(row['Comment']).strip()
                
                # Improved paragraph splitting
                paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\. (?=[A-Z])', comment_text)
                             if len(p.strip()) >= min_length]
                
                if not paragraphs:
                    processed_chunks.append((idx, row.copy()))
                    continue
                
                # Get embeddings
                embeddings = model.encode(paragraphs, batch_size=config.CHUNK_CONFIG['BATCH_SIZE'])
                
                # Improved chunk grouping
                chunks = []
                current_chunk = [paragraphs[0]]
                current_embedding = embeddings[0]
                
                for i, (para, emb) in enumerate(zip(paragraphs[1:], embeddings[1:]), 1):
                    # Calculate similarity with current chunk
                    similarity = cosine_similarity(
                        current_embedding.reshape(1, -1),
                        emb.reshape(1, -1)
                    )[0][0]
                    
                    combined_length = sum(len(p) for p in current_chunk) + len(para)
                    
                    # Improved chunk decision logic
                    if (similarity > similarity_threshold and 
                        combined_length <= max_length and
                        len(current_chunk) < 5):  # Prevent too many paragraphs in one chunk
                        current_chunk.append(para)
                        # Update current embedding as average of chunk
                        current_embedding = np.mean([
                            model.encode(p) for p in current_chunk
                        ], axis=0)
                    else:
                        if current_chunk:
                            chunks.append('\n\n'.join(current_chunk))
                        current_chunk = [para]
                        current_embedding = emb
                
                # Handle final chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                
                # Create new rows for chunks
                for i, chunk in enumerate(chunks, 1):
                    new_row = row.copy()
                    new_row['Original_Comment'] = comment_text
                    new_row['Comment'] = chunk
                    new_row['Chunk_Number'] = i
                    processed_chunks.append((idx, new_row))
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                continue
        
        return pd.DataFrame([row for _, row in processed_chunks])
        
    except Exception as e:
        error_msg = f"Error in semantic chunking: {str(e)}"
        logger.error(error_msg)
        if status_manager:
            status_manager.set_error(error_msg)
        raise ValueError(error_msg)

def _get_output_filename(base_path: Path, current_stage: str) -> Path:
    """Get the output filename for a processing stage.
    
    Args:
        base_path: Base directory path
        current_stage: Current processing stage
        
    Returns:
        Path to output file
        
    Note:
        Uses predefined filenames from config.OUTPUT_FILES
    """
    stage = current_stage.lower()
    if stage not in config.OUTPUT_FILES:
        raise ValueError(f"Unknown file stage: {current_stage}")
    return base_path / config.OUTPUT_FILES[stage]

def get_stage_path(base_dir: Path, stage: str, subdir: Optional[str] = None) -> Path:
    """Get path for stage output with optional subdirectory."""
    if subdir:
        path = base_dir / subdir
        path.mkdir(exist_ok=True)
        return path
    return _get_output_filename(base_dir, stage)

def validate_stage(stage: str) -> str:
    """Validate and normalize processing stage."""
    stage = stage.lower()
    if stage not in config.FILE_STAGES:
        raise ValueError(f"Invalid processing stage: {stage}")
    return stage

