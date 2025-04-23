"""Core type definitions and protocols.

This module centralizes type definitions and protocols used across the application
to reduce circular dependencies and improve maintainability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Protocol, TypeAlias, Union,
    runtime_checkable, Literal, Sequence, TypedDict, TypeVar, ParamSpec, Tuple, AsyncGenerator
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from aiohttp import ClientSession
from datetime import datetime

logger = logging.getLogger(__name__)

# Type aliases for common data types
DataFrameType: TypeAlias = pd.DataFrame
SeriesType: TypeAlias = pd.Series
PathLike: TypeAlias = Union[str, Path]
ArrayLike: TypeAlias = Union[NDArray[Any], pd.Series, Sequence[Any]]
JsonDict: TypeAlias = Dict[str, Any]
BytesContent: TypeAlias = bytes
TextContent: TypeAlias = str
ClientSessionAlias: TypeAlias = ClientSession

# Type aliases for status management
StatusHandler: TypeAlias = Callable[[str, float, str], None]

# Topic modeling types
class TopicInfo(TypedDict):
    """Type definition for topic information."""
    topic_id: int
    name: str
    size: int
    keywords: List[str]
    weights: List[float]

class TopicKeywords(TypedDict):
    """Type definition for topic keywords and their weights."""
    topic_id: int
    keywords: List[str]
    weights: List[float]

class TopicDistribution(TypedDict):
    """Type definition for topic distribution data."""
    topic_id: int
    frequency: float
    label: str

class TopicVisualizationData(TypedDict):
    """Topic visualization data structure."""
    topic_info: pd.DataFrame  # From get_topic_info()
    document_topics: List[int]  # From document_topics property
    topic_words: Dict[int, List[Tuple[str, float]]]  # From get_topic()

class StatusEntry(TypedDict):
    """Status entry with timestamp and message."""
    timestamp: str
    message: str
    progress: float

class StatusDict(TypedDict):
    """Status information dictionary."""
    message: str
    type: str
    progress: float
    stage: str

# Function types
P = ParamSpec('P')
R_co = TypeVar('R_co')

class AsyncFunction(Protocol[P, R_co]):
    """Protocol for async functions."""
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R_co: ...

@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for topic modeling models.
    
    This protocol defines the required interface for topic modeling classes,
    ensuring they provide essential methods for training, inference, and
    visualization. Implementing classes must provide all these methods
    with compatible signatures.
    
    Required methods include:
        - fit_transform: Train model and transform documents
        - get_topic_info: Get information about discovered topics
        - get_topics: Get all topics and their keywords
        - get_topic: Get specific topic keywords
        - transform: Transform new documents
        - visualize_topics: Generate topic visualization
        - visualize_hierarchy: Generate topic hierarchy visualization
        - visualize_topics_over_time: Generate temporal topic visualization
    """
    def fit_transform(self, documents: List[str], **kwargs: Any) -> Any: ...
    def get_topic_info(self) -> pd.DataFrame: ...
    def get_topics(self) -> Dict[int, List[Any]]: ...
    def get_topic(self, topic_id: int) -> List[Tuple[str, float]]: ...
    def transform(self, documents: List[str], **kwargs: Any) -> Any: ...
    def visualize_topics(self) -> Any: ...
    def visualize_hierarchy(self) -> Any: ...
    def visualize_topics_over_time(self, data: DataFrameType) -> Any: ...
    @property
    def document_topics(self) -> List[int]: ...

@runtime_checkable
class StatusProtocol(Protocol):
    """Protocol for status management."""
    def set_status(self, message: str, progress: float = 0) -> None: ...
    def set_error(self, message: str) -> None: ...
    def get_status(self) -> StatusDict: ...

@runtime_checkable
class SessionProtocol(Protocol):
    """Protocol for session management."""
    def add_file(self, file_path: Path) -> None: ...
    def get_session_files(self) -> Dict[str, Path]: ...
    @property
    def session_dir(self) -> Optional[Path]: ...

# UI Types
ShinyTag = Any  # Type alias for Shiny UI tag
UIElements = List[ShinyTag]

class PanelConfig(TypedDict):
    """Configuration for UI panels."""
    title: str
    content: ShinyTag

class SessionState:
    """Enumeration of possible session states."""
    INITIALIZING = "initializing"
    ACTIVE = "active" 
    PROCESSING = "processing"
    ERROR = "error"
    COMPLETED = "completed"
    
    def __str__(self) -> str:
        """Return string representation of state."""
        return self.value

class StatusManager:
    """Manages application status and progress tracking."""
    
    def __init__(self) -> None:
        self.current_stage: str = ""
        self.progress: float = 0.0
        self.status_message: str = ""
        self.error_message: Optional[str] = None
        self.last_update: datetime = datetime.now()
        self._update_handlers: List[StatusHandler] = []
        self._history: List[StatusEntry] = []
    
    def on_update(self, handler: StatusHandler) -> StatusHandler:
        """Register a status update handler."""
        self._update_handlers.append(handler)
        return handler
    
    def update_status(self, stage: str, progress: float = 0.0, message: str = "") -> None:
        """Update current status and notify handlers."""
        self.current_stage = stage
        self.progress = min(max(float(progress), 0.0), 100.0)
        self.status_message = str(message)
        self.last_update = datetime.now()
        
        entry = {
            'timestamp': self.last_update,
            'stage': stage,
            'progress': self.progress,
            'message': message,
            'is_error': self.has_error
        }
        self._history.append(entry)
        
        # Notify handlers
        for handler in self._update_handlers:
            try:
                handler(stage, self.progress, message)
            except Exception as e:
                logger.error(f"Error in status handler: {str(e)}")
    
    def set_error(self, message: str) -> None:
        """Set error state with message."""
        self.error_message = str(message)
        self.update_status("Error", 0.0, message)
    
    def get_status(self) -> StatusDict:
        """Get current status information."""
        return {
            'message': self.status_message,
            'type': 'error' if self.has_error else 'info',
            'progress': self.progress,
            'stage': self.current_stage
        }
    
    @property
    def has_error(self) -> bool:
        """Check if there is an active error."""
        return bool(self.error_message)

    def get_stage_data(self) -> Dict[str, Any]:
        """Get current stage data with progress information."""
        stages = {
            'load': {'order': 1, 'label': 'Load Data'},
            'clean': {'order': 2, 'label': 'Clean Data'},
            'split': {'order': 3, 'label': 'Split Paragraphs'},
            'model': {'order': 4, 'label': 'Topic Modeling'}
        }
        
        current_entries = {}
        for entry in self._history:
            stage = entry['stage'].lower()
            if stage in stages:
                current_entries[stage] = {
                    'progress': entry['progress'],
                    'message': entry['message'],
                    'is_error': entry['is_error'],
                    'timestamp': entry['timestamp']
                }
        
        return {
            'stages': stages,
            'current': current_entries,
            'active_stage': self.current_stage
        }

# Add specific report types
class ReportContent(TypedDict):
    """Type definition for report content."""
    title: str
    timestamp: datetime
    topics: List[TopicInfo]
    visualizations: List[VisualizationInfo]
    metrics: Dict[str, float]

class VisualizationInfo(TypedDict):
    """Type definition for visualization metadata."""
    type: str
    path: Path
    caption: str
    dimensions: Tuple[int, int]

# Add after existing TypedDict definitions
class TopicReportSection(TypedDict):
    """Type definition for report sections."""
    title: str
    content: str
    order: int

class TopicReport(TypedDict):
    """Type definition for complete report."""
    title: str
    timestamp: str
    sections: Dict[str, TopicReportSection]
    metadata: Dict[str, Any]
