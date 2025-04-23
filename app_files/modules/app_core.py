"""Core application components and configuration.

This module provides core functionality for:
- Session state management
- Status tracking and reporting
- Path and file management with symbolic link support for file serving
"""

from __future__ import annotations

from .core_types import (
    SessionState, StatusManager, PathLike,
    ModelProtocol, SessionProtocol
)
from . import config

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import os
import time
import tempfile
import shutil

logger = logging.getLogger(__name__)

class PathManager:
    """Manages file paths and directory operations.
    
    This class handles:
    - Directory creation and validation
    - Path resolution and safety checks
    """
    
    @staticmethod
    def ensure_directory(path: PathLike) -> Path:
        """Ensure directory exists and return Path object."""
        path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def ensure_absolute(path: PathLike) -> Path:
        """Convert path to absolute if not already."""
        path = Path(path)
        return path if path.is_absolute() else Path.cwd() / path
    
    @staticmethod
    def get_relative_path(path: PathLike, base: PathLike) -> Path:
        """Get path relative to a base directory."""
        return Path(path).resolve().relative_to(Path(base).resolve())
    
    @staticmethod
    def is_safe_path(path: PathLike, base: PathLike) -> bool:
        """Check if path is safe relative to base directory."""
        try:
            base_path = Path(base).resolve()
            target_path = Path(path).resolve()
            return base_path in target_path.parents or target_path == base_path
        except Exception:
            return False

class SessionManager:
    """Manages application sessions and state."""
    
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        """Initialize session manager."""
        self.base_dir = Path(base_dir) if base_dir else Path.cwd() / "outputs"
        self.session_id: Optional[str] = None
        self.session_dir: Optional[Path] = None
        self.state = SessionState.INITIALIZING
        self.status = StatusManager()
        self.data_df = None
        self.model = None
        self.files: Dict[str, Dict[str, Any]] = {}
        self.stats: Dict[str, Any] = {
            'total_size': 0,
            'file_count': 0,
            'created_at': datetime.now(),
            'last_accessed': datetime.now()
        }
        # Add configuration values
        self.max_files_per_session = config.MAX_FILES_PER_SESSION
        self.max_session_size_mb = config.MAX_SESSION_SIZE_MB
        
        # Ensure base directory exists
        PathManager.ensure_directory(self.base_dir)
        logger.info("SessionManager initialized")
        
        self.resources = []
    
    def __enter__(self):
        self.session_dir = tempfile.mkdtemp()
        self.resources.append(self.session_dir)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def create_session(self) -> None:
        """Create new session with unique ID and directory structure."""
        try:
            self.status.update_status("Creating Session", 0, "Initializing new session")
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = PathManager.ensure_directory(
                self.base_dir / self.session_id
            )
            
            # Create standard subdirectories
            for dirname in ['temp', 'visualizations', 'reports']:
                PathManager.ensure_directory(self.session_dir / dirname)
            
            self.state = SessionState.ACTIVE
            self.status.update_status(
                "Session Created",
                100,
                f"Session {self.session_id} ready"
            )
            logger.info(f"Created new session: {self.session_id}")
            
        except Exception as e:
            self.state = SessionState.ERROR
            self.status.set_error(f"Failed to create session: {str(e)}")
            logger.error(f"Session creation failed: {e}")
            raise
    
    def add_file(self, file_path: str) -> None:
        """Add a file to the session, managing storage limits."""
        if len(self.files) >= self.max_files_per_session:
            # Remove oldest files until under limit
            files_to_remove = len(self.files) - self.max_files_per_session + 1
            oldest_files = sorted(
                self.files.items(), 
                key=lambda x: x[1]['timestamp']
            )[:files_to_remove]
            
            for path, _ in oldest_files:
                self._remove_file(path)
        
        # Add new file
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        self.files[file_path] = {
            'timestamp': time.time(),
            'size': file_size
        }
    
    def validate_file_limits(self, file_size: int) -> bool:
        """Validate file against session limits."""
        file_size_mb = file_size / (1024 * 1024)
        current_size_mb = self.stats['total_size'] / (1024 * 1024)
        
        if self.stats['file_count'] >= config.MAX_FILES_PER_SESSION:
            logger.warning("Maximum file count reached")
            return False
        
        if (current_size_mb + file_size_mb) > config.MAX_SESSION_SIZE_MB:
            logger.warning("Maximum session size would be exceeded")
            return False
        
        if file_size_mb > config.FILE_SIZE_WARN_MB:
            logger.warning(f"Large file detected: {file_size_mb:.1f}MB")
        
        return True
    
    def get_session_files(self) -> Dict[str, Dict[str, Any]]:
        """Get dictionary of session files."""
        return self.files.copy()
    
    def update(self, **kwargs: Any) -> None:
        """Update session attributes."""
        for key, value in kwargs.items():
            if key == 'model' and value is not None:
                if not isinstance(value, ModelProtocol):
                    raise TypeError("Model must implement ModelProtocol")
            setattr(self, key, value)
        
        self.stats['last_accessed'] = datetime.now()
    
    def cleanup(self) -> None:
        """Clean up all session resources"""
        for res in self.resources:
            if os.path.exists(res):
                try:
                    if os.path.isdir(res):
                        shutil.rmtree(res, ignore_errors=True)
                    else:
                        os.remove(res)
                except Exception as e:
                    logger.error(f"Failed to cleanup {res}: {e}")
        self.resources.clear()
        
        try:
            self.status.update_status("Cleanup", 0, "Cleaning up session resources")
            self.data_df = None
            self.model = None
            self.files = {}
            self.state = SessionState.COMPLETED
            self.status.update_status("Cleanup", 100, "Session cleanup complete")
            logger.info(f"Cleaned up session: {self.session_id}")
            
        except Exception as e:
            self.state = SessionState.ERROR
            self.status.set_error(f"Cleanup failed: {str(e)}")
            logger.error(f"Session cleanup failed: {e}")
            raise

    def register_resource(self, path):
        """Register a resource for cleanup"""
        if path and os.path.exists(path):
            self.resources.append(path)

def register_stage_files(
    session_manager: SessionManager,
    stage: str,
    files: List[Path]
) -> None:
    """Register all files for a processing stage."""
    for file in files:
        session_manager.add_file(str(file))