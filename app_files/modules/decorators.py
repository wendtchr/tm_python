"""Decorators for standardized error handling and status updates.

This module provides decorators for:
- Error handling with consistent logging
- Status updates with progress tracking
- Async operation management
"""

from __future__ import annotations

import asyncio
import functools
import logging
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Optional, ParamSpec, TypeVar

from .core_types import (
    DataFrameType, StatusProtocol, PathLike, 
    StatusHandler, StatusEntry, BytesContent, 
    ClientSessionAlias, SessionProtocol
)

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')

def handle_errors(
    error_msg: str = "Operation failed",
    log_level: int = logging.ERROR
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for standardized error handling.
    
    Args:
        error_msg: Base error message
        log_level: Logging level for errors
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(log_level, f"{error_msg}: {str(e)}")
                raise
        return wrapper
    return decorator

def async_handle_errors(
    error_msg: str = "Operation failed",
    log_level: int = logging.ERROR
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator for standardized async error handling.
    
    Args:
        error_msg: Base error message
        log_level: Logging level for errors
        
    Returns:
        Decorated async function with error handling
    """
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.log(log_level, f"{error_msg}: {str(e)}")
                raise
        return wrapper
    return decorator

@asynccontextmanager
async def status_context(
    status_manager: Optional[StatusProtocol],
    stage: str,
    initial_progress: float = 0.0,
    final_progress: float = 100.0,
    initial_message: str = "Starting...",
    final_message: str = "Complete"
):
    """Context manager for status updates.
    
    Args:
        status_manager: Status manager instance
        stage: Current processing stage
        initial_progress: Starting progress percentage
        final_progress: Ending progress percentage
        initial_message: Starting status message
        final_message: Completion status message
        
    Yields:
        Status update function
    """
    if status_manager:
        status_manager.update_status(stage, initial_progress, initial_message)
    try:
        yield lambda progress, message: status_manager.update_status(stage, progress, message) if status_manager else None
    except Exception as e:
        if status_manager:
            status_manager.set_error(str(e))
        raise
    finally:
        if status_manager:
            status_manager.update_status(stage, final_progress, final_message)

def with_status(
    stage: str,
    initial_message: str = "Starting...",
    final_message: str = "Complete"
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Add status updates to async functions."""
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            status_manager = next(
                (arg for arg in (*args, *kwargs.values()) 
                if isinstance(arg, StatusProtocol)),
                None
            )
            
            if status_manager:
                status_manager.update_status(stage, 0.0, initial_message)
                logger.info(f"Status update: {stage} - 0.0% - {initial_message}")
            try:
                result = await func(*args, **kwargs)
                if status_manager:
                    status_manager.update_status(stage, 100.0, final_message)
                    logger.info(f"Status update: {stage} - 100.0% - {final_message}")
                return result
            except Exception as e:
                if status_manager:
                    status_manager.set_error(str(e))
                raise
        return wrapper
    return decorator