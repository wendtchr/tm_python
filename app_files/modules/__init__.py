"""Topic Modeling Application Modules."""

# Version
__version__ = '0.1.0'

# Core components - import these first as they have fewer dependencies
from .config import (
    BASE_OUTPUT_DIR,
    APP_FILES_DIR,
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
    TEMP_DIR,
    CACHE_DIR,
    DATA_DIR
)

from .core_types import StatusManager
from .app_core import SessionManager, PathManager

# Feature components - import these after core components
from .topic_modeling import TopicModeler
from .visualization import VisualizationService
from .ui import create_ui
from .server import create_server

# Define public API
__all__ = [
    'SessionManager',
    'StatusManager', 
    'PathManager',
    'TopicModeler',
    'VisualizationService',
    'create_ui',
    'create_server',
    'BASE_OUTPUT_DIR',
    'APP_FILES_DIR',
    'REQUIRED_COLUMNS',
    'OPTIONAL_COLUMNS',
    'TEMP_DIR',
    'CACHE_DIR',
    'DATA_DIR'
]
