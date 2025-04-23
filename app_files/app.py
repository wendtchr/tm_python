from __future__ import annotations

# Environment variables must come after __future__ imports
import os
# Set environment variables before other imports
os.environ["WATCHFILES_DISABLE"] = "1"  # Completely disable watchfiles
os.environ["SHINY_DISABLE_RELOAD"] = "1"  # Disable Shiny's reload
os.environ["SHINY_DEV_MODE"] = "0"  # Disable development mode
os.environ["WATCHFILES_FORCE_POLLING"] = "0"  # Disable file system polling
os.environ["SHINY_NO_RELOAD"] = "1"  # Alternative reload disable

# Now import other modules
import logging
import sys
from pathlib import Path
from typing import Callable
import shiny as sh
from shiny import App, Inputs, Outputs, Session, ui, reactive
from starlette.staticfiles import StaticFiles
import shinyswatch

# Import from modules package
from modules import (
    SessionManager, 
    create_ui,
    create_server,
    BASE_OUTPUT_DIR,
    APP_FILES_DIR  # Import APP_FILES_DIR instead of using PROJECT_ROOT
)

# Configure logging - use APP_FILES_DIR
LOG_FILE = APP_FILES_DIR / 'app.log'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)

logger = logging.getLogger(__name__)

# Suppress verbose loggers
mpl_logger = logging.getLogger('matplotlib.font_manager')
mpl_logger.setLevel(logging.ERROR)
logging.getLogger('umap').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('bertopic').setLevel(logging.WARNING)
logging.getLogger('watchfiles').setLevel(logging.ERROR)

# Initialize application session manager
session_manager = SessionManager(BASE_OUTPUT_DIR)

# Add this at the top of app.py, just for debugging
import sys
print(f"Running with Python: {sys.executable}")
print(f"App.py location: {__file__}")

def debug_paths() -> None:
    """Debug application paths on startup."""
    logger.info("Application Paths:")
    logger.info(f"APP_FILES_DIR: {APP_FILES_DIR}")
    logger.info(f"BASE_OUTPUT_DIR: {BASE_OUTPUT_DIR}")
    logger.info(f"LOG_FILE: {LOG_FILE}")

# Call debug function early in startup
debug_paths()

def server(session_manager: SessionManager) -> Callable[[Inputs, Outputs, Session], None]:
    """Create server function with theme picker and session management.
    
    Args:
        session_manager: Session manager instance for handling file and state management
        
    Returns:
        Server function for Shiny application
    """
    def server(input: Inputs, output: Outputs, session: Session) -> None:
        logger.info(f"New session started: {session.id}")
        shinyswatch.theme_picker_server()
        try:
            create_server(session_manager)(input, output, session)
        except sh.types.SilentException:
            logger.warning("SilentException encountered and handled.")
    return server

def create_app(session_manager: SessionManager) -> sh.App:
    """Create the Shiny Application.
    
    Args:
        session_manager: Session manager for handling file operations
        
    Returns:
        sh.App: Configured Shiny application instance
        
    Raises:
        RuntimeError: If required directories don't exist
    """
    # Verify directories exist
    www_dir = APP_FILES_DIR / "www"
    if not www_dir.exists():
        raise RuntimeError(f"Static assets directory not found: {www_dir}")
    
    if not BASE_OUTPUT_DIR.exists():
        logger.warning(f"Outputs directory not found: {BASE_OUTPUT_DIR}")
        BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Configure static assets with both www and outputs directories
    # Note: Mount points MUST start with '/'
    static_dirs = {
        "/www": str(www_dir),  # Serve www files at /www/*
        "/outputs": str(BASE_OUTPUT_DIR)  # Serve output files at /outputs/*
    }
    
    logger.info("Configuring static directories:")
    for mount_point, directory in static_dirs.items():
        logger.info(f"  {mount_point}: {directory}")
    
    # Create Shiny app with static asset configuration
    app = sh.App(
        ui=create_ui(),
        server=server(session_manager),
        static_assets=static_dirs
    )
    
    return app

# Create the app instance 
app = create_app(session_manager)

# Development server configuration
if __name__ == "__main__":
    from shiny import run_app
    import warnings
    
    # Suppress all watchfiles warnings
    warnings.filterwarnings("ignore", module="watchfiles")
    
    logger.info("Starting server with ALL reload mechanisms disabled")
    
    # Explicitly disable all reload-related features
    run_app(
        app,
        reload=False,
        reload_includes=None,
        reload_excludes=None,
        reload_dirs=None,
        exclude_dirs=None,
        launch_browser=False,
        port=8000,
        host="localhost",
        _dev_mode=False,
        autoreload_warning=False
    )