"""Server-side logic for the Shiny application."""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from shiny import Inputs, Outputs, Session, reactive, render, ui

from . import core_types
from . import config
from . import topic_modeling
from . import visualization
from .visualization import VisualizationService

logger = logging.getLogger(__name__)

"""Server module for the Shiny application.

This module provides:
- Server-side state management
- Data processing handlers
- Visualization rendering
- Status updates and error handling
- File management and URL generation
"""

import sys
import asyncio
from pathlib import Path
from typing import (
    Dict, Any, Optional, Union, Callable,
    TypeAlias, TYPE_CHECKING, List
)
from functools import partial, lru_cache  # Add if using decorators

import pandas as pd
import shiny as sh
from shiny import Inputs, Outputs, Session, reactive, render, ui
from bertopic import BERTopic

from .core_types import (
    DataFrameType, StatusProtocol, PathLike, 
    StatusHandler, StatusEntry, BytesContent, 
    ClientSessionAlias, SessionProtocol,
    ShinyTag, SessionState
)

from .app_core import SessionManager, StatusManager, PathManager
from .data_processing import (
    clean_data, process_attachments, process_file_upload, split_paragraphs, DataFrameProcessor
)
from . import topic_modeling
from .utils import parse_list_input, create_data_url
from . import visualization
from . import ui as local_ui
import shiny.ui as sui
from datetime import datetime

__all__ = ['create_server', 'handle_run_modeling']

# Type aliases for server-specific types
if TYPE_CHECKING:
    ServerFunc: TypeAlias = Callable[[Inputs, Outputs, Session], None]
    PathType: TypeAlias = Union[str, Path]

class ServerManager:
    """Manages server state and reactive values."""
    def __init__(self):
        self.state = SessionState.INITIALIZING
        # Core reactive values
        self.data_df = reactive.Value(None)
        self.model = reactive.Value(None)
        self.current_output_dir = reactive.Value(None)
        self.current_status = reactive.Value({})
        self.topic_viz_data = reactive.Value(None)
        
        # Chunking related reactive values
        self.chunking_enabled = reactive.Value(False)
        self.chunk_settings = reactive.Value({
            'similarity_threshold': config.CHUNK_CONFIG['SIMILARITY_THRESHOLD'],
            'min_length': config.CHUNK_CONFIG['MIN_LENGTH'],
            'max_length': config.CHUNK_CONFIG['MAX_CHUNK_LENGTH']
        })
        
        # Topic modeling related reactive values
        self.seed_topic_count = reactive.Value(0)
        
    def set_state(self, new_state: SessionState) -> None:
        """Update server state."""
        self.state = new_state

class TopicComparisonHandler:
    """Handles topic comparison UI updates and rendering.
    
    Manages the display and interaction of topic comparison results
    in the Shiny application interface.
    """
    
    def __init__(self, session_manager: SessionManager):
        """Initialize with session manager for file access."""
        self.session_manager = session_manager
    
    def _get_comparison_path(self, filename: str) -> Optional[Path]:
        """Get path to comparison file with validation.
        
        Args:
            filename: Name of comparison output file
            
        Returns:
            Path to file if it exists, None otherwise
            
        Note:
            Validates both session directory and file existence
        """
        if not self.session_manager.session_dir:
            return None
        path = self.session_manager.session_dir / 'topic_comparison' / filename
        return path if path.exists() else None
    
    def get_summary(self, df: Optional[pd.DataFrame]) -> ShinyTag:
        """Get topic comparison summary for display.
        
        Generates a formatted summary of topic comparison results,
        including metrics and alignments.
        
        Args:
            df: DataFrame containing topic assignments
            
        Returns:
            ShinyTag: Formatted HTML content for display
            
        Note:
            Returns appropriate message if comparison data unavailable
        """
        if df is None or 'Topic-Human' not in df.columns:
            return sui.p("No comparison data available. Ensure dataset has 'Topic-Human' column.")
            
        path = self._get_comparison_path('comparison_report.txt')
        if not path:
            return sui.p("Topic comparison analysis not available")
            
        try:
            report_text = path.read_text()
            return sui.div(
                sui.h4("Topic Comparison Analysis"),
                sui.pre(report_text, class_="report-text"),
                class_="comparison-summary"
            )
        except Exception as e:
            logger.error(f"Error reading comparison report: {str(e)}")
            return sui.p("Error loading topic comparison")
    
    def get_comparison_table(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Get topic comparison contingency table.
        
        Retrieves and formats the contingency table showing the relationship
        between model-assigned and human-assigned topics.
        
        Args:
            df: DataFrame containing topic assignments
            
        Returns:
            DataFrame: Formatted contingency table or error message
            
        Note:
            Returns single-column DataFrame with message if data unavailable
        """
        if df is None or 'Topic-Human' not in df.columns:
            return pd.DataFrame({'Message': ['No comparison data available']})
            
        try:
            path = self._get_comparison_path('topic_comparison.csv')
            if not path:
                return pd.DataFrame({'Message': ['Comparison table not available']})
                
            return pd.read_csv(path, index_col=0).fillna('')
            
        except Exception as e:
            logger.error(f"Error loading comparison table: {str(e)}")
            return pd.DataFrame({'Error': [str(e)]})

def _get_file_url(file_path: Path) -> str:
    """Generate URL for file access in UI.
    
    Args:
        file_path: Absolute path to file that needs web access
        
    Returns:
        URL string for accessing file through web interface
        
    Note:
        Uses Shiny's static file serving with /files prefix
    """
    try:
        # Get path relative to APP_FILES_DIR instead of BASE_OUTPUT_DIR
        rel_path = file_path.resolve().relative_to(config.APP_FILES_DIR)
        url = f"/files/{rel_path.as_posix()}"
        logger.debug(f"Generated URL: {url} for file: {file_path}")
        return url
    except Exception as e:
        logger.error(f"Error generating URL for {file_path}: {str(e)}")
        return ""

async def handle_run_modeling(
    data_df: DataFrameType,
    seed_topics: Optional[List[str]] = None,
    status_manager: Optional[StatusProtocol] = None,
    output_dir: Optional[PathLike] = None
) -> Any:
    """Handle topic modeling execution.
    
    Args:
        data_df: Input DataFrame
        seed_topics: Optional list of seed topics
        status_manager: Optional status manager for progress updates
        output_dir: Optional output directory for results
        
    Returns:
        Modeling results
    """
    try:
        if status_manager:
            status_manager.update_status("modeling", 0, "Running topic modeling...")
        
        # Initialize modeling parameters
        model_params = {
            "data": data_df,
            "seed_topics": seed_topics if seed_topics else [],
            "output_dir": output_dir
        }
        
        # Create and run topic modeler
        topic_modeler = topic_modeling.TopicModeler(**model_params)
        result = await topic_modeler.fit_transform_dataframe(data_df)
        
        if status_manager:
            status_manager.update_status("modeling", 100, "Topic modeling complete")
            
        return result
        
    except Exception as e:
        logger.error(f"Topic modeling failed: {str(e)}")
        if status_manager:
            status_manager.set_error(f"Topic modeling failed: {str(e)}")
        raise

def create_server(session_manager: SessionManager) -> ServerFunc:
    """Create server function with session management."""
    def server(input: Inputs, output: Outputs, session: Session) -> None:
        # Initialize state
        state = ServerManager()
        status_manager = StatusManager()
        comparison_handler = TopicComparisonHandler(session_manager)  # Initialize the handler
        
        # Add modeling state tracking
        modeling_state = reactive.Value({
            'in_progress': False,
            'last_run': None
        })
        
        # Initialize seed topic count with default topics
        seed_topic_count = reactive.Value(len(config.TOPIC_MODELING['SEED_TOPICS']['DEFAULT']))
        
        @reactive.Effect
        def handle_remove_topic() -> None:
            """Handle removal of seed topics.
            
            Monitors all remove topic buttons and handles their click events.
            Updates the topic count and removes the corresponding UI elements.
            """
            # Increased range to 20 to match our max number of seed topics
            for i in range(1, 21):  # Support up to 20 topics
                btn_id = f"remove_topic_{i}"
                if hasattr(input, btn_id) and input[btn_id]() > 0:
                    try:
                        # Remove the topic container
                        container_id = f"seed_topic_{i}"
                        logger.debug(f"Attempting to remove topic container: {container_id}")
                        ui.remove_ui(f"#{container_id}")
                        
                        # Update topic count
                        current_count = seed_topic_count.get()
                        if current_count > 0:  # Prevent negative counts
                            seed_topic_count.set(current_count - 1)
                            logger.debug(f"Removed seed topic {i}, new count: {current_count - 1}")
                        
                    except Exception as e:
                        logger.error(f"Error removing topic {i}: {str(e)}", exc_info=True)
        
        @reactive.Effect
        @reactive.event(input.add_seed_topic)
        def handle_add_seed_topic() -> None:
            """Add a new empty seed topic input box."""
            try:
                current_count = seed_topic_count.get()
                new_count = current_count + 1
                
                # Create new empty input using our local ui module
                new_input = local_ui.create_seed_topic_input(new_count)
                
                sh.ui.insert_ui(
                    selector="#seed-topics-container",
                    where="beforeEnd",
                    ui=new_input
                )
                
                seed_topic_count.set(new_count)
                logger.debug(f"Added new seed topic input box {new_count}")
                
            except Exception as e:
                logger.error(f"Error adding seed topic: {str(e)}")
                status_manager.set_error(f"Failed to add seed topic: {str(e)}")

        @status_manager.on_update
        def handle_status_update(stage: str, progress: float, message: str) -> None:
            """Handle status updates from status manager.
            
            Updates the current_status reactive value with the latest status information
            to reflect current processing stage, progress, and message.

            Args:
                stage: Current processing stage identifier
                progress: Progress percentage (0-100)
                message: Status message describing current operation
                
            Note:
                Updates reactive state which triggers UI refresh
            """
            state.set_state(SessionState.PROCESSING)
            state.current_status.set({
                "message": message,
                "type": "error" if status_manager.error_message else "info",
                "progress": progress,
                "stage": stage
            })

        @output
        @render.download(filename=lambda: "file.txt")
        def download_file() -> bytes:
            """Handle file downloads through Shiny's download interface.
            
            This function provides secure file serving by:
            - Extracting file paths from request parameters
            - Converting URL paths to system paths
            - Validating file access permissions
            - Streaming file contents safely
            - Handling errors gracefully
            
            Returns:
                File contents as bytes
                Empty bytes if file not found or access error
                
            Note:
                - Uses PathManager for secure path resolution
                - Handles Windows/Unix path differences
                - Prevents access outside base directory
                - Streams files to handle large files efficiently 
                - Logs download process for debugging
                - Returns empty bytes instead of raising errors
            """
            try:
                req_path = session.request.query_params.get("path", "")
                if not req_path:
                    logger.error("No path specified for download")
                    return b""
                    
                # Validate session ID in path
                parts = req_path.strip("/").split("/")
                if len(parts) < 1 or not parts[0].startswith("202"):
                    logger.error(f"Invalid session ID in path: {parts[0] if parts else 'none'}")
                    return b""
                    
                # Resolve and validate path using PathManager
                base_dir = PathManager.ensure_absolute(config.BASE_OUTPUT_DIR)
                full_path = base_dir / req_path
                
                if not PathManager.is_safe_path(full_path, base_dir):
                    logger.error(f"Attempted access outside base directory: {full_path}")
                    return b""
                    
                if not full_path.exists():
                    logger.error(f"File not found: {full_path}")
                    return b""
                    
                # Add file size check
                if full_path.stat().st_size > config.MAX_ATTACHMENT_SIZE:
                    logger.error(f"File too large: {full_path}")
                    return b""
                    
                return full_path.read_bytes()
                
            except Exception as e:
                logger.error(f"File download error: {str(e)}")
                return b""

        def _file_path(path: str) -> Path:
            """Validate and resolve file path for download.
            
            This function ensures secure file access by:
            - Converting relative paths to absolute paths
            - Validating paths are within allowed base directory
            - Converting URL paths to system-appropriate paths
            
            Args:
                path: Requested file path from URL parameter
                
            Returns:
                Resolved Path object for file access
                
            Raises:
                ValueError: If path is invalid or outside base directory
                
            Note:
                - Handles path separator conversion between web/system
                - Prevents directory traversal attacks
                - Ensures paths remain within base directory
            """
            base_dir = PathManager.ensure_absolute(config.BASE_OUTPUT_DIR)
            file_path = Path(path).resolve()
            if not PathManager.is_safe_path(file_path, base_dir):
                raise ValueError("Invalid file path")
            return file_path

        @output
        @render.ui
        @reactive.event(state.data_df)
        def table_header() -> ShinyTag:
            """Display document metadata as header labels."""
            df = state.data_df.get()
            if df is None or df.empty:
                return sh.ui.div()
            
            # Get values from first row
            first_row = df.iloc[0]
            agency_id = first_row.get('Agency ID', 'N/A')
            doc_type = first_row.get('Document Type', 'N/A')
            docket_id = first_row.get('Docket ID', 'N/A')
            
            return sh.ui.div(
                sh.ui.div(
                    sh.ui.div(
                        # Three columns layout
                        sh.ui.span(
                            sh.ui.span("Docket ID: ", class_="label"),
                            sh.ui.span(docket_id, class_="value"),
                            class_="info-item"
                        ),
                        sh.ui.span(
                            sh.ui.span("Document Type: ", class_="label"),
                            sh.ui.span(doc_type, class_="value"),
                            class_="info-item"
                        ),
                        sh.ui.span(
                            sh.ui.span("Agency ID: ", class_="label"),
                            sh.ui.span(agency_id, class_="value"),
                            class_="info-item"
                        ),
                        class_="document-info"
                    ),
                    class_="document-info-container"
                ),
                class_="document-info-wrapper"
            )

        @output
        @render.table
        @reactive.event(state.data_df, state.model)
        def comment_table() -> pd.DataFrame:
            """Render enhanced comment table with stage tracking."""
            df = state.data_df.get()
            if df is None:
                return pd.DataFrame()
            
            display_df = df.copy()
            stage_data = status_manager.get_stage_data()
            active_stage = stage_data['active_stage']
            
            # Define base columns without Topic columns
            base_columns = ['Authors', 'First Name', 'Last Name', 'Posted Date', 'Comment']

            # Add Topic columns only if they exist in the DataFrame
            if 'Topic_Name' in display_df.columns:
                base_columns = ['Topic_Name'] + base_columns

            # Define columns for all stages
            columns = {
                'base': base_columns,
                'load': [],
                'clean': ['Cleaned_Text'],
                'split': ['Paragraph_Number', 'Original_Comment'],
                'model': ['Topic']
            }
            
            # Get columns for current stage
            stage_cols = columns['base'] + columns.get(active_stage, [])
            available_cols = [col for col in stage_cols if col in display_df.columns]
            
            # Format data
            result_df = display_df[available_cols].copy()
            
            # Handle date formatting with NaT values
            if 'Posted Date' in result_df.columns:
                result_df['Posted Date'] = pd.to_datetime(
                    result_df['Posted Date'], 
                    errors='coerce'
                ).apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else '')
            
            # Replace all NaN values with empty string
            result_df = result_df.fillna('')
            
            return result_df

        @output
        @render.text
        def status_history_text() -> str:
            """Render formatted status history.
            
            Provides a formatted text representation of all status updates
            that have occurred during the session.
            
            Returns:
                Formatted string containing timestamped status history
                
            Note:
                Includes timestamps and progress information for each status update
            """
            return status_manager.get_formatted_history()
            
        @output
        @render.table
        @reactive.event(state.model)
        def results_table() -> pd.DataFrame:
            """Display topic modeling summary table.
            
            Returns:
                pd.DataFrame: Table containing topic summaries with:
                - Topic ID
                - Size (document count)
                - Top Words (most relevant words)
                - Topic Name (formatted topic identifier)
            """
            try:
                model = state.model.get()
                if not model or not isinstance(model.model, BERTopic):
                    return pd.DataFrame({'Message': ['No topic model available']})
                
                # Get topic info directly from BERTopic model
                topic_info = model.model.get_topic_info()
                summary = []
                
                for _, row in topic_info.iterrows():
                    topic_id = row['Topic']
                    if topic_id != -1:  # Skip outlier topic
                        try:
                            # Get topic words directly from BERTopic
                            words = model.model.get_topic(topic_id)
                            top_words = ", ".join(word for word, _ in words[:3])
                            summary.append({
                                'Topic': topic_id,
                                'Size': row['Count'],
                                'Top Words': top_words,
                                'Name': f"Topic {topic_id}: {top_words}"
                            })
                        except Exception as e:
                            logger.warning(f"Error processing topic {topic_id}: {str(e)}")
                            continue
                
                return pd.DataFrame(summary)
                
            except Exception as e:
                logger.error(f"Error generating results table: {str(e)}")
                return pd.DataFrame({'Error': [str(e)]})

        @output
        @render.ui
        @reactive.event(state.model)
        def topic_summary() -> ShinyTag:
            """Render topic model summary report."""
            model = state.model.get()
            if not model:
                return sh.ui.p("No model available")

            try:
                # Get topic info for summary
                topic_info = model.model.get_topic_info()
                if topic_info.empty:
                    return sh.ui.p("No topics available")

                # Create summary report
                total_docs = topic_info['Count'].sum()
                outliers = topic_info[topic_info['Topic'] == -1]['Count'].sum()
                
                summary_div = sh.ui.div(
                    # Model overview
                    sh.ui.div(
                        sh.ui.h4("Topic Model Summary", class_="card-header"),
                        sh.ui.div(
                            sh.ui.p(f"Total Documents: {total_docs}"),
                            sh.ui.p(f"Topics Found: {len(topic_info) - 1}"),
                            sh.ui.p(f"Outlier Documents: {outliers} ({(outliers/total_docs)*100:.1f}%)"),
                            class_="card-body"
                        ),
                        class_="card mb-3"
                    ),
                    # Topic details
                    sh.ui.div(
                        sh.ui.h4("Topic Details", class_="card-header"),
                        sh.ui.div(
                            sh.ui.tags.table(
                                sh.ui.tags.thead(
                                    sh.ui.tags.tr(
                                        sh.ui.tags.th("ID"),
                                        sh.ui.tags.th("Name"),
                                        sh.ui.tags.th("Documents"),
                                        sh.ui.tags.th("Keywords")
                                    )
                                ),
                                sh.ui.tags.tbody(
                                    *[
                                        sh.ui.tags.tr(
                                            sh.ui.tags.td(str(row['Topic'])),
                                            sh.ui.tags.td(model._generate_topic_names(model.document_topics).get(row['Topic'], '')),
                                            sh.ui.tags.td(str(row['Count'])),
                                            sh.ui.tags.td(
                                                ", ".join(f"{word} ({weight:.2f})" 
                                                        for word, weight in model.model.get_topic(row['Topic'])[:5])
                                            )
                                        )
                                        for _, row in topic_info.iterrows()
                                        if row['Topic'] != -1
                                    ]
                                ),
                                class_="table table-striped table-bordered"
                            ),
                            class_="card-body"
                        ),
                        class_="card"
                    )
                )
                return summary_div
            except Exception as e:
                logger.error(f"Error rendering topic summary: {str(e)}")
                return sh.ui.p("Error generating topic summary")

        @output
        @render.ui
        @reactive.event(state.model)
        def topic_hierarchy_frame() -> ShinyTag:
            """Render topic hierarchy visualization."""
            model = state.model.get()
            if not model:
                return sh.ui.p("No model available")

            try:
                # Get hierarchy visualization
                viz = model.get_visualization('hierarchy')
                return sh.ui.tags.iframe(
                    srcDoc=viz.to_html(),
                    height=config.UI.DIMENSIONS['plot_height'],
                    width="100%",
                    title=config.UI.COMPONENTS['sections']['results']['topic_hierarchy']
                )
            except Exception as e:
                logger.error(f"Error rendering hierarchy: {str(e)}")
                return sh.ui.p("Error generating visualization")

        @output
        @render.ui
        @reactive.event(state.model)
        def wordcloud_plot() -> ShinyTag:
            """Render word cloud visualization."""
            model = state.model.get()
            if not model:
                return sh.ui.p("No model available")

            try:
                # Get first non-outlier topic
                topic_info = model.model.get_topic_info()
                topic_ids = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()
                
                if not topic_ids:
                    return sh.ui.p("No valid topics found")
                
                # Generate wordcloud for first topic
                wordcloud = model.get_wordcloud(topic_ids[0])
                return sh.ui.tags.iframe(
                    srcDoc=wordcloud.to_html(),
                    height=config.UI.DIMENSIONS['plot_height'],
                    width="100%",
                    title=f"{config.UI.COMPONENTS['sections']['results']['word_cloud']} - Topic {topic_ids[0]}"
                )
            except Exception as e:
                logger.error(f"Error rendering wordcloud: {str(e)}")
                return sh.ui.p("Error generating visualization")

        @output
        @render.ui
        def topic_visualization() -> ShinyTag:
            """Render topic visualization."""
            logger.debug("Rendering topic visualization")
            model = state.model.get()
            if not model:
                logger.warning("No model available")
                return sh.ui.p("No model available")
            
            try:
                viz_service = VisualizationService(session_manager.session_dir)
                logger.debug("Created visualization service")
                
                fig = viz_service.get_topic_visualization(model)
                logger.debug("Generated visualization")
                
                return sh.ui.div(
                    sh.ui.tags.iframe(
                        srcDoc=fig.to_html(include_plotlyjs=True),
                        style="width: 100%; height: 600px; border: none;"
                    ),
                    class_="visualization-container"
                )
            except Exception as e:
                logger.error(f"Error rendering visualization: {str(e)}", exc_info=True)
                return sh.ui.p(f"Error: {str(e)}")

        @output
        @render.ui
        @reactive.event(state.model)
        def word_scores_plot() -> ShinyTag:
            """Render word scores visualization."""
            model = state.model.get()
            if not model:
                logger.error("No model available for word scores")
                return sh.ui.p("No model available")

            try:
                # Get first non-outlier topic
                topic_info = model.model.get_topic_info()
                topic_ids = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()
                
                if not topic_ids:
                    return sh.ui.p("No valid topics found")
                
                # Generate word scores visualization
                word_scores = model.get_word_scores(topic_ids[0])
                return sh.ui.tags.iframe(
                    srcDoc=word_scores.to_html(),
                    height="800px",  # Fixed height
                    width="100%",
                    title=f"Word Scores - Topic {topic_ids[0]}"
                )
            except Exception as e:
                logger.error(f"Error rendering word scores: {str(e)}")
                return sh.ui.p("Error generating visualization")
        
        @output
        @render.ui
        def status_message() -> ShinyTag:
            """Enhanced status display with progress.
            
            Renders a status message with optional progress bar.
            
            Returns:
                ShinyTag: UI element containing status message and progress bar
                
            Note:
                - Displays alert with contextual styling based on status type
                - Shows progress bar when progress > 0
                - Handles missing status gracefully
            """
            current_status = state.current_status.get()
            if not current_status:
                return sh.ui.div()
                
            status_type = current_status.get('type', 'info')
            progress = current_status.get('progress', 0)
            message = current_status.get('message', '')
            
            return sh.ui.div(
                sh.ui.tags.div(
                    sh.ui.tags.div(
                        message,
                        class_=f"alert alert-{status_type}"
                    ),
                    sh.ui.tags.div(
                        sh.ui.tags.div(
                            class_="progress-bar",
                            style=f"width: {progress}%",
                            role="progressbar",
                            **{"aria-valuenow": str(progress)}
                        ),
                        class_="progress"
                    ) if progress > 0 else None
                )
            )
        # Event Handlers
        @reactive.Effect
        @reactive.event(input.load_data)
        async def handle_load_data() -> None:
            """Handle data file upload and initial processing."""
            try:
                # Add debug logging for file input
                logger.info("Checking for uploaded files...")
                file_info = input.file()
                
                if not file_info or not file_info[0]:
                    logger.error("No file detected in input.file()")
                    raise ValueError("No file uploaded")

                file_path = file_info[0]['datapath']
                logger.info(f"Processing file at temp path: {file_path}")
                
                status_manager.update_status("Data Loading", 0, "Initializing data loading process")
                session_manager.create_session()

                base_output_dir = session_manager.session_dir
                status_manager.update_status("Data Loading", 10, "Reading uploaded file")
                df = await process_file_upload(
                    file_path,
                    status_manager=status_manager,
                    output_dir=base_output_dir
                )

                # Save initial DataFrame with consistent naming
                df_path = base_output_dir / "df_initial.csv"
                df.to_csv(df_path, index=False)
                logger.info(f"Saved initial DataFrame to {df_path}")
                session_manager.add_file(df_path)
                
                session_manager.update(data_df=df)
                state.data_df.set(df)
                state.current_output_dir.set(base_output_dir)
                status_manager.update_status("Data Loading", 100, "Data loaded successfully")

            except FileNotFoundError:
                logger.error("Error loading data: File not found", exc_info=True)
                status_manager.set_error("Error loading data: File not found")
            except ValueError as ve:
                logger.error(f"Error loading data: {str(ve)}", exc_info=True)
                status_manager.set_error(f"Error loading data: {str(ve)}")
            except Exception as e:
                logger.error(f"Unexpected error loading data: {str(e)}", exc_info=True)
                status_manager.set_error(f"Unexpected error loading data: {str(e)}")
        
        @reactive.Effect
        @reactive.event(input.process_attachments)
        async def handle_process_attachments() -> None:
            """Handle attachment processing."""
            try:
                df = state.data_df.get()
                if df is None or df.empty:
                    raise ValueError("No data to process attachments")
                    
                processed_df = await process_attachments(
                    df,
                    status_manager=status_manager,
                    output_dir=session_manager.session_dir
                )
                
                # Update both state and session
                state.data_df.set(processed_df)
                session_manager.update(data_df=processed_df)
                
            except Exception as e:
                logger.error(f"Error processing attachments: {str(e)}")
                status_manager.set_error(f"Error processing attachments: {str(e)}")
        
        @reactive.Effect
        @reactive.event(input.clean_data)
        async def handle_clean_data() -> None:
            """Handle data cleaning process."""
            try:
                df = state.data_df.get()
                if df is None or df.empty:
                    status_manager.set_error("No data to clean")
                    return
                    
                status_manager.update_status("Data Cleaning", 0, "Starting data cleaning")
                
                # Clean data using clean_data function which handles file saving
                df_clean = await clean_data(
                    df,
                    status_manager=status_manager,
                    output_dir=session_manager.session_dir
                )
                
                if df_clean.empty:
                    status_manager.set_error("Cleaning resulted in empty dataset")
                    return
                
                # Update both state and session
                state.data_df.set(df_clean)
                session_manager.update(data_df=df_clean)
                
                status_manager.update_status(
                    "Data Cleaning", 
                    100, 
                    f"Data cleaned successfully. Rows: {len(df)} -> {len(df_clean)}"
                )
                
            except Exception as e:
                logger.error(f"Error cleaning data: {str(e)}")
                status_manager.set_error(f"Error cleaning data: {str(e)}")
        
        @reactive.Effect
        @reactive.event(input.run_modeling)
        async def _handle_run_modeling() -> None:
            """Handle topic modeling process."""
            try:
                logger.info("=== Starting Modeling Process ===")
                current_df = state.data_df.get()
                
                if current_df is None:
                    raise ValueError("No data loaded")
                
                # Get model parameters with defaults
                model_params = {
                    'status_manager': status_manager,
                    'config_dict': config.TOPIC_MODELING.copy(),  # Make copy to avoid modifying original
                    'num_topics': "auto"
                }
                
                # Collect seed topics from UI
                seed_topics = []
                default_topics = config.TOPIC_MODELING['SEED_TOPICS']['DEFAULT']
                
                # Try to get topics from UI, fall back to defaults if empty
                for i in range(len(default_topics)):
                    topic_input = getattr(input, f'seed_topic_{i+1}', lambda: None)()
                    if topic_input and topic_input.strip():
                        seed_topics.append(topic_input)
                    elif i < len(default_topics):  # Use default if available
                        seed_topics.append(default_topics[i])
                
                if seed_topics:
                    model_params['seed_topics'] = seed_topics
                    logger.info(f"Using {len(seed_topics)} seed topics")
                    for i, topic in enumerate(seed_topics):
                        logger.debug(f"Seed topic {i}: {topic}")
                
                # Create and run topic modeler
                topic_modeler = topic_modeling.TopicModeler(**model_params)
                processed_df = await topic_modeler.fit_transform_dataframe(current_df)
                
                # Save outputs
                if session_manager.session_dir:
                    results = await topic_modeling.save_topic_modeling_outputs(
                        model=topic_modeler,
                        df=processed_df,
                        output_dir=session_manager.session_dir,
                        session_manager=session_manager
                    )
                    
                    state.model.set(results['model'])
                    state.data_df.set(results['distribution'])
                    state.current_output_dir.set(session_manager.session_dir)
                    
            except Exception as e:
                error_msg = f"Topic modeling failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                status_manager.set_error(error_msg)

        @reactive.Effect
        def update_button_states() -> None:
            """Update button states based on data availability and processing stage.
            
            Disables/enables buttons based on:
            - Data presence
            - Current processing stage
            - Previous stage completion
            """
            df = state.data_df.get()
            has_data = df is not None and len(df) > 0
            
            # Update button states
            sh.ui.update_action_button('clean_data', disabled=not has_data)
            sh.ui.update_action_button('process_attachments', disabled=not has_data)
            sh.ui.update_action_button('run_modeling', disabled=not has_data)

        # File List Display
        @output
        @render.ui
        def file_list() -> ShinyTag:
            """Display list of generated files.
            
            Returns:
                ShinyTag: UI component containing file list grouped by category
                
            Note:
                - Groups files into Data Files, Reports, and Visualizations
                - Validates file paths before displaying
                - Handles file metadata and path conversion
                - Provides download links for valid files
            """
            try:
                if not session_manager.session_dir:
                    return sh.ui.div("No files generated yet")
                
                files = session_manager.get_session_files()
                if not files:
                    return sh.ui.div("No files generated yet")
                
                # Convert and validate paths
                file_paths = []
                for file_info in files.values():
                    try:
                        # Handle both string paths and file info dictionaries
                        if isinstance(file_info, dict):
                            path_str = file_info.get('path', '')
                            if not path_str:
                                continue
                        else:
                            path_str = str(file_info)
                        
                        path = Path(path_str)
                        if path.exists():
                            file_paths.append(path)
                        else:
                            logger.warning(f"File not found: {path}")
                        
                    except Exception as e:
                        logger.warning(f"Invalid path info: {file_info} - {str(e)}")
                
                if not file_paths:
                    return sh.ui.div("No valid files found")
                
                # Group files by category with better logging
                categories = {
                    'Data Files': [p for p in file_paths if p.suffix.lower() == '.csv'],
                    'Reports': [p for p in file_paths if p.suffix.lower() == '.html'],
                    'Visualizations': [p for p in file_paths if 'visualizations' in str(p)]
                }
                
                logger.info(f"Found files by category: {dict((k, len(v)) for k, v in categories.items())}")
                
                elements = []
                for category, paths in categories.items():
                    if paths:
                        elements.extend([
                            sh.ui.h4(category),
                            sh.ui.tags.ul([
                                sh.ui.tags.li(
                                    sh.ui.tags.a(
                                        p.name,
                                        href=f"/download?path={p.relative_to(session_manager.session_dir)}",
                                        target="_blank"
                                    )
                                ) for p in sorted(paths)
                            ])
                        ])
                
                return sh.ui.div(*elements) if elements else sh.ui.div("No files to display")
                
            except Exception as e:
                logger.error(f"Error generating file list: {str(e)}")
                return sh.ui.div(f"Error listing files: {str(e)}")
        
        # Cleanup
        @reactive.Effect
        def cleanup_on_session_end() -> None:
            """Clean up resources when session ends.
            
            Handles:
            - Temporary file cleanup
            - Model cleanup
            - Memory management
            """
            session.on_ended(lambda: session_manager.cleanup())
        
        @output
        @render.ui
        def topic_comparison_panel() -> ShinyTag:
            """Conditionally render topic comparison content."""
            df = state.data_df.get()
            if df is None or 'Topic-Human' not in df.columns:
                return sui.p("Human-assigned topics not available")
            return local_ui._create_topic_comparison_content()
        
        # Comparison outputs
        @output
        @render.ui
        def topic_comparison_summary() -> ShinyTag:
            return comparison_handler.get_summary(state.data_df.get())

        @output
        @render.ui
        def topic_alignment_plot() -> ShinyTag:
            return comparison_handler.get_alignment_plot(state.data_df.get())

        @output
        @render.table
        def topic_comparison_table() -> pd.DataFrame:
            return comparison_handler.get_comparison_table(state.data_df.get())

        @reactive.Effect
        def _validate_model_params() -> None:
            """Validate model parameters reactively."""
            ngram_min = input.ngram_min()
            ngram_max = input.ngram_max()
            
            if not local_ui._validate_ngram_range(ngram_min, ngram_max):
                ui.update_numeric(
                    "ngram_max",
                    value=max(ngram_min, ngram_max),
                    min=ngram_min
                )

        @output
        @render.ui
        def stage_indicators() -> ui.tags.div:
            """Render stage progression indicators."""
            stage_data = status_manager.get_stage_data()
            stages = stage_data['stages']
            current = stage_data['current']
            active_stage = stage_data['active_stage']
            
            indicators = []
            for stage_id, info in stages.items():
                stage_class = "stage-pending"
                if stage_id in current:
                    stage_class = "stage-complete" if current[stage_id]['progress'] >= 100 else "stage-active"
                
                indicators.append(
                    ui.span(
                        info['label'],
                        class_=f"stage-indicator {stage_class}"
                    )
                )
            
            return ui.div(
                *indicators,
                class_="d-flex align-items-center"
            )

        @reactive.Effect
        def _update_chunk_settings_visibility() -> None:
            """Show/hide chunk settings based on switch."""
            enabled = input.enable_chunking()
            state.chunking_enabled.set(enabled)
            logger.debug(f"Chunking enabled: {enabled}")  # Add debug logging

        @reactive.Effect
        def _handle_seed_topics():
            """Handle seed topic updates."""
            seed_topics = input.seed_topics()
            if seed_topics:
                try:
                    # Parse and validate seed topics
                    topics = [t.strip() for t in seed_topics.split('\n') if t.strip()]
                    state.seed_topic_count.set(len(topics))
                    
                    # Update model if needed
                    if state.model.get():
                        state.model.get().update_seed_topics(topics)
                except Exception as e:
                    logger.error(f"Error updating seed topics: {str(e)}")

    return server