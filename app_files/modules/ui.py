"""User interface module for the Topic Modeling Application."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict

from shiny import ui
import shinyswatch

from . import config
from .core_types import ShinyTag, UIElements, PanelConfig

__all__ = ['create_ui', 'create_seed_topic_input']

logger = logging.getLogger(__name__)

def create_seed_topic_input(index: int = 1, topic_str: Optional[str] = None) -> ui.tags.div:
    """Create a single seed topic input box."""
    # Parse default topic if provided
    topic_name = ""
    topic_keywords = ""
    
    if topic_str:
        parts = [p.strip() for p in topic_str.split(',')]
        if parts:
            topic_name = parts[0]
            topic_keywords = ', '.join(parts[1:])
    
    return ui.div(
        ui.div(
            f"Topic {index}",
            ui.input_action_button(
                f"remove_topic_{index}",
                "Remove",
                class_="btn btn-sm btn-danger"
            ),
            class_="d-flex justify-content-between align-items-center"
        ),
        ui.input_text(
            f"seed_topic_{index}",  # Single input for full topic string
            "Topic and Keywords",
            value=f"{topic_name}, {topic_keywords}" if topic_name else "",
            placeholder="topic name, keyword1, keyword2, ..."
        ),
        id=f"seed_topic_{index}",
        class_="seed-topic-input p-3 mb-3"
    )

def _create_iframe(
    src: str = "",
    height: str = "",
    width: str = "",
    title: str = "",
    id: Optional[str] = None
) -> ShinyTag:
    """Create an iframe element for embedding visualizations.
    
    Args:
        src: Source URL or HTML content for the iframe
        height: Height of the iframe in pixels or percentage
        width: Width of the iframe in pixels or percentage
        title: Accessibility title for the iframe
        id: Optional unique identifier for the iframe element
        
    Returns:
        ShinyTag: An iframe element configured with the specified parameters
        
    Note:
        This helper function ensures consistent iframe creation across the UI,
        particularly for visualizations that need to be embedded.
    """
    kwargs = {'src': src, 'height': height, 'width': width}
    if id is not None:
        kwargs['id'] = id
    if title:
        kwargs['title'] = title
    return ui.tags.iframe(**kwargs)

def _create_visualizations_section() -> ShinyTag:
    """Create the visualizations section of the UI.
    
    This function generates a structured layout for all topic modeling visualizations
    including:
    - Topic summary report at the top
    - Topic visualization and word scores side by side
    - Word clouds and topic hierarchy below
    - Generated files list at the bottom
    
    The layout is organized in a responsive grid system with appropriate spacing
    and consistent styling for section headers.
    
    Returns:
        ShinyTag: A div containing all visualization components in a structured layout
    """
    sections = config.UI.COMPONENTS['sections']['results']
    return ui.div(
        # Topic Summary Report
        ui.div(
            ui.row(
                ui.column(12,
                    ui.h4(sections['topic_summary'], class_="section-title"),
                    ui.output_ui("topic_summary")
                )
            ),
            style="margin-top: 1rem;"
        ),
        # Topic Visualization and Word Scores side by side
        ui.div(
            ui.row(
                ui.column(6,
                    ui.h4(sections['topic_viz'], class_="section-title"),
                    ui.output_ui("topic_visualization")
                ),
                ui.column(6,
                    ui.h4("Word Scores", class_="section-title"),
                    ui.output_ui("word_scores_plot")
                )
            ),
            style="margin-top: 2rem;"
        ),
        # Word Cloud and Topic Hierarchy side by side
        ui.div(
            ui.row(
                ui.column(6,
                    ui.h4(sections['word_cloud'], class_="section-title"),
                    ui.output_ui("wordcloud_plot")
                ),
                ui.column(6,
                    ui.h4(sections['topic_hierarchy'], class_="section-title"),
                    ui.output_ui("topic_hierarchy_frame")
                )
            ),
            style="margin-top: 2rem;"
        ),
        # Generated Files
        ui.div(
            ui.row(
                ui.column(12,
                    ui.h4(sections['generated_files'], class_="section-title"),
                    ui.output_ui("file_list")
                )
            ),
            style="margin-top: 2rem;"
        )
    )

def _create_model_parameters() -> ShinyTag:
    """Create controls for BERTopic model parameters.
    
    Creates a form section containing:
    - Basic parameters:
        - Minimum topic size
        - N-gram range controls
        - Top words per topic
    - Advanced parameters (collapsible):
        - UMAP parameters (n_neighbors, n_components, min_dist)
    
    Returns:
        ShinyTag: A div containing all model parameter controls with:
            - Tooltips explaining each parameter
            - Appropriate input validation
            - Collapsible advanced options section
            
    Note:
        Advanced parameters are hidden by default and can be shown
        via the "Show Advanced Options" checkbox.
    """
    return ui.div(    
        # Advanced options toggle
        ui.div(
            ui.input_checkbox(
                "show_advanced",
                "Show Advanced Options",
                value=False
            ),
            class_="mb-3"
        ),
        
        # Advanced options section
        ui.div(
            # Basic parameters
            ui.div(
                ui.h4("Model Parameters", class_="mt-4"),
                
                # Minimum topic size
                ui.div(
                    ui.tags.label(
                        "Minimum Topic Size:",
                        ui.tags.i(
                            class_="fas fa-question-circle ms-1",
                            **{"data-bs-toggle": "tooltip", 
                            "title": "Minimum number of documents per topic"}
                        )
                    ),
                    ui.input_numeric(
                        "min_topic_size",
                        None,
                        value=config.TOPIC_MODELING['MIN_TOPIC_SIZE'],
                        min=2,
                        step=1
                    ),
                    class_="mb-3"
                ),
                
                # N-gram range
                ui.div(
                    ui.tags.label(
                        "N-gram Range:",
                        ui.tags.i(
                            class_="fas fa-question-circle ms-1",
                            **{"data-bs-toggle": "tooltip", 
                            "title": "Range of word combinations to consider"}
                        )
                    ),
                    ui.div(
                        ui.input_numeric(
                            "ngram_min",
                            "Min",
                            value=config.TOPIC_MODELING['NGRAM_RANGE'][0],
                            min=1,
                            max=3
                        ),
                        ui.input_numeric(
                            "ngram_max",
                            "Max",
                            value=config.TOPIC_MODELING['NGRAM_RANGE'][1],
                            min=1,
                            max=3
                        ),
                        class_="d-flex gap-2"
                    ),
                    class_="mb-3"
                ),
                
                # Top N words per topic
                ui.div(
                    ui.tags.label(
                        "Top Words per Topic:",
                        ui.tags.i(
                            class_="fas fa-question-circle ms-1",
                            **{"data-bs-toggle": "tooltip", 
                            "title": "Number of keywords to show per topic"}
                        )
                    ),
                    ui.input_numeric(
                        "top_n_words",
                        None,
                        value=config.TOPIC_MODELING['TOP_N_WORDS'],
                        min=5,
                        max=30
                    ),
                    class_="mb-3"
                ),   
                # UMAP parameters
                ui.div(
                    ui.h5("UMAP Parameters", class_="mt-3"),
                    ui.input_numeric(
                        "umap_n_neighbors",
                        "n_neighbors",
                        value=config.UMAP_N_NEIGHBORS,
                        min=5
                    ),
                    ui.input_numeric(
                        "umap_n_components",
                        "n_components", 
                        value=5,
                        min=2
                    ),
                    ui.input_numeric(
                        "umap_min_dist",
                        "min_dist",
                        value=config.UMAP_MIN_DIST,
                        min=0.0,
                        max=1.0,
                        step=0.1
                    ),
                    class_="ms-3"
                )
            ),
            id="advanced_options",
            style="display: none;"
        ),
        class_="model-parameters"
    )

def _create_chunk_controls() -> ShinyTag:
    """Create controls for chunk settings."""
    return ui.div(
        ui.div(
            ui.h4("Semantic Chunking", class_="d-inline me-2"),
            ui.input_switch(
                "enable_chunking",
                None,
                value=False
            ),
            class_="d-flex align-items-center mb-2"
        ),
        ui.div(
            ui.input_slider(
                "similarity_threshold",
                "Similarity Threshold",
                min=0.5,
                max=0.9,
                value=config.CHUNK_CONFIG['SIMILARITY_THRESHOLD'],
                step=0.05
            ),
            ui.input_numeric(
                "min_chunk_length",
                "Minimum Chunk Length",
                value=config.CHUNK_CONFIG['MIN_LENGTH'],
                min=20,
                max=200
            ),
            ui.input_numeric(
                "max_chunk_length", 
                "Maximum Chunk Length",
                value=config.CHUNK_CONFIG['MAX_CHUNK_LENGTH'],
                min=500,
                max=5000
            ),
            id="chunk_settings_panel",
            class_="chunk-settings",
            style="display: none;"  # Initially hidden
        ),
        class_="mb-3"
    )

def _create_sidebar() -> ShinyTag:
    """Create the sidebar with input controls.
    
    Creates a structured sidebar containing:
    - File upload section
    - Processing step buttons
    - Seed topics management
    - Model parameter controls
    
    Returns:
        ShinyTag: A sidebar div with organized sections for:
            - Data input and processing controls
            - Topic management interface
            - Model configuration options
            
    Note:
        Uses Bootstrap classes for consistent spacing and styling.
        All sections include help tooltips for user guidance.
    """
    return ui.sidebar(
        # File input section
        ui.div(
            ui.input_file(
                "file",
                "Upload Data File",
                accept=".csv"
            ),
            class_="mb-4"
        ),
        
        # Processing steps with chunking switch
        ui.div(
            ui.div(
                ui.h4("Processing Steps", class_="d-inline me-2"),
                ui.div(
                    ui.input_switch(
                        "enable_chunking",
                        "Semantic Chunking",
                        value=False
                    ),
                    class_="d-inline"
                ),
                class_="d-flex justify-content-between align-items-center mb-2"
            ),
            # Chunk settings panel
            ui.div(
                ui.input_slider(
                    "similarity_threshold",
                    "Similarity Threshold",
                    min=0.5,
                    max=0.9,
                    value=config.CHUNK_CONFIG['SIMILARITY_THRESHOLD'],
                    step=0.05
                ),
                ui.input_numeric(
                    "min_chunk_length",
                    "Minimum Chunk Length",
                    value=config.CHUNK_CONFIG['MIN_LENGTH'],
                    min=20,
                    max=200
                ),
                ui.input_numeric(
                    "max_chunk_length", 
                    "Maximum Chunk Length",
                    value=config.CHUNK_CONFIG['MAX_CHUNK_LENGTH'],
                    min=500,
                    max=5000
                ),
                id="chunk_settings_panel",
                class_="chunk-settings",
                style="display: none;"  # Initially hidden
            ),
            # Processing buttons
            ui.div(
                ui.input_action_button(
                    "load_data",
                    "Load Data",
                    class_="btn-primary w-100 mb-2"
                ),
                ui.input_action_button(
                    "process_attachments",
                    "Process Attachments",
                    class_="btn-primary w-100 mb-2"
                ),
                ui.input_action_button(
                    "clean_data",
                    "Clean Data",
                    class_="btn-primary w-100 mb-2"
                ),
                ui.input_action_button(
                    "run_modeling",
                    "Run Topic Modeling",
                    class_="btn-primary w-100"
                )
            ),
            class_="mb-4"
        ),
        
        # Seed topics section
        ui.div(
            ui.h4(
                "Seed Topics",
                ui.tags.i(
                    class_="fas fa-question-circle ms-1",
                    **{"data-bs-toggle": "tooltip", 
                       "title": "Define topics to guide the model"}
                )
            ),
            ui.div(
                *[create_seed_topic_input(i+1, topic_str) 
                  for i, topic_str in enumerate(config.TOPIC_MODELING['SEED_TOPICS']['DEFAULT'])],
                id="seed-topics-container"
            ),
            ui.input_action_button(
                "add_seed_topic",
                "Add Topic",
                class_="btn btn-secondary btn-sm w-100 mt-2"
            ),
            class_="mb-4"
        ),
        
        # Model parameters
        _create_model_parameters(),
        
        width="450px",
        class_="bg-light sidebar p-3"
    )

def _create_topic_comparison_content() -> ShinyTag:
    """Create the topic comparison section content.
    
    Creates a structured layout for comparing model-generated topics with 
    human-assigned topics, including:
    - Topic alignment analysis summary
    - Visual heatmap of topic alignments
    - Detailed comparison table
    
    Returns:
        ShinyTag: A div containing the comparison visualization components
        
    Note:
        This section is only displayed when human-assigned topics are available
        in the input data.
    """
    return ui.div(
        ui.h4("Topic Alignment Analysis"),
        ui.output_ui("topic_comparison_summary"),
        ui.div(
            ui.h4("Topic Alignment Heatmap"),
            ui.output_ui("topic_alignment_plot"),
            style="margin-top: 2rem;"
        ),
        ui.div(
            ui.h4("Detailed Comparison"),
            ui.output_table("topic_comparison_table"),
            style="margin-top: 2rem;"
        )
    )

def _validate_ngram_range(min_val: int, max_val: int) -> bool:
    """Validate n-gram range values.
    
    Args:
        min_val: Minimum n-gram value
        max_val: Maximum n-gram value
        
    Returns:
        bool: True if range is valid
    """
    return 1 <= min_val <= max_val <= 3

def _validate_topic_size(size: int, total_docs: int) -> bool:
    """Validate minimum topic size.
    
    Args:
        size: Minimum topic size
        total_docs: Total number of documents
        
    Returns:
        bool: True if size is valid
    """
    return 2 <= size <= total_docs // 10

def _create_table_styles() -> str:
    """Create consolidated table styles."""
    return """
    .data-table {
        width: 100%;
        border-collapse: collapse;
    }
    .data-table th {
        position: sticky;
        top: 0;
        background: #f8f9fa;
        z-index: 10;
        padding: 8px;
        border-bottom: 2px solid #dee2e6;
        padding-right: 20px;
    }
    .data-table td {
        padding: 8px;
        border-bottom: 1px solid #dee2e6;
        max-width: 0;
        vertical-align: top;
        position: relative;
        cursor: pointer;
        max-height: 100px;
        overflow: hidden;
        transition: max-height 0.3s ease-out;
        white-space: pre-wrap;
    }
    .data-table td.expanded {
        max-height: none;
    }
    .resizer {
        position: absolute;
        right: 0;
        top: 0;
        height: 100%;
        width: 4px;
        background: rgba(0, 0, 0, 0.1);
        cursor: col-resize;
        user-select: none;
    }
    .resizer:hover,
    .resizer.resizing {
        background: rgba(0, 0, 0, 0.2);
    }
    .document-info-container {
        margin-bottom: 1rem;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 4px;
        width: auto;
    }
    .document-info {
        margin: 0;
    }
    .document-info .label {
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .document-info p {
        margin: 0.25rem 0;
    }
    """

def create_enhanced_table(id: str = "comment_table") -> ui.tags.div:
    """Create enhanced table component with stage tracking and document info."""
    return ui.div(
        # Document info section
        ui.div(
            ui.output_ui("table_header"),
            class_="document-info mb-3"
        ),
        # Stage indicators
        ui.div(
            ui.output_ui("stage_indicators"),
            class_="mb-3"
        ),
        # Table container
        ui.div(
            ui.output_table(id),
            class_="table-container"
        ),
        # Table initialization script
        ui.tags.script("""
            document.addEventListener('DOMContentLoaded', function() {
                function initTable() {
                    const table = document.querySelector('.data-table');
                    if (!table) return;
                    
                    // Cell expansion
                    table.addEventListener('click', (e) => {
                        if (e.target.tagName === 'TD') {
                            e.target.classList.toggle('expanded');
                        }
                    });
                    
                    // Column resizing
                    table.querySelectorAll('th').forEach(col => {
                        const resizer = document.createElement('div');
                        resizer.className = 'resizer';
                        col.appendChild(resizer);
                        
                        let startX, startWidth;
                        
                        resizer.addEventListener('mousedown', e => {
                            startX = e.pageX;
                            startWidth = col.offsetWidth;
                            resizer.classList.add('resizing');
                            
                            const mouseMoveHandler = e => {
                                const dx = e.pageX - startX;
                                col.style.width = `${startWidth + dx}px`;
                            };
                            
                            const mouseUpHandler = () => {
                                resizer.classList.remove('resizing');
                                document.removeEventListener('mousemove', mouseMoveHandler);
                                document.removeEventListener('mouseup', mouseUpHandler);
                            };
                            
                            document.addEventListener('mousemove', mouseMoveHandler);
                            document.addEventListener('mouseup', mouseUpHandler);
                        });
                    });
                }
                
                // Initialize and handle updates
                initTable();
                new MutationObserver(initTable).observe(document.body, { 
                    childList: true, 
                    subtree: true 
                });
            });
        """)
    )

def _create_topic_modeling_options() -> ui.tags.div:
    """Create topic modeling options section."""
    return ui.div(
        ui.input_numeric(
            "num_topics",
            "Number of Topics",
            value=10,
            min=2,
            max=50
        ),
        # ... other inputs
    )

def create_ui() -> ShinyTag:
    """Create the complete Shiny user interface."""
    return ui.page_fluid(
        ui.tags.head(
            ui.tags.link(rel="stylesheet", type="text/css", href="style.css")
        ),
        
        # Title section
        ui.div(
            ui.h2("Topic Modeling Analysis"),
            class_="d-flex justify-content-between align-items-center mb-3"
        ),
        
        # Main layout with sidebar and content area
        ui.layout_sidebar(
            # Sidebar controls
            _create_sidebar(),
            
            # Main content area
            ui.div(
                # Status section
                ui.div(
                    ui.output_ui("status_message"),
                    ui.output_ui("status_history"),
                    class_="status-container"
                ),
                
                # Document info between status and table
                ui.div(
                    ui.output_ui("table_header"),
                    class_="document-info-wrapper"
                ),
                
                # Table and results
                ui.navset_tab(
                    ui.nav_panel(
                        "Comment Table",
                        ui.div(
                            ui.output_table("comment_table"),
                            class_="table-container"
                        )
                    ),
                    ui.nav_panel("Results", _create_visualizations_section()),
                    ui.nav_panel("Topic Comparison", ui.output_ui("topic_comparison_panel"))
                ),
                class_="main-content"
            )
        ),
        
        # Add JavaScript for advanced options toggle
        ui.tags.script("""
        $(document).ready(function() {
            $('#show_advanced').change(function() {
                if($(this).is(':checked')) {
                    $('#advanced_options').slideDown();
                } else {
                    $('#advanced_options').slideUp();
                }
            });
        });
        """),
        
        theme=shinyswatch.theme.flatly
    )
 
 