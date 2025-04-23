"""Configuration settings for the application.

This module contains all configuration settings to prevent circular imports.
Settings are grouped by functionality and can be overridden via environment variables.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Any, Final, List, Union

logger = logging.getLogger(__name__)

# Base directories - simplified and documented
APP_FILES_DIR: Final[Path] = Path(__file__).parent.parent.resolve()  # The app_files directory
BASE_OUTPUT_DIR: Final[Path] = APP_FILES_DIR.parent / "outputs"  # Directory for all generated files

# Create outputs directory if it doesn't exist
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Application subdirectories - all under app_files
TEMP_DIR: Final[Path] = APP_FILES_DIR / "temp"  # Temporary processing files
CACHE_DIR: Final[Path] = APP_FILES_DIR / "cache"  # Cache for models and embeddings
DATA_DIR: Final[Path] = APP_FILES_DIR / "data"   # Input data files

# Directory names for organization within output folders
TEMP_DIR_NAME: Final[str] = "temp"
REPORT_DIR_NAME: Final[str] = "reports"
VIZ_DIR_NAME: Final[str] = "visualizations"

def debug_directory_structure() -> None:
    """Print directory structure for debugging."""
    logger.info("Application Directory Structure:")
    logger.info(f"APP_FILES_DIR: {APP_FILES_DIR}")
    logger.info(f"BASE_OUTPUT_DIR: {BASE_OUTPUT_DIR}")
    logger.info(f"TEMP_DIR: {TEMP_DIR}")
    logger.info(f"CACHE_DIR: {CACHE_DIR}")
    logger.info(f"DATA_DIR: {DATA_DIR}")
    
    # Check directory existence
    for name, path in [
        ("APP_FILES_DIR", APP_FILES_DIR),
        ("BASE_OUTPUT_DIR", BASE_OUTPUT_DIR),
        ("TEMP_DIR", TEMP_DIR),
        ("CACHE_DIR", CACHE_DIR),
        ("DATA_DIR", DATA_DIR)
    ]:
        logger.info(f"{name} exists: {path.exists()}")

# Call debug function when config is imported
debug_directory_structure()

# Model Parameters --------------------------------------------------
TOPIC_MODELING: Final[Dict[str, Any]] = {
    'UMAP': {
        'n_neighbors': 15,
        'min_dist': 0.1,
        'metric': 'euclidean'
    },
    'EMBEDDING': {
        'model': 'all-MiniLM-L6-v2',
        'batch_size': 64,
        'random_seed': 42
    },
    'TOPIC': {
        'nr_topics': 9,  # Changed from 'auto' to match seed topics
        'min_topic_size': 4  # Reduced to allow smaller topics
    },
    'MIN_KEYWORDS': 3,
    'MAX_KEYWORDS': 10,
    'TOP_N_WORDS': 12,
    'NGRAM_RANGE': (1, 2),
    'CALCULATE_PROBABILITIES': True,  # Changed to True for better topic assignment
    'LOW_MEMORY': False,
    'LANGUAGE': 'english',
    'HDBSCAN_METRIC': 'euclidean',
    'MIN_TOPIC_SIZE': 4,
    
    # Seed topics for guided modeling
    'SEED_TOPICS': {
        'DEFAULT': [
            "health hazards, respiratory illness, asthma, copd, cancer risk, lung disease, heart disease, breathing difficulties, chronic conditions, acute symptoms, mortality, morbidity",
            "ppe, masks, respirators, n95, protective equipment, eye protection, safety gear, protective clothing, face coverings, goggles, personal protection",
            "exposures, particulate matter, air quality, smoke inhalation, duration, intensity, concentration levels, exposure limits, chemical compounds, toxic substances, occupational exposure",
            "health equity, vulnerable populations, access to care, disparities, socioeconomic factors, environmental justice, rural communities, underserved populations, language barriers, resource allocation",
            "administrative controls, work schedules, rotation policies, break policies, outdoor work limits, air quality monitoring, communication protocols, training requirements, standard operating procedures",
            "mental health, anxiety, stress, depression, psychological impact, emotional wellbeing, trauma, mental strain, psychological distress, cognitive effects, behavioral health",
            "research needs, data gaps, study requirements, evidence base, scientific investigation, knowledge gaps, research funding, longitudinal studies, health surveys, impact assessment",
            "engineering controls, ventilation systems, filtration, hepa filters, air purifiers, building modifications, mechanical controls, isolation barriers, dust suppression, air handling",
            "wildfire constituents, smoke composition, pm2.5, carbon monoxide, volatile organic compounds, nitrogen oxides, particulate composition, chemical analysis, ash content, combustion products"
        ]
    }
}

# Text processing settings
CATEGORY_THRESHOLD: Final[float] = 0.5  # Threshold for categorical conversion
CACHE_SIZE: Final[int] = 1000  # Size of LRU cache
INITIAL_DATA_FILE: Final[str] = "initial_data.csv"
ATTACHMENTS_FILE: Final[str] = "df_with_attachments.csv"  # File for saving processed attachments

# Define required and optional columns
REQUIRED_COLUMNS: Final[set[str]] = {'Comment'}  # Only Comment is strictly required
OPTIONAL_COLUMNS: Final[set[str]] = {
    'First Name', 'Last Name', 'Posted Date', 'Authors',
    'Attachment Files', 'Document ID', 'Agency ID', 'Docket ID',
    'Tracking Number', 'Document Type', 'Topic-Human'
}
STATUS_DELAY: Final[float] = 0.1  # Delay between status updates

# File settings
MAX_ATTACHMENT_SIZE: Final[int] = 10 * 1024 * 1024  # 10MB
TOPIC_DIST_FILE: Final[str] = "topic_distribution.csv"
MAX_FILES_PER_SESSION: Final[int] = 100  # Maximum number of files per session
MAX_SESSION_SIZE_MB: Final[int] = 500  # Maximum session size in MB
FILE_SIZE_WARN_MB: Final[int] = 50  # Size in MB to trigger large file warning

# Stopwords ----------------------------------------------------------------
# Stopwords are words that are not considered for topic modeling
STOPWORDS: Final[set[str]] = {
    # Common verbs and modals
    "regarding", "concerned", "think", "need", "needs", "like", "would", "could", 
    "should", "must", "make", "made", "making", "want", "wanting", "wants", "wanted",
    "get", "getting", "gets", "got", "know", "knowing", "knows", "knew", "see", 
    "seeing", "sees", "saw", "look", "looking", "looks", "looked", "feel", "feeling",
    "feels", "felt", "believe", "believing", "believes", "believed", "seem", "seeming",
    "seems", "seemed", "appear", "appearing", "appears", "appeared", "consider",
    "considering", "considers", "considered", "find", "finding", "finds", "found",
    "going", "goes", "went", "gone", "come", "coming", "comes", "came", "take",
    "taking", "takes", "took", "taken", "give", "giving", "gives", "gave", "given",
    "put", "putting", "puts", "use", "using", "uses", "used", "try", "trying",
    "tries", "tried", "call", "calling", "calls", "called", "work", "working",
    "works", "worked", "say", "saying", "says", "said", "show", "showing", "shows",
    "showed", "shown", "ask", "asking", "asks", "asked", "tell", "telling", "tells",
    "told", "write", "writing", "writes", "wrote", "written"
}

# Visualization settings
VIZ_CONFIG: Final[Dict[str, Any]] = {
    'max_keywords': 10,
    'chart_height': 400,
    'chart_width': 800,
    'color_scheme': [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ],
    'font_family': 'Arial, sans-serif',
    'background_color': '#ffffff',
    'grid_color': '#e0e0e0'
}

# UI Components --------------------------------------------------------
class UI:
    """UI configuration settings.
    
    This class defines all UI-related configuration including:
    - Dimensions for responsive layout
    - Theme colors for consistent styling
    - Component configurations for UI elements
    - Style definitions for specific UI elements
    
    The configuration is structured to maintain consistency across
    the application while making it easy to update styles centrally.
    """
    
    DIMENSIONS = {
        'plot_height': '400px',
        'plot_width': '100%',
        'sidebar_width': '400px'
    }
    
    THEME = {
        'primary_color': '#007bff',
        'secondary_color': '#6c757d',
        'success_color': '#28a745',
        'danger_color': '#dc3545',
        'warning_color': '#ffc107',
        'info_color': '#17a2b8'
    }
    
    STYLES = {
        'status_history': {
            'max-height': '200px',
            'overflow-y': 'auto',
            'padding': '10px',
            'background-color': '#f8f9fa',
            'border': '1px solid #dee2e6',
            'border-radius': '4px',
            'margin-top': '10px',
            'font-family': 'monospace',
            'font-size': '0.9em',
            'white-space': 'pre-wrap'
        },
        'sidebar': {
            'background-color': '#f8f9fa',
            'padding': '20px',
            'border-right': '1px solid #dee2e6'
        },
        'main_content': {
            'padding': '20px',
            'flex-grow': '1'
        },
        'visualization': {
            'border': '1px solid #dee2e6',
            'border-radius': '4px',
            'padding': '15px',
            'margin-bottom': '20px'
        },
        'button_group': {
            'display': 'grid',
            'gap': '10px',
            'margin-top': '15px',
            'margin-bottom': '15px'
        }
    }
    
    VISUALIZATION_STYLES = {
        'plot_container': {
            'width': '100%',
            'height': '600px',
            'margin': '20px 0'
        },
        'wordcloud_container': {
            'width': '100%',
            'height': 'auto',
            'text-align': 'center'
        },
        'hierarchy_container': {
            'width': '100%',
            'height': '800px',
            'border': '1px solid #dee2e6'
        }
    }
    
    COMPONENTS = {
        'sections': {
            'title': 'Topic Modeling Analysis',
            'results': {
                'word_cloud': 'Word Cloud',
                'topic_dist': 'Topic Distribution',
                'topic_hierarchy': 'Topic Hierarchy',
                'topic_viz': 'Topic Visualization',
                'topic_summary': 'Topic Summary',
                'generated_files': 'Generated Files'
            }
        },
        'inputs': {
            'file': {
                'label': 'Upload Data File',
                'accept': '.csv'
            },
            'model_params': {
                'embedding_model': {
                    'label': 'Embedding Model',
                    'value': TOPIC_MODELING['EMBEDDING']['model'],
                    'choices': ['all-mpnet-base-v2', 'all-MiniLM-L6-v2']
                },
                'n_neighbors': {
                    'label': 'UMAP Neighbors',
                    'value': TOPIC_MODELING['UMAP']['n_neighbors'],
                    'min': 2,
                    'max': 100,
                    'step': 1
                },
                'min_dist': {
                    'label': 'UMAP Min Distance',
                    'value': TOPIC_MODELING['UMAP']['min_dist'],
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.1
                },
                'nr_topics': {
                    'label': 'Number of Topics',
                    'value': TOPIC_MODELING['TOPIC']['nr_topics'],
                    'min': 2,
                    'step': 1
                },
                'min_topic_size': {
                    'label': 'Minimum Topic Size',
                    'value': TOPIC_MODELING['TOPIC']['min_topic_size'],
                    'min': 2,
                    'max': 20,
                    'step': 1
                },
                'top_n_words': {
                    'label': 'Keywords per Topic',
                    'value': TOPIC_MODELING['TOP_N_WORDS'],
                    'min': 5,
                    'max': 20,
                    'step': 1
                },
                'mmr_diversity': {
                    'label': 'Keyword Diversity',
                    'value': TOPIC_MODELING['CALCULATE_PROBABILITIES'],
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.1
                }
            }
        },
        'buttons': {
            'load_data': {
                'label': 'Load Data',
                'class': 'btn-primary',
                'width': '100%'
            },
            'clean_data': {
                'label': 'Clean Data',
                'class': 'btn-primary',
                'width': '100%'
            },
            'process_attachments': {
                'label': 'Process Attachments',
                'class': 'btn-primary',
                'width': '100%'
            },
            'run_modeling': {
                'label': 'Run Topic Modeling',
                'class': 'btn-primary',
                'width': '100%'
            }
        }
    }

OUTPUT_FILES: Final[Dict[str, str]] = {
    'topic_model': 'topic_model.pkl',
    'embeddings': 'embeddings.npy',
    'clusters': 'clusters.json',
    'viz_2d': 'topic_viz_2d.html',
    'wordcloud': 'wordcloud.png',
    'initial': 'df_initial.csv',
    'attach': 'df_initial_attach.csv',
    'cleaned': 'df_initial_attach_clean.csv',
    'topics': 'df_initial_attach_clean_topics.csv'
}

# File stage order for consistent processing
FILE_STAGES: Final[List[str]] = [
    'initial',
    'attach',
    'cleaned',
    'topics'
]

# Chunking parameters
CHUNK_CONFIG = {
    'MIN_LENGTH': 200,
    'SIMILARITY_THRESHOLD': 0.75,
    'MAX_CHUNK_LENGTH': 2000,
    'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
    'BATCH_SIZE': 64
}

# Remove duplicate individual constants and use the nested dict values
UMAP_N_NEIGHBORS: Final[int] = TOPIC_MODELING['UMAP']['n_neighbors']
UMAP_MIN_DIST: Final[float] = TOPIC_MODELING['UMAP']['min_dist'] 
UMAP_METRIC: Final[str] = TOPIC_MODELING['UMAP']['metric']
EMBEDDING_MODEL: Final[str] = TOPIC_MODELING['EMBEDDING']['model']
BATCH_SIZE: Final[int] = TOPIC_MODELING['EMBEDDING']['batch_size']
RANDOM_SEED: Final[int] = TOPIC_MODELING['EMBEDDING']['random_seed']
DEFAULT_NUM_TOPICS: Final[Union[int, str]] = TOPIC_MODELING['TOPIC']['nr_topics']
MIN_DOCUMENTS: Final[int] = TOPIC_MODELING['TOPIC']['min_topic_size']

# Add after line 288 (after OUTPUT_FILES definition)
TOPIC_OUTPUT_CONFIG: Final[Dict[str, Any]] = {
    'DEFAULT_FILENAME': 'df_topics.csv',
    'REQUIRED_COLUMNS': {'Topic', 'Topic Name'},
    'VISUALIZATION_DIR': 'visualizations',
    'REPORTS_DIR': 'reports',
    'TEMP_DIR': 'temp'
}

VISUALIZATION_CONFIG = {
    'PLOT_DIMENSIONS': {
        'width': 800,
        'height': 600,
        'dpi': 100
    },
    'WORDCLOUD': {
        'width': 1200,
        'height': 800,
        'background_color': 'white',
        'colormap': 'viridis',
        'prefer_horizontal': 0.7,
        'min_font_size': 8,
        'max_font_size': 80
    },
    'COLORS': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'background': '#ffffff',
        'grid': '#e0e0e0'
    }
}

# Add validation for report configuration
REPORT_CONFIG = {
    'templates': {
        'summary': 'summary_template.html',
        'full_report': 'full_report_template.html'
    },
    'output_formats': ['html', 'pdf'],
    'include_visualizations': True,
    'max_topics_per_page': 10
}

def validate_topic_config(config: Dict[str, Any]) -> None:
    """Validate topic modeling configuration."""
    required_keys = {
        'EMBEDDING', 'UMAP', 'TOPIC', 'SEED_TOPICS',
        'MIN_KEYWORDS', 'MAX_KEYWORDS', 'TOP_N_WORDS'
    }
    
    if missing := required_keys - set(config.keys()):
        raise ValueError(f"Missing required config keys: {missing}")
        
    # Validate embedding model
    if 'model' not in config['EMBEDDING']:
        raise ValueError("Missing embedding model configuration")
        
    # Validate UMAP parameters
    umap_params = config['UMAP']
    if not all(k in umap_params for k in ['n_neighbors', 'min_dist', 'metric']):
        raise ValueError("Missing UMAP parameters")
        
    # Validate topic parameters
    topic_params = config['TOPIC']
    if not all(k in topic_params for k in ['nr_topics', 'min_topic_size']):
        raise ValueError("Missing topic parameters")