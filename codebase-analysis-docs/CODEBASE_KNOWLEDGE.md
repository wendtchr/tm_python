# TM_PYTHON: Codebase Knowledge Document

**Version:** 0.1.0  
**Last Updated:** 2026-03-11  
**Type:** Topic Modeling Web Application

---

## 1. HIGH-LEVEL OVERVIEW

### 1.1 Application Purpose

**TM_PYTHON** is a **Shiny web application** for topic modeling and analysis of text data from public comments, documents, or survey responses. It enables epidemiologists, researchers, and data analysts to:

- Upload and process CSV datasets containing text comments
- Extract and clean text content using NLP pipelines
- Generate **guided topic models** using BERTopic with optional seed topics
- Visualize and analyze discovered topics with interactive charts
- Compare model-assigned topics with human-assigned labels (optional)
- Export results as HTML reports, CSV files, and visualizations

**Target Users:** Public health researchers, epidemiologists, policy analysts, environmental health specialists analyzing wildfire health impacts and worker exposure data.

### 1.2 Core Features

| Feature | Purpose | Entry Point |
|---------|---------|-------------|
| **File Upload & Processing** | Ingest CSV data with text comments | `app.py` → UI file input → `process_file_upload()` |
| **Data Cleaning** | Remove noise, normalize text, handle encoding | `data_processing.py` → `clean_data()` |
| **Paragraph Splitting** | Split long comments into semantic chunks | `data_processing.py` → `split_paragraphs()` |
| **Guided Topic Modeling** | BERTopic with optional seed topics | `topic_modeling.py` → `TopicModeler` |
| **Interactive Visualizations** | Hierarchy, distribution, word clouds | `visualization.py` → `VisualizationService` |
| **Seed Topic Management** | Define and modify guided topic keywords | UI dynamic controls → server side aggregation |
| **Status Tracking** | Real-time progress updates during processing | `app_core.py` → `StatusManager` |
| **Session Management** | Handle file storage and cleanup | `app_core.py` → `SessionManager` |

### 1.3 Technology Stack

```
Frontend:         Shiny UI components (Python)
Web Framework:    Shiny 1.2.1 + Starlette 0.45.3
Topic Modeling:   BERTopic 0.16.4 (HDBSCAN clustering, UMAP dimensionality reduction)
Embeddings:       Sentence Transformers (all-MiniLM-L6-v2)
Visualizations:   Plotly (interactive), Matplotlib/WordCloud (publication-quality)
Data Processing:  Pandas 2.2.3, NumPy 2.0.2, NLTK 3.8.1
Text Tools:       BeautifulSoup4, pypdf (PDF extraction), python-docx (DOCX extraction)
Async:            aiohttp, nest-asyncio, asyncio
```

### 1.4 Directory Structure

```
tm_python/
├── launcher.py                 # Application entry point (runs Shiny app on localhost:8000)
├── app_files/
│   ├── app.py                  # Main Shiny application initialization
│   ├── requirements.txt         # Python dependencies
│   ├── www/                     # Static web assets (CSS, JS, images)
│   │   └── style.css
│   └── modules/                 # Core application modules
│       ├── __init__.py          # Module exports
│       ├── config.py            # Configuration and constants
│       ├── core_types.py        # Type definitions and protocols
│       ├── app_core.py          # Core components (SessionManager, StatusManager)
│       ├── ui.py                # UI layout and components
│       ├── server.py            # Server-side request handlers
│       ├── app.py               # (see app_files/app.py)
│       ├── data_processing.py   # Data cleaning and validation
│       ├── topic_modeling.py    # BERTopic wrapper and modeling logic
│       ├── visualization.py     # Visualization generation service
│       ├── utils.py             # Utility functions (file ops, encoding detection)
│       ├── decorators.py        # Error handling and async decorators
│       └── __pycache__/
├── data/                       # Input data directory
│   └── wfs_analyzed_1-24-25.csv # Sample dataset
├── outputs/                    # Generated outputs (created on first run)
│   └── {session_id}/
│       ├── temp/               # Temporary processing files
│       ├── visualizations/     # Generated visualization HTML/PNG
│       └── reports/            # Generated text reports
├── setup.bat                   # Windows setup script (creates venv, installs deps)
├── run.bat                     # Windows runner script (activates venv, runs app)
├── readme.md                   # User-facing documentation
└── .github/                    # GitHub configuration
```

---

## 2. SYSTEM ARCHITECTURE

### 2.1 Architectural Patterns

**Pattern:** Layered Architecture with Reactive Components

```
┌─────────────────────────────────────────────┐
│           Shiny Web UI Layer                │
│  (ui.py: reactive components, I/O bindings) │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│       Server Logic Layer (server.py)        │
│  (event handlers, state management,         │
│   reactive computation, file serving)       │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│      Business Logic Layer                   │
│  ┌─────────────────────────────────────┐    │
│  │ Data Processing (data_processing.py)│    │
│  │ • File validation & encoding        │    │
│  │ • Text cleaning & normalization     │    │
│  │ • Paragraph splitting               │    │
│  └─────────────────────────────────────┘    │
│  ┌─────────────────────────────────────┐    │
│  │ Topic Modeling (topic_modeling.py)  │    │
│  │ • BERTopic initialization           │    │
│  │ • Guided topic processing           │    │
│  │ • Model training & inference        │    │
│  └─────────────────────────────────────┘    │
│  ┌─────────────────────────────────────┐    │
│  │ Visualization (visualization.py)    │    │
│  │ • Plotly chart generation           │    │
│  │ • Word cloud creation               │    │
│  │ • HTML export                       │    │
│  └─────────────────────────────────────┘    │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│      Infrastructure Layer                   │
│  ┌─────────────────────────────────────┐    │
│  │ Session Management (app_core.py)    │    │
│  │ • SessionManager                    │    │
│  │ • PathManager                       │    │
│  │ • StatusManager                     │    │
│  └─────────────────────────────────────┘    │
│  ┌─────────────────────────────────────┐    │
│  │ Utilities                           │    │
│  │ • Type definitions (core_types.py)  │    │
│  │ • Configuration (config.py)         │    │
│  │ • Helper functions (utils.py)       │    │
│  │ • Decorators (decorators.py)        │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

### 2.2 Data Flow Architecture

```
START: User opens app @ localhost:8000
   │
   ├─→ launcher.py:run_app() 
   │       └─→ app_files/app.py:create_app()
   │           ├─→ Create UI layout (ui.py:create_ui())
   │           ├─→ Initialize SessionManager
   │           └─→ Setup server handlers (server.py:create_server())
   │
   ├─→ USER UPLOADS CSV FILE
   │       └─→ server.py:handle_file_upload()
   │           ├─→ data_processing.py:process_file_upload()
   │           │   ├─→ Validate encoding (detect from attempts)
   │           │   ├─→ Read CSV → DataFrame
   │           │   └─→ Validate required columns
   │           ├─→ Save to session temp directory
   │           └─→ Update reactive.Value(data_df)
   │
   ├─→ USER CONFIGURES PARAMETERS & SEED TOPICS
   │       └─→ server.py reactive handlers
   │           ├─→ handle_add_seed_topic() - add UI elements
   │           ├─→ handle_remove_topic() - remove UI elements
   │           └─→ Parse seed topic inputs on form submission
   │
   ├─→ USER CLICKS "RUN MODELING"
   │       └─→ server.py:@reactive.Calc(run_modeling)
   │           ├─→ data_processing.py:clean_data()
   │           │   ├─→ Normalize whitespace
   │           │   └─→ Remove null/empty comments
   │           │
   │           ├─→ data_processing.py:split_paragraphs()
   │           │   ├─→ Use SentenceTransformer embeddings
   │           │   ├─→ Calculate cosine similarity
   │           │   └─→ Group related paragraphs by threshold
   │           │
   │           ├─→ topic_modeling.py:TopicModeler.fit_transform_dataframe()
   │           │   ├─→ Initialize BERTopic with config
   │           │   ├─→ Process seed topics → keyword lists
   │           │   ├─→ Generate embeddings (all-MiniLM-L6-v2)
   │           │   ├─→ UMAP dimensionality reduction
   │           │   ├─→ HDBSCAN clustering
   │           │   ├─→ Fit model with seed guidance
   │           │   └─→ Generate topic names from keywords
   │           │
   │           ├─→ StatusManager updates progress (0-100%)
   │           └─→ Update reactive.Value(model)
   │
   ├─→ VISUALIZATIONS RENDER (triggered by reactive model)
   │       └─→ server.py @render() decorators
   │           ├─→ visualization.py:VisualizationService.get_topic_visualization()
   │           ├─→ visualization.py:VisualizationService.get_topic_hierarchy()
   │           ├─→ visualization.py:VisualizationService.get_topic_wordcloud()
   │           └─→ Update UI with Plotly/Matplotlib figures
   │
   ├─→ USER EXPORTS RESULTS
   │       └─→ server.py:handle_export()
   │           ├─→ Generate HTML report
   │           ├─→ Save visualizations to outputs/{session_id}/
   │           └─→ Serve files via /outputs mount point
   │
   └─→ SESSION CLEANUP (on browser close or timeout)
           └─→ SessionManager.cleanup()
               ├─→ Remove temp files
               └─→ Free embeddings cache
```

### 2.3 Component Interaction Map

| Component | Depends On | Used By | Purpose |
|-----------|-----------|---------|---------|
| `app.py` (entry) | launcher.py | None | Initialize Shiny app, set environment vars |
| `ui.py` | config, core_types | server.py, app.py | Define UI layout and input components |
| `server.py` | All business logic layers | Shiny runtime | Handle reactive events and render outputs |
| `topic_modeling.py` | config, utils, visualization, BERTopic | server.py | Train and manage topic models |
| `data_processing.py` | config, utils, core_types | server.py | Clean and validate input data |
| `visualization.py` | Plotly, Matplotlib, config | server.py, topic_modeling | Render interactive visualizations |
| `app_core.py` | config, core_types | app.py, server.py | Manage sessions, status, paths |
| `config.py` | None (leaf) | All modules | Centralized configuration |
| `core_types.py` | typing, pandas, numpy | All modules | Type definitions and protocols |
| `utils.py` | core_types, aiohttp | All modules | File I/O, encoding, HTTP utilities |
| `decorators.py` | core_types, asyncio | server.py, topic_modeling | Error handling, status tracking |

### 2.4 Reactive State Model

Shiny applications use a reactive programming model. Key reactive values in the server:

```python
# Server-side state in server.py:create_server()
data_df = reactive.Value(None)              # Current DataFrame being analyzed
model = reactive.Value(None)                # Trained BERTopic model
current_output_dir = reactive.Value(None)   # Session output directory
current_status = reactive.Value({})         # Status updates
topic_viz_data = reactive.Value(None)       # Cached visualization data
seed_topic_count = reactive.Value(N)        # Number of seed topics in UI
modeling_state = reactive.Value({           # Modeling execution state
    'in_progress': False,
    'last_run': None
})

# Reactive computations (triggered by input changes)
@reactive.Calc
def processed_data():
    """Automatically updates when data_df changes."""
    if data_df.get() is None:
        return None
    return clean_data(data_df.get())

# Render outputs (triggered by reactive dependency changes)
@render.ui
def topic_visualization():
    """Automatically re-renders when model changes."""
    if model.get() is None:
        return "No model trained"
    return visualization_service.get_topic_visualization(model.get())
```

When a user changes an input → reactive values update → dependent computations run → outputs render.

---

## 3. FEATURE-BY-FEATURE ANALYSIS

### FEATURE 1: File Upload & Data Processing

**Business Purpose:** Enable users to input their text data in CSV format

**User Flow:**

1. User selects CSV file via file upload widget
2. System validates file existence and encoding
3. DataFrame is read and validated for required columns
4. File is saved to session directory for reference
5. DataFrame is passed to next processing stage

**Technical Implementation:**

| Component | Method | Responsibility |
|-----------|--------|-----------------|
| `ui.py` | `create_ui()` → file input widget | Render file upload button |
| `server.py` | `@reactive.Effect` on file input | Detect file upload event |
| `server.py` | `process_file_upload()` | Orchestrate file handling |
| `data_processing.py` | `process_file_upload()` | Read file, validate structure |
| `data_processing.py` | `read_csv_with_encoding()` | Multi-encoding fallback reading |
| `app_core.py` | `SessionManager.add_file()` | Track file in session |

**Key Code Locations:**

- File input UI: [ui.py](app_files/modules/ui.py#L1-L50)
- File handler: [server.py](app_files/modules/server.py#L150-L200)
- CSV reader: [data_processing.py](app_files/modules/data_processing.py#L80-L120)

**Data Transformations:**

```
CSV File (disk)
    ↓
read_csv_with_encoding()
    ↓
Pandas DataFrame (UTF-8 or detected encoding)
    ↓
Validation checks:
  - Not empty
  - Required columns present ('Comment')
  - Data types correct
    ↓
Saved to: outputs/{session_id}/temp/initial_data.csv
```

**Error Handling:**

- File encoding detection tries: utf-8 → cp1252 → iso-8859-1 → latin1
- Missing required columns raises ValueError
- Empty DataFrames are rejected during processing

---

### FEATURE 2: Data Cleaning & Validation

**Business Purpose:** Normalize text and remove outliers/corrupted data

**User Flow:**

1. User confirms upload and clicks data processing step
2. System cleans text (whitespace normalization, null handling)
3. Optional timestamp parsing if 'Posted Date' column present
4. DataFrame passes validation before topic modeling starts

**Technical Implementation:**

| Component | Function | Operation |
|-----------|----------|-----------|
| `data_processing.py` | `DataFrameProcessor.process_dataframe()` | Main cleaning orchestrator |
| `data_processing.py` | N/A (inline logic) | Normalize paragraph breaks in 'Comment' |
| `data_processing.py` | N/A (inline logic) | Parse 'Posted Date' → datetime |

**Cleaning Rules:**

```python
# Text cleaning (in 'Comment' column)
- Remove leading/trailing whitespace: strip()
- Normalize paragraph breaks: \s*\n\s*\n → \n\n
- Preserve content within paragraphs

# Date parsing (in 'Posted Date' column if exists)
- Use pd.to_datetime(format='mixed', errors='coerce')
- Invalid dates converted to NaT
```

**Code Location:** [data_processing.py](app_files/modules/data_processing.py#L60-L90)

---

### FEATURE 3: Paragraph Splitting (Semantic Chunking)

**Business Purpose:** Break long comments into manageable semantic units for topic modeling

**Why Important:** Long documents with multiple topics confound TM algorithms; splitting preserves topic purity

**User Flow:**

1. User enables "Paragraph Splitting" checkbox in UI (or it's automatic)
2. System uses embeddings to group related sentences/paragraphs
3. Similar paragraphs are kept together; dissimilar ones split
4. Each chunk becomes an independent "document" for topic modeling

**Technical Implementation:**

| Component | Method | Responsibility |
|-----------|--------|-----------------|
| `data_processing.py` | `split_paragraphs()` | Main orchestrator |
| `data_processing.py` | `_chunk_text()` | Split text into sentences/chunks |
| `data_processing.py` | `_calculate_similarity()` | Compute cosine similarity |
| `SentenceTransformer` | embedded model | Generate vector embeddings |

**Algorithm:**

```
1. For each comment in DataFrame:
   a. Split by "\n\n" → candidate paragraphs
   b. For lines ≤ min_length, skip to next
   
2. Initialize chunks = [para_0]

3. For each subsequent paragraph:
   a. Embed paragraph → vector
   b. Compare with last chunk in chunks → cosine_similarity
   c. If similarity > SIMILARITY_THRESHOLD (0.75):
       - Append to last chunk (group related content)
   d. Else:
       - Start new chunk
   
4. Store chunks with metadata:
   chunk_text, chunk_id, parent_comment_id, char_length

5. Return expanded DataFrame with chunk_based documents
```

**Configuration Parameters:**

```python
# From config.py CHUNK_CONFIG
SIMILARITY_THRESHOLD = 0.75   # Similarity must exceed to group
MIN_LENGTH = 10               # Skip paragraphs shorter than this
MAX_CHUNK_LENGTH = 1024       # Soft max on chunk size
```

**Code Location:** [data_processing.py](app_files/modules/data_processing.py#L150-L220)

**Practical Example:**

```
Original comment:
"The wildfire smoke affects my health. 
I get respiratory issues. 
It costs me money for medical bills.
New policies help."

↓ (split_paragraphs with threshold=0.75)

Chunk 1: "The wildfire smoke affects my health. I get respiratory issues."
         (high similarity: both about health impacts)

Chunk 2: "It costs me money for medical bills."
         (lower similarity: economic impact)

Chunk 3: "New policies help."
         (dissimilar: policy sentiment)
```

---

### FEATURE 4: Topic Modeling with BERTopic

**Business Purpose:** Discover latent topics in comment text using unsupervised and guided learning

**User Flow:**

1. User enters model parameters (UMAP neighbors, min topic size, etc.)
2. User optionally defines seed topics (guided modeling)
3. User clicks "Run Topic Modeling"
4. System trains BERTopic model on processed data
5. Topics and assignments displayed in UI

**Technical Implementation:**

| Component | Class/Method | Responsibility |
|-----------|--------------|-----------------|
| `topic_modeling.py` | `TopicModeler` class | Main topic modeling interface |
| `topic_modeling.py` | `__init__()` | Initialize components (embeddings, UMAP, HDBSCAN) |
| `topic_modeling.py` | `fit_transform_dataframe()` | Train model on DataFrame |
| `topic_modeling.py` | `_process_seed_topics()` | Parse seed topic strings |
| `visualization.py` | `VisualizationService` | Generate visualizations post-training |

**BERTopic Pipeline:**

```
Input: Documents (list of strings)
   ↓
1. Extract Embeddings
   ├─→ SentenceTransformer (all-MiniLM-L6-v2)
   ├─→ Batch processing (batch_size=64)
   └─→ Output: (n_docs, 384)-dim vectors
   
2. Dimensionality Reduction
   ├─→ UMAP (n_neighbors=15, min_dist=0.1, metric='euclidean')
   └─→ Output: (n_docs, 2)-dim vectors for visualization
   
3. Clustering
   ├─→ HDBSCAN (min_cluster_size=4, metric='euclidean')
   ├─→ Handles outliers as topic -1 (noise)
   └─→ Output: cluster_labels = [0, 1, -1, 0, ...]
   
4. Vectorization
   ├─→ CountVectorizer (1-2 grams)
   ├─→ TF-IDF weighting (ClassTfidfTransformer)
   └─→ Output: term-document matrix
   
5. Representation
   ├─→ KeyBERTInspired (MMR diversity)
   ├─→ Extract top 12 keywords per topic
   └─→ Output: topic_keywords = {0: [('health',0.9), ...], ...}
   
6. (Optional) Seed Topic Guidance
   ├─→ If seed_topics provided:
   │   ├─→ Primary model trained on full data
   │   ├─→ Auxiliary model trained on seed topic keywords
   │   └─→ BERTopic merges both → guided topics
   └─→ Output: Topics aligned with external knowledge
   
Result: BERTopic model + topic info DataFrame
```

**Seed Topic Processing:**

When user provides seed topics (e.g., "health hazards, respiratory illness, asthma"):

```python
# Parsing (in _process_seed_topics)
seed_topics = [
    "health hazards, respiratory illness, asthma, ...",
    "ppe, masks, respirators, ...",
    ...
]
↓
topic_keywords = [
    ["health hazards", "respiratory illness", "asthma", ...],
    ["ppe", "masks", "respirators", ...],
    ...
]
↓
topic_name_map = {
    0: "health hazards",
    1: "ppe",
    ...
}
```

These keywords are passed to BERTopic as `seed_topic_list`, which:

- Trains an auxiliary model on seed keywords alone
- Merges guided topics with unsupervised model
- Results in topics aligned with domain expertise

**Configuration Parameters (from config.py):**

```python
TOPIC_MODELING = {
    'TOPIC': {
        'nr_topics': 9,           # Initially detected, adjusted for seeds
        'min_topic_size': 4       # Min docs per topic
    },
    'UMAP': {
        'n_neighbors': 15,        # Local neighborhood size
        'min_dist': 0.1,          # Minimum distance between points
        'metric': 'euclidean'
    },
    'EMBEDDING': {
        'model': 'all-MiniLM-L6-v2',  # SentenceTransformer model
        'batch_size': 64,
        'random_seed': 42
    },
    'NGRAM_RANGE': (1, 2),            # Unigram + bigram keywords
    'TOP_N_WORDS': 12,                # Keywords per topic
    'CALCULATE_PROBABILITIES': True   # Topic assignment probabilities
}
```

**Code Location:** [topic_modeling.py](app_files/modules/topic_modeling.py#L50-L200)

---

### FEATURE 5: Visualization & Report Generation

**Business Purpose:** Display discovered topics in human-interpretable formats for analysis and publication

**User Flow:**

1. After model training, visualizations automatically render
2. User sees 4 main visualizations (distribution, hierarchy, scores, wordcloud)
3. User can download visualizations as HTML/PNG
4. Generated files appear in "Downloads" panel

**Visualizations Generated:**

| Visualization | Type | Purpose | Code |
|---------------|------|---------|------|
| **Topic Distribution** | Bar chart (Plotly) | Show document count per topic | [visualization.py](app_files/modules/visualization.py#L40-L70) |
| **Topic Hierarchy** | Dendrogram (BERTopic) | Show topic relationships/merges | [visualization.py](app_files/modules/visualization.py#L75-L95) |
| **Word Cloud** | Image (Matplotlib/WordCloud) | Visual representation of keyword frequencies | [visualization.py](app_files/modules/visualization.py#L100-L140) |
| **Word Scores** | Bar chart (Plotly) | Top keywords per topic with weights | server.py render_word_scores |

**Technical Implementation:**

| Component | Method | Output |
|-----------|--------|--------|
| `visualization.py` | `VisualizationService.get_topic_visualization()` | Plotly Figure (bar chart) |
| `visualization.py` | `VisualizationService.get_topic_hierarchy()` | Plotly Figure (dendrogram) |
| `visualization.py` | `VisualizationService.get_topic_wordcloud()` | Matplotlib Figure (PNG) |
| `topic_modeling.py` | `save_topic_modeling_outputs()` | Saves all outputs to disk |

**File Output Structure:**

```
outputs/{session_id}/
├── topic_distribution.html    # Plotly bar chart
├── topic_hierarchy.html        # Plotly dendrogram
├── topic_wordcloud.png         # Matplotlib image
├── topic_info.csv              # DataFrame of topics + keywords
├── document_topics.csv         # Document-to-topic assignments
└── model.pkl                   # Pickled BERTopic model (for reload)
```

**Code Locations:**

- Visualization service: [visualization.py](app_files/modules/visualization.py)
- Report generation: [topic_modeling.py](app_files/modules/topic_modeling.py#L250-L350)

---

### FEATURE 6: Seed Topic Management

**Business Purpose:** Allow users to specify domain-specific topics for guided modeling (optional)

**User Flow:**

1. User checks "Use Seed Topics" checkbox
2. Default seed topics load (9 topics for wildfire/health)
3. User can add more topics with "Add Topic" button
4. User can edit each topic's keywords (comma-separated)
5. User can remove topics with "Remove" button
6. When modeling runs, seed keywords guide BERTopic

**Technical Implementation:**

| Component | Event | Responsibility |
|-----------|-------|-----------------|
| `ui.py` | `create_seed_topic_input()` | Generate single topic input box |
| `server.py` | `handle_add_seed_topic()` | Add new input box dynamically |
| `server.py` | `handle_remove_topic()` | Remove input box, update count |
| `server.py` | Form parsing on submit | Extract all seed topic strings |
| `topic_modeling.py` | `_process_seed_topics()` | Parse strings → keywords |

**Default Seed Topics (from config.py):**

```python
'SEED_TOPICS': {
    'DEFAULT': [
        "health hazards, respiratory illness, asthma, copd, cancer risk, ...",
        "ppe, masks, respirators, n95, protective equipment, ...",
        "exposures, particulate matter, air quality, smoke inhalation, ...",
        "health equity, vulnerable populations, access to care, ...",
        "administrative controls, work schedules, rotation policies, ...",
        "mental health, anxiety, stress, depression, psychological, ...",
        "research needs, data gaps, study requirements, evidence, ...",
        "engineering controls, ventilation, filtration, hepa, ...",
        "wildfire constituents, smoke composition, pm2.5, carbon, ..."
    ]
}
```

**UI Seed Topic Input Format:**

```html
<input type="text" id="seed_topic_1" 
       placeholder="topic name, keyword1, keyword2, ..." 
       value="health hazards, respiratory illness, asthma, ..." />
```

Each input can be edited, and topics are parsed on form submission.

**Code Locations:**

- Seed topic UI: [ui.py](app_files/modules/ui.py#L15-L45)
- Event handlers: [server.py](app_files/modules/server.py#L320-L380)
- Seed processing: [topic_modeling.py](app_files/modules/topic_modeling.py#L150-L190)

---

### FEATURE 7: Status Tracking & Progress Updates

**Business Purpose:** Provide real-time feedback to user during long-running operations

**User Flow:**

1. User clicks "Run Modeling"
2. Status bar appears showing "Processing: 0%"
3. As each stage completes (cleaning, embedding, modeling), progress updates
4. User sees stage name and percentage
5. On completion or error, status changes

**Technical Implementation:**

| Component | Class/Method | Responsibility |
|-----------|--------------|-----------------|
| `app_core.py` | `StatusManager` | Track status state |
| `decorators.py` | `@status_context` decorator | Wrap operations with status |
| `server.py` | `@render.ui` on status_display | Render status in UI |

**StatusManager API:**

```python
class StatusManager:
    def update_status(self, stage: str, progress: float, message: str) -> None:
        """Update status with stage name, progress %, and message."""
        
    def set_error(self, message: str) -> None:
        """Set error status."""
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status as dictionary."""
        
    def get_status_history(self) -> List[Dict]:
        """Get list of all status updates in session."""
```

**Example Status Sequence:**

```
1. update_status("Initializing", 0, "Starting topic modeling...")
2. update_status("Cleaning Data", 10, "Removed 5 null comments")
3. update_status("Embedding", 30, "Generated embeddings for 1000 docs")
4. update_status("Clustering", 60, "Running HDBSCAN...")
5. update_status("Extracting Topics", 80, "Extracting keywords...")
6. update_status("Complete", 100, "Topic modeling finished!")
```

**Code Locations:**

- StatusManager: [app_core.py](app_files/modules/app_core.py#L150-L200)
- Status decorator: [decorators.py](app_files/modules/decorators.py#L50-L80)
- UI binding: [server.py](app_files/modules/server.py#L400-L430)

---

### FEATURE 8: Session Management & File Serving

**Business Purpose:** Isolate user sessions, manage temporary files, and serve outputs via HTTP

**Architecture:**

- Each browser session gets unique session ID: `YYYYMMDD_HHMMSS`
- Session directory: `outputs/{session_id}/`
- Subdirectories: `temp/`, `visualizations/`, `reports/`
- Files automatically served via `/outputs` mount point
- Cleanup on session end or timeout

**Technical Implementation:**

| Component | Class | Responsibility |
|-----------|-------|-----------------|
| `app_core.py` | `SessionManager` | Create/track/cleanup sessions |
| `app_core.py` | `PathManager` | Safe path operations |
| `app.py` | Static asset config | Mount `/outputs` for file serving |

**SessionManager Lifecycle:**

```python
# Initialization (on app startup)
session_manager = SessionManager(base_dir=Path("outputs"))

# On request (server handler)
def server(input: Inputs, output: Outputs, session: Session):
    # Each Shiny session auto-creates unique ID
    session_manager.create_session()  
    # Creates: outputs/20260311_120530/
    
    # During data processing
    file_path = session_manager.session_dir / "temp" / "data.csv"
    file_path.write_text(...)

# On cleanup (browser close or 60s timeout)
session_manager.cleanup()
# Removes temp files, model from memory
```

**File Serving (in app.py):**

```python
# Mount outputs directory at /outputs URL prefix
static_dirs = {
    "/www": str(www_dir),         # CSS, JS @ /www/*
    "/outputs": str(BASE_OUTPUT_DIR)  # Generated files @ /outputs/*
}
app = sh.App(
    ui=create_ui(),
    server=server(session_manager),
    static_assets=static_dirs
)
```

**Code Locations:**

- SessionManager: [app_core.py](app_files/modules/app_core.py#L90-L180)
- PathManager: [app_core.py](app_files/modules/app_core.py#L60-L88)
- File mounting: [app.py](app_files/app.py#L90-L120)

---

## 4. NUANCES, SUBTLETIES & GOTCHAS

### 4.1 Critical Implementation Details

#### Embedding Model Caching

The `SentenceTransformer` model (all-MiniLM-L6-v2, ~50MB) is loaded once at `TopicModeler` initialization. Subsequent calls reuse the cached model. This improves performance but uses memory per session.

**Implication:** Multiple concurrent users may exhaust available RAM. Consider model pooling if scaling to 10+ users.

#### Seed Topic Merging

BERTopic's guided modeling works by:

1. Training primary model on full documents
2. Training auxiliary model on seed keywords only
3. Mapping discovered clusters to seed topics
4. Optionally merging similar topics

If seed topics are poorly chosen or overlapping, the merge step may produce fewer topics than requested.

**Best Practice:** Seed topics should be distinct, non-overlapping, and cover ~60-80% of expected topic space. See default topics in [config.py](app_files/modules/config.py#L65-L80).

#### Paragraph Splitting Similarity Threshold

The `SIMILARITY_THRESHOLD = 0.75` is hard-coded in processing. Values >0.75 group more content together; <0.75 create finer splits.

**Trade-off:**

- High threshold (0.9): Fewer, larger documents → coarser topics
- Low threshold (0.5): Many small documents → finer-grained topics

For wildfire health comments, 0.75 balances these well.

#### Status Message Delays

In `data_processing.py`, status updates include `await asyncio.sleep(config.STATUS_DELAY)` to prevent UI thread starvation. Default is 0.1s.

**If modified:** Too low (<0.01s) causes choppy UI; too high (>1s) causes perceived lag.

#### Stopwords List

A custom stopwords set is defined in `config.py` (common verbs, modals like "think", "need", "would"). These are excluded from keyword extraction.

**Extension Point:** To customize, edit `STOPWORDS` set in [config.py](app_files/modules/config.py#L130-L160).

---

### 4.2 Error Scenarios & Recovery

| Error | Cause | Recovery |
|-------|-------|----------|
| **File encoding mismatch** | Non-UTF-8 CSV (e.g., Excel saved as Latin-1) | `read_csv_with_encoding()` tries 4 encodings; user sees error if all fail |
| **Missing Column 'Comment'** | User uploads wrong CSV format | Error message shown; must reupload correct file |
| **Empty DataFrame after cleaning** | All comments are null or empty | Error raised; user must provide valid data |
| **Seed topics malformed** | User enters topic without comma separator | Parser skips invalid topics; warning logged |
| **Out of memory during embedding** | Large dataset (>50k docs) with large batch size | HDBSCAN may fail; recommend increasing `min_topic_size` or reducing batch |
| **UMAP convergence timeout** | Rare with default params; high-D data issues | UMAP fails silently; model still trains but visualization may be off |

**Error Handling Pattern (decorators.py):**

```python
@handle_errors(error_msg="Data cleaning failed")
def clean_data(df: DataFrameType) -> DataFrameType:
    # If exception raised, decorator logs and re-raises
    # Server catches and displays to user via error message
    ...
```

---

### 4.3 Performance Characteristics

| Operation | Time | Depends On |
|-----------|------|-----------|
| **File upload + validation** | <1s | File size, encoding detection |
| **Data cleaning (1k docs)** | <1s | Comment length, null count |
| **Paragraph splitting** | 10-30s | Doc count, avg length, similarity threshold |
| **Embedding generation** | 30-120s | Doc count, batch size, GPU available |
| **BERTopic clustering** | 20-60s | Doc count, dimensionality, seed topics |
| **Visualization generation** | 5-10s | Topic count, keyword count |
| **Total modeling (1k-5k docs)** | 2-5 min | All above |

**Scaling:** Time roughly O(N) where N = document count.

---

### 4.4 Architecture Decision Rationale

#### Why BERTopic over LDA/NMF?

- **BERTopic:** Uses semantic embeddings (contextual) → better topic coherence
- **LDA/NMF:** Bag-of-words → misses semantic similarity

For public comments with varied vocabulary, BERTopic superior.

#### Why Shiny for UI?

- **Pros:** Python-native, reactive, minimal JavaScript knowledge needed
- **Drawbacks:** Slower than React for 1000+ interactive elements
  
For this use case (10-20 inputs, <10 outputs), Shiny performance adequate.

#### Why HDBSCAN over KMeans?

- **KMeans:** Requires pre-specified K, assumes spherical clusters
- **HDBSCAN:** Auto-detects K, handles variable density, outliers

For topic discovery, HDBSCAN's flexibility essential.

---

## 5. TECHNICAL REFERENCE & GLOSSARY

### 5.1 Key Modules & Classes

#### `app.py` (Entry Point)

**Location:** `app_files/app.py`  
**Responsibility:** Initialize Shiny application, configure environment, set up static file serving

**Key Functions:**

- `create_app(session_manager) → sh.App`: Create Shiny application instance
- `server(session_manager) → ServerFunc`: Create server function for request handling
- `debug_paths()`: Log application directory structure

**Key Variables:**

- `LOG_FILE`: Path to application.log
- `session_manager`: Global SessionManager instance
- `static_dirs`: Mount points for static files (`/www`, `/outputs`)

---

#### `ui.py` (User Interface)

**Location:** `app_files/modules/ui.py`  
**Responsibility:** Define all UI components and layout

**Key Functions:**

- `create_ui() → sh.App`: Main UI layout generator
- `create_seed_topic_input(index, topic_str) → ui.tags.div`: Generate single seed topic input box

**UI Sections:**

1. **Sidebar:** File upload, model parameters, seed topics
2. **Main Content:** Status indicator, processing controls
3. **Results:** Visualizations (distribution, hierarchy, wordcloud), file downloads

---

#### `server.py` (Request Handlers)

**Location:** `app_files/modules/server.py`  
**Responsibility:** Handle user interactions, manage state, render outputs

**Key Classes:**

- `ServerManager`: Manages reactive state variables
- `TopicComparisonHandler`: Handles topic comparison UI (if human labels present)

**Key Functions:**

- `create_server(session_manager) → ServerFunc`: Create server function
- `handle_run_modeling()`: Orchestrate topic modeling pipeline
- `@render.ui` and `@render.text` decorated functions: Render outputs

**Reactive Effects:**

- `handle_file_upload()`: Process uploaded file
- `handle_add_seed_topic()`: Add new topic input
- `handle_remove_topic()`: Remove topic input
- `@reactive.Calc` for computed values: `processed_data()`, `model()`

---

#### `topic_modeling.py` (Modeling Logic)

**Location:** `app_files/modules/topic_modeling.py`  
**Responsibility:** Train BERTopic models, manage topic information

**Key Class:**

- `TopicModeler`: Core topic modeling interface

**Key Methods:**

- `__init__()`: Initialize with config, seed topics, models
- `fit_transform_dataframe(df) → tuples`: Train model and assign topics to docs
- `_process_seed_topics()`: Parse seed topic strings into keyword lists
- `_generate_topic_names()`: Create human-readable names from keywords
- `_get_model_config() → Dict`: Build BERTopic initialization config

**Attributes:**

- `model`: Trained BERTopic instance
- `topic_keywords`: Seed topic keywords per ID
- `topic_name_map`: Mapping of topic IDs to names
- `topic_info`: DataFrame of topics and statistics

---

#### `data_processing.py` (Data Pipeline)

**Location:** `app_files/modules/data_processing.py`  
**Responsibility:** Clean text, validate data, prepare for modeling

**Key Functions:**

- `process_file_upload()`: Read CSV, validate structure
- `clean_data()`: Normalize text, handle nulls
- `split_paragraphs()`: Split long docs using semantic similarity
- `read_csv_with_encoding()`: Multi-encoding CSV reader

**Key Class:**

- `DataFrameProcessor`: Utility methods for DataFrame operations

---

#### `visualization.py` (Visualization Service)

**Location:** `app_files/modules/visualization.py`  
**Responsibility:** Generate interactive and static visualizations

**Key Class:**

- `VisualizationService`: Centralized visualization generation

**Key Methods:**

- `get_topic_visualization()`: Bar chart of topic distribution
- `get_topic_hierarchy()`: Dendrogram of topic relationships
- `get_topic_wordcloud()`: Word cloud image
- `save_visualizations(model)`: Save all outputs to disk async

**Outputs:**

- Plotly Figures (interactive HTML)
- Matplotlib Figures (static PNG)

---

#### `app_core.py` (Infrastructure)

**Location:** `app_files/modules/app_core.py`  
**Responsibility:** Session, path, and status management

**Key Classes:**

- `SessionManager`: Create and manage user sessions
- `PathManager`: Safe path operations
- `StatusManager`: Track and report status

**SessionManager Methods:**

- `create_session()`: Create new session with unique ID
- `add_file()`: Track file in session
- `cleanup()`: Remove temp files and free memory

**PathManager Static Methods:**

- `ensure_directory(path)`: Create directory if not exists
- `is_safe_path(path, base)`: Verify path is within base (security)

**StatusManager Methods:**

- `update_status(stage, progress, message)`: Update status
- `set_error(message)`: Set error status
- `get_status()`: Get current status dict

---

#### `config.py` (Configuration)

**Location:** `app_files/modules/config.py`  
**Responsibility:** Centralize all configuration constants

**Key Variables:**

- `TOPIC_MODELING`: BERTopic and embedding config
- `CHUNK_CONFIG`: Paragraph splitting parameters
- `UI.COMPONENTS`: UI section names and labels
- `STOPWORDS`: Words excluded from keywords
- `REQUIRED_COLUMNS`: Mandatory DataFrame columns ('Comment')
- `OPTIONAL_COLUMNS`: Optional columns (timestamps, IDs, etc.)

---

#### `core_types.py` (Type Definitions)

**Location:** `app_files/modules/core_types.py`  
**Responsibility:** Centralize type definitions to reduce circular imports

**Key Types:**

- `DataFrameType`: Alias for `pd.DataFrame`
- `PathLike`: Union of str and Path
- `SessionState`: Enum of session states
- `StatusProtocol`: Protocol for status managers
- `ModelProtocol`: Protocol for topic modeling models

---

### 5.2 Data Flow Schemas

#### Input CSV Format (Required)

| Column | Type | Required | Example |
|--------|------|----------|---------|
| `Comment` | string | ✓ | "The smoke from the wildfire caused respiratory issues..." |
| `Posted Date` | datetime or string | ✗ | "2024-01-24" or "2024-01-24 14:30:00" |
| `First Name` | string | ✗ | "John" |
| `Last Name` | string | ✗ | "Doe" |
| `Document ID` | string | ✗ | "DOC12345" |
| `Topic-Human` | string | ✗ | "health impacts" (for comparison) |
| *Other* | any | ✗ | User can include any additional columns |

**Minimal Valid CSV:**

```csv
Comment
"The air quality is poor today due to wildfire smoke."
"I experienced chest pain and coughing."
```

#### Topic Info Output (DataFrame)

Generated by `model.get_topic_info()`:

| Column | Type | Example |
|--------|------|---------|
| `Topic` | int | 0, 1, 2, -1 (noise) |
| `Count` | int | 145 |
| `Name` | string | "health hazards" |
| `Top_Words` | string | "health, respiratory, disease, ..." |

#### Document-Topic Assignments

| Document Index | Topic ID | Probability |
|---|---|---|
| 0 | 2 | 0.87 |
| 1 | 1 | 0.65 |
| 2 | -1 | 0.00 (noise) |

---

### 5.3 Configuration Schema (config.py)

**TOPIC_MODELING:**

```python
{
    'UMAP': {
        'n_neighbors': 15,        # KNN neighbors for local density
        'min_dist': 0.1,          # Min distance in 2D projection
        'metric': 'euclidean'
    },
    'EMBEDDING': {
        'model': 'all-MiniLM-L6-v2',  # HuggingFace model ID
        'batch_size': 64,
        'random_seed': 42
    },
    'TOPIC': {
        'nr_topics': 9,
        'min_topic_size': 4
    },
    'TOP_N_WORDS': 12,
    'NGRAM_RANGE': (1, 2),        # Unigram + bigram tokens
    'CALCULATE_PROBABILITIES': True,
    'SEED_TOPICS': {
        'DEFAULT': [...]          # List of seed topic strings
    }
}
```

**CHUNK_CONFIG:**

```python
{
    'SIMILARITY_THRESHOLD': 0.75,  # Min similarity to group
    'MIN_LENGTH': 10,              # Min chars to process
    'MAX_CHUNK_LENGTH': 1024
}
```

**UI.COMPONENTS:**

```python
{
    'sections': {
        'title': 'Topic Modeling Analysis',
        'results': {
            'word_cloud': 'Word Cloud',
            'topic_dist': 'Topic Distribution',
            ...
        }
    },
    'inputs': {
        'file': {...},
        'model_params': {...}
    }
}
```

---

### 5.4 API Reference

#### TopicModeler Class

```python
class TopicModeler:
    """Topic modeling using BERTopic."""
    
    def __init__(
        self,
        config_dict: Optional[Dict[str, Any]] = None,
        seed_topics: Optional[List[str]] = None,
        num_topics: Union[int, str] = "auto",
        status_manager: Optional[StatusProtocol] = None,
        **kwargs
    ) -> None:
        """Initialize modeler with config and seed topics."""
        
    async def fit_transform_dataframe(
        self,
        df: DataFrameType
    ) -> Tuple[BERTopic, DataFrameType, List[int]]:
        """Train model and return model, updated df, topic assignments."""
        
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Get cached embedding model."""
```

**Usage Example:**

```python
modeler = TopicModeler(
    seed_topics=["health, illness, disease", "policy, regulations, controls"],
    num_topics=9
)
model, df_updated, topic_ids = await modeler.fit_transform_dataframe(df)
topics = model.get_topic_info()
```

---

#### VisualizationService Class

```python
class VisualizationService:
    """Visualization generation and saving."""
    
    def __init__(self, output_dir: Path) -> None:
        """Initialize with output directory."""
        
    def get_topic_visualization(self, model: ModelProtocol) -> go.Figure:
        """Generate topic distribution bar chart."""
        
    def get_topic_hierarchy(self, model: ModelProtocol) -> go.Figure:
        """Generate topic hierarchy dendrogram."""
        
    def get_topic_wordcloud(self, model: ModelProtocol) -> plt.Figure:
        """Generate word cloud image."""
        
    async def save_visualizations(
        self,
        model: ModelProtocol,
        status_manager: Optional[StatusProtocol] = None
    ) -> Dict[str, Path]:
        """Generate and save all visualizations."""
```

---

### 5.5 Domain Glossary

| Term | Definition | Context |
|------|-----------|---------|
| **Topic** | Cluster of semantically similar documents, represented by keywords | BERTopic output |
| **Topic ID** | Integer identifier; -1 = noise/outliers | Cluster assignment |
| **Seed Topic** | User-defined topic with keywords for guided modeling | Domain expertise input |
| **Embedding** | Dense vector (384-dim) representing document semantics | Sentence Transformers output |
| **UMAP** | Dimensionality reduction: high-dim → 2D for visualization | Visualization layer |
| **HDBSCAN** | Density-based clustering algorithm | Clustering layer |
| **Vectorizer** | Text → count matrix (TF-IDF) | Pre-clustering |
| **Coherence** | Topic quality metric: how semantically related are keywords? | Quality measure |
| **Document-Topic Assignment** | Mapping of each document to its best-fit topic | Inference output |
| **Paragraph Splitting** | Chunking long documents using semantic similarity | Data preprocessing |
| **Stopwords** | Common words excluded from keywords (the, and, think, ...) | Preprocessing |
| **Session** | User's isolated workspace with unique ID and file storage | Infrastructure |
| **Status Manager** | Component tracking operation progress and errors | UX feedback |

---

## 6. DEPENDENCY GRAPH

```
app.py (entry)
  ├─→ config.py
  ├─→ ui.py
  │   ├─→ config.py
  │   ├─→ core_types.py
  │   └─→ Shiny
  ├─→ server.py
  │   ├─→ core_types.py
  │   ├─→ config.py
  │   ├─→ app_core.py
  │   ├─→ data_processing.py
  │   ├─→ topic_modeling.py
  │   ├─→ visualization.py
  │   └─→ Shiny
  └─→ app_core.py
      ├─→ core_types.py
      ├─→ config.py
      ├─→ pathlib, tempfile, shutil
      └─→ logging

topic_modeling.py
  ├─→ config.py
  ├─→ core_types.py
  ├─→ utils.py
  ├─→ visualization.py
  ├─→ BERTopic, SentenceTransformers, UMAP, NLTK
  ├─→ Plotly, Matplotlib, WordCloud
  └─→ Pandas, NumPy

data_processing.py
  ├─→ config.py
  ├─→ core_types.py
  ├─→ utils.py
  ├─→ SentenceTransformers (for similarity)
  ├─→ Pandas, NumPy, aiohttp
  └─→ asyncio, re, pathlib

visualization.py
  ├─→ config.py
  ├─→ core_types.py
  ├─→ Plotly, Matplotlib, WordCloud
  └─→ Pandas, NumPy

utils.py
  ├─→ core_types.py
  ├─→ aiohttp, asyncio
  ├─→ Pandas, NumPy
  └─→ Standard library (re, pathlib, io, etc.)

decorators.py
  ├─→ core_types.py
  ├─→ asyncio, functools
  └─→ logging

core_types.py
  ├─→ Typing, Protocols
  ├─→ Pandas, NumPy
  ├─→ aiohttp, datetime
  └─→ No local imports (leaf module)

config.py
  └─→ Standard library only (pathlib, logging, typing)
```

**Circular Import Strategy:**

- `core_types.py` has NO local imports (safe to import everywhere)
- `config.py` has only standard library imports (safe second)
- Other modules imported in dependency order to prevent cycles

---

## 7. THINGS YOU MUST KNOW BEFORE CHANGING CODE

### 7.1 Don't Break These

1. **`core_types.py` Isolation**: Keep it dependency-free to prevent circular imports
2. **`config.py` Precedence**: All configuration reads from config.py; hardco
ded values create inconsistencies
3. **SessionManager Lifecycle**: Always call `cleanup()` on session end to prevent file handle leaks
4. **Status Manager Updates**: Must include both progress (0-100) and message for UI to update properly
5. **WATCHFILES Suppression**: Environment variables in app.py prevent reload chaos; don't remove

### 7.2 Performance Seams

1. **Embedding Cache**: SentenceTransformer loads on first use; ~30s delay. Cache is per-session.
2. **Paragraph Splitting**: O(N²) similarity comparisons; disable for >50k docs or increase threshold
3. **BERTopic Clustering**: HDBSCAN may timeout on very high-dimensional sparse data; increase `min_topic_size`
4. **Visualization Generation**: Save async to prevent UI thread blocking (VisualizationService.save_visualizations)

### 7.3 Common Pitfalls

1. **Forgetting `.get()`**: Reactive values must call `.get()` to retrieve current value in Shiny
2. **Topic ID -1**: Reserved for noise/outliers; exclude from keyword extraction
3. **Seed Topic Merging**: BERTopic auto-merges similar seed topics; may result in <num_topics actual topics
4. **File Path Encoding**: Use `Path.resolve()` before serialization to prevent relative path issues
5. **Status Delays**: Async operations need `await asyncio.sleep(config.STATUS_DELAY)` between updates

### 7.4 Extension Points

1. **Add Visualization**: Create method in `VisualizationService`, wire in server.py `@render` decorator
2. **Add Model Parameter**: Add to `TOPIC_MODELING` in config.py, add UI input in ui.py, pass through topic_modeling.py
3. **Custom Stopwords**: Edit `STOPWORDS` set in config.py
4. **Parallel Processing**: Wrap file uploads in `asyncio.gather()` in server.py
5. **Custom Embeddings Model**: Change `EMBEDDING.model` in config.py (test compatibility with BERTopic)

---

## 8. QUICK REFERENCE: FILE LOCATIONS

| Task | File(s) |
|------|---------|
| Add new visualization | `visualization.py`, `server.py` |
| Change model parameters | `config.py` (TOPIC_MODELING), `ui.py`, `topic_modeling.py` |
| Fix data validation | `data_processing.py` |
| Modify UI layout | `ui.py` |
| Add status tracking | `decorators.py`, `app_core.py` |
| Handle new input column | `data_processing.py`, `server.py` |
| Customize seed topics | `config.py` (SEED_TOPICS.DEFAULT) |
| Fix file serving | `app.py` (static_dirs), `app_core.py` (PathManager) |
| Add environment variable | `app.py` (os.environ at top) |
| Debug path issues | `app_core.py:debug_paths()`, `config.py:debug_directory_structure()` |

---

## 9. STATE TRACKING FOR CONTINUATION

### Current Knowledge State

- ✅ Phase 1: Initial Context (complete)
- ✅ Phase 2: Architecture Deep Dive (complete)
- ✅ Phase 3: Feature Analysis (complete)
- ✅ Phase 4: Gotchas & Best Practices (complete)
- ✅ Phase 5: Technical Reference (complete)
- ✅ Phase 6: Master Document (complete)

### File Index Summary

| Priority | Path | Type | Lines | Purpose |
|----------|------|------|-------|---------|
| 1 | app.py | Python | 150 | Entry point, app initialization |
| 1 | modules/server.py | Python | 500+ | Request handlers, state management |
| 1 | modules/topic_modeling.py | Python | 350+ | BERTopic wrapper, model training |
| 1 | modules/ui.py | Python | 300+ | UI layout, components |
| 2 | modules/data_processing.py | Python | 250+ | Data pipeline, cleaning |
| 2 | modules/visualization.py | Python | 200+ | Visualization service |
| 2 | modules/app_core.py | Python | 250+ | Session, path, status mgmt |
| 3 | modules/config.py | Python | 200+ | Configuration constants |
| 3 | modules/core_types.py | Python | 150+ | Type definitions |
| 3 | modules/utils.py | Python | 250+ | Generic utilities |
| 4 | modules/decorators.py | Python | 100+ | Error handling, async decorators |
| 4 | requirements.txt | Text | 20 | Dependencies |
| 4 | readme.md | Markdown | 50 | User documentation |

### Architectural Decisions Documented

- ✅ Shiny for reactive UI (vs. React/Streamlit)
- ✅ BERTopic for semantic modeling (vs. LDA/NMF)
- ✅ HDBSCAN for clustering (vs. KMeans)
- ✅ Guided modeling with seed topics (vs. unsupervised only)
- ✅ Paragraph splitting with similarity (vs. fixed-length chunking)

### Open Questions / Future Work

1. **Scalability**: How to handle 1000+ concurrent users? (Currently single-threaded per session)
2. **Real-time Collaboration**: Support multiple users editing same dataset?
3. **Model Comparison**: A/B test different seed topics or hyperparameters?
4. **Export Formats**: Add JSON, Parquet, SQLite export options?
5. **Topic Comparison**: Implement inter-rater reliability (currently placeholder)

---

## Document Metadata

**Document Version:** 1.0  
**Creation Date:** 2026-03-11  
**Scope:** TM_PYTHON application codebase (app_files/ and entry points)  
**Audience:** Developers, maintainers, future contributors  
**Completeness:** All 6 phases of codebase analysis completed  
**Validation:** References to 10+ core files verified  

**Next Steps for External Developers:**

1. Read this document in order
2. Clone repo and run `setup.bat` → `run.bat`
3. Examine files referenced in Section 8 "Quick Reference"
4. Trace data flow end-to-end (Section 2.2)
5. Try modifying a feature (Section 4.4 "Extension Points")

---

*End of Document*
