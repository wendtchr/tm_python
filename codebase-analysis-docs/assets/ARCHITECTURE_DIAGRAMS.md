# TM_PYTHON Architecture Diagrams

## Component Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                    User Browser (localhost:8000)             │
│            Shiny Web UI rendered as interactive HTML          │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/WebSocket
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Shiny Server (Python asyncio)                   │
│           ┌──────────────────────────────────┐               │
│           │  app.py: Application Setup       │               │
│           │  - Set environment variables     │               │
│           │  - Create SessionManager         │               │
│           │  - Mount static files            │               │
│           └──────────────────────────────────┘               │
│                         │                                    │
│           ┌─────────────┴──────────────┐                      │
│           ▼                            ▼                      │
│    ┌──────────────────┐      ┌──────────────────┐            │
│    │  ui.py           │      │  server.py       │            │
│    │ (UI Layout)      │      │(Request Handlers)│            │
│    ├──────────────────┤      ├──────────────────┤            │
│    │ • Input widgets  │      │ • File handlers  │            │
│    │ • Text inputs    │      │ • Reactive @    │            │
│    │ • Output targets │      │ • Render @      │            │
│    │ • CSS refs       │      │ • Status updates │            │
│    └──────────────────┘      └──────────────────┘            │
│           ▲                            │                     │
│           └────────────────┬───────────┘                      │
│                            │                                 │
│                   Reactive Propagation                       │
│                   (inputs → effects → outputs)               │
│                            │                                 │
│           ┌────────────────┴──────────────────┐               │
│           ▼                                   ▼              │
│   ┌──────────────────────────┐    ┌──────────────────────┐   │
│   │  Business Logic Modules  │    │ Infrastructure Layer │   │
│   ├──────────────────────────┤    ├──────────────────────┤   │
│   │ • data_processing.py     │    │ • app_core.py        │   │
│   │   - Read/validate CSV    │    │   - SessionManager   │   │
│   │   - Clean text           │    │   - PathManager      │   │
│   │   - Split paragraphs     │    │   - StatusManager    │   │
│   │                          │    │                      │   │
│   │ • topic_modeling.py      │    │ • config.py          │   │
│   │   - BERTopic wrapper     │    │   - All constants    │   │
│   │   - Seed topic handling  │    │                      │   │
│   │   - Topic generation     │    │ • core_types.py      │   │
│   │                          │    │   - Type defs        │   │
│   │ • visualization.py       │    │   - Protocols        │   │
│   │   - Plotly charts        │    │                      │   │
│   │   - Word clouds          │    │ • utils.py           │   │
│   │   - Export to HTML/PNG   │    │   - File I/O         │   │
│   │                          │    │   - Encoding detect  │   │
│   │ • decorators.py          │    │                      │   │
│   │   - Error handling @     │    │                      │   │
│   │   - Status context @     │    │                      │   │
│   └──────────────────────────┘    └──────────────────────┘   │
│           │                                                  │
│           └─── All depend on config.py & core_types.py ──────│
│                                                             │
│              External Dependencies (pip packages)           │
│              ┌─────────────────────────────────────┐         │
│              │ BERTopic, SentenceTransformers      │         │
│              │ Plotly, Matplotlib, WordCloud       │         │
│              │ Pandas, NumPy, NLTK, UMAP, HDBSCAN │         │
│              │ aiohttp, asyncio, Starlette        │         │
│              └─────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
           │
           │ File I/O
           ▼
┌─────────────────────────────────────────────────────────────┐
│              File System (Disk Storage)                      │
│  /outputs/{session_id}/                                      │
│    ├── temp/          (temporary processing)                │
│    ├── visualizations/ (HTML/PNG outputs)                   │
│    └── reports/       (text reports)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Processing Pipeline

```
User Upload (CSV)
    │
    ├─→ [validation] file encoding detection
    │
    ├─→ [read] pd.read_csv() with fallback encodings
    │
    ├─→ [validate] check required columns present
    │   ├─→ REQUIRED: 'Comment'
    │   └─→ OPTIONAL: 'Posted Date', 'Topic-Human', etc.
    │
    ├─→ [clean_data]
    │   ├─→ Normalize paragraph breaks in Comment
    │   ├─→ Parse Posted Date → datetime
    │   └─→ Remove null/empty rows
    │
    ├─→ [split_paragraphs] semantic chunking
    │   ├─→ Embed paragraphs → vectors (SentenceTransformer)
    │   ├─→ Calculate similarity (cosine)
    │   ├─→ Group similar (similarity > threshold)
    │   └─→ Result: expanded DataFrame with chunks
    │
    ├─→ [BERTopic pipeline]
    │   ├─→ embed: SentenceTransformer
    │   │   └─→ all-MiniLM-L6-v2 model → (N, 384) vectors
    │   │
    │   ├─→ reduce: UMAP dimensionality
    │   │   ├─→ n_neighbors=15 (local structure)
    │   │   ├─→ n_components=2 (for viz)
    │   │   └─→ min_dist=0.1 (spread)
    │   │   └─→ Result: (N, 2) vectors
    │   │
    │   ├─→ cluster: HDBSCAN
    │   │   ├─→ min_cluster_size=4 (minimum docs per topic)
    │   │   ├─→ Result: topic_ids = [0, 1, -1, 0, 2, ...]
    │   │   └─→ -1 = noise/outlier (not in any topic)
    │   │
    │   ├─→ vectorize: CountVectorizer + ClassTfidfTransformer
    │   │   ├─→ (1,2)-grams (NGRAM_RANGE)
    │   │   └─→ TF-IDF weighting per class (topic)
    │   │
    │   └─→ represent: KeyBERTInspired + MMR
    │       ├─→ Extract top keywords per topic
    │       ├─→ Apply MMR for diversity
    │       └─→ Result: keywords={0: [('health',0.9), ...], ...}
    │
    ├─→ [seed_topics] (if provided)
    │   ├─→ Parse seed strings → keyword lists
    │   ├─→ Create auxiliary model on seed keywords
    │   ├─→ Merge topic assignments
    │   └─→ Result: topics aligned with domain knowledge
    │
    ├─→ [visualize]
    │   ├─→ Topic distribution (bar chart)
    │   ├─→ Topic hierarchy (dendrogram)
    │   ├─→ Word cloud (image)
    │   └─→ Keyword scores (bar chart)
    │
    └─→ [export]
        ├─→ Save model.pkl (for reload)
        ├─→ Save topic_info.csv
        ├─→ Save document_topics.csv
        ├─→ Export visualizations (HTML/PNG)
        └─→ Serve via /outputs mount point
```

---

## Reactive Flow in Shiny

```
┌─────────────────────────────────────────┐
│         User Interaction (UI)           │
│  • File upload click                     │
│  • Parameter slider change               │
│  • "Run Modeling" button click           │
│  • Seed topic text edit                  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Reactive Values     │
        │  & Effects Update    │
        │  input$file_upload   │
        │  input$min_topic_size│
        │  input$seed_topic_1  │
        └──────────────────────┘
                   │
                   ▼
   ┌───────────────────────────────┐
   │  @reactive.Effect handlers    │
   │  • handle_file_upload()       │
   │  • handle_run_modeling()      │
   │  • handle_add_seed_topic()    │
   └───────────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Reactive Values Update      │
    │  data_df.set(new_df)         │
    │  model.set(new_model)        │
    │  status.set(new_status)      │
    └──────────────────────────────┘
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
   ┌─────────────┐   ┌─────────────┐
   │ @render.ui  │   │ @render.text│
   │ decorators  │   │ decorators  │
   │             │   │             │
   │ Triggered   │   │ Triggered   │
   │ when:       │   │ when:       │
   │ • data_df   │   │ • status    │
   │   changes   │   │   changes   │
   │ • model     │   │ • model     │
   │   changes   │   │   changes   │
   └────────┬────┘   └────────┬────┘
            │                 │
            │   ┌─────────────┘
            │   │
            │   ▼
            │ ┌──────────────────┐
            │ │ Re-render Output │
            │ │ • Update charts  │
            │ │ • Update status  │
            │ │ • Show results   │
            │ └──────────────────┘
            │
            └─→ Send HTML to Browser
                (via WebSocket/HTTP)
                │
                ▼
        ┌──────────────────┐
        │ Browser Updates  │
        │ visible UI       │
        └──────────────────┘
```

---

## Session Lifecycle

```
START: User opens http://localhost:8000
│
├─→ app.py:create_app()
│   ├─→ SessionManager(base_dir="outputs")
│   ├─→ Register UI (ui.py:create_ui)
│   ├─→ Register server handlers (server.py:create_server)
│   └─→ Mount static dirs (/www, /outputs)
│
├─→ browser connects → Shiny assigns session ID
│
├─→ server() function called with:
│   ├─→ input: reactive inputs
│   ├─→ output: reactive outputs
│   └─→ session: Shiny session object
│
├─→ SessionManager.create_session()
│   ├─→ Generate session_id = "YYYYMMDD_HHMMSS"
│   ├─→ Create directories:
│   │   ├─→ outputs/{session_id}/
│   │   ├─→ outputs/{session_id}/temp/
│   │   ├─→ outputs/{session_id}/visualizations/
│   │   └─→ outputs/{session_id}/reports/
│   └─→ state = SessionState.ACTIVE
│
├─→ User uploads file
│   └─→ file data → process_file_upload()
│       ├─→ validate encoding
│       ├─→ read CSV
│       ├─→ save to session temp/
│       ├─→ SessionManager.add_file(path)
│       └─→ data_df.set(new_df)  [reactive value]
│
├─→ User configures and clicks "Run Modeling"
│   └─→ @reactive.Effect handler
│       ├─→ clean_data(df)
│       ├─→ split_paragraphs(df)
│       ├─→ TopicModeler.fit_transform_dataframe(df)
│       ├─→ VisualizationService.save_visualizations()
│       ├─→ Save outputs to session dir
│       ├─→ model.set(trained_model)  [reactive value]
│       └─→ status.set("Complete")    [reactive value]
│
├─→ Visualizations render
│   └─→ @render.ui/@render.text triggered
│       ├─→ get_topic_visualization()
│       ├─→ get_topic_hierarchy()
│       ├─→ get_topic_wordcloud()
│       └─→ Update UI with Plotly/Matplotlib
│
├─→ User downloads results
│   └─→ Browser fetches from /outputs/{session_id}/
│
├─→ Idle timeout OR browser close
│   └─→ SessionManager.cleanup()
│       ├─→ Delete temp files
│       ├─→ Free model memory
│       ├─→ Close file handles
│       ├─→ state = SessionState.COMPLETED
│       └─→ Remove session data
│
END
```

---

## Module Import Order (Circular Dependency Resolution)

```
1. LEAF MODULES (no app imports)
   ├─→ core_types.py       [only typing, pandas, numpy]
   └─→ config.py           [only standard lib + core_types]

2. UTILITY MODULES (depend on leaf)
   ├─→ utils.py            [core_types, standard lib]
   └─→ decorators.py       [core_types, asyncio]

3. APP CORE (infrastructure, depends on leaf)
   └─→ app_core.py         [core_types, config, standard lib]

4. BUSINESS LOGIC (depends on 1-3)
   ├─→ data_processing.py  [core_types, config, utils, decorators]
   ├─→ visualization.py    [core_types, config]
   └─→ topic_modeling.py   [core_types, config, utils, 
                              visualization, decorators]

5. UI LAYER (depends on 1-4)
   ├─→ ui.py               [core_types, config]
   └─→ server.py           [all above]

6. APP ENTRY (depends on all)
   └─→ app.py              [ui, server, app_core, config]

Launch (from launcher.py):
   └─→ run_app(app, ...)   [calls app.py:app instance]
```

**Key Principle:**  

- `core_types.py` and `config.py` have NO local imports
- This ensures they can be safely imported by any module
- Other modules import in strict dependency order
- No circular imports possible

---

## Configuration Hierarchy

```
System Defaults (config.py)
    │
    ├─→ TOPIC_MODELING config
    │   ├─→ UMAP: n_neighbors, min_dist, metric
    │   ├─→ EMBEDDING: model, batch_size, seed
    │   ├─→ TOPIC: nr_topics, min_topic_size
    │   ├─→ SEED_TOPICS: DEFAULT list (9 topics)
    │   └─→ Other: TOP_N_WORDS, NGRAM_RANGE, etc.
    │
    ├─→ DATA_PROCESSING config
    │   ├─→ REQUIRED_COLUMNS = {'Comment'}
    │   ├─→ OPTIONAL_COLUMNS = {date, names, IDs, ...}
    │   └─→ STATUS_DELAY = 0.1
    │
    ├─→ UI config
    │   ├─→ DIMENSIONS: plot_height, sidebar_width
    │   ├─→ THEME: colors, fonts
    │   ├─→ STYLES: CSS for UI elements
    │   └─→ COMPONENTS: section names, labels
    │
    ├─→ PATH config
    │   ├─→ APP_FILES_DIR = project/app_files/
    │   ├─→ BASE_OUTPUT_DIR = project/outputs/
    │   ├─→ TEMP_DIR, CACHE_DIR, DATA_DIR
    │   └─→ All created on startup if missing
    │
    └─→ LIMITS config
        ├─→ MAX_FILES_PER_SESSION = 100
        ├─→ MAX_SESSION_SIZE_MB = 500
        ├─→ MAX_ATTACHMENT_SIZE = 10MB
        └─→ FILE_SIZE_WARN_MB = 50
```

**Runtime Overrides:**

- `TopicModeler(config_dict=...)` can override TOPIC_MODELING defaults
- No command-line args or env var overrides currently
- Extension point: add env var support in app.py:create_app()

---

*Diagrams generated for codebase-analysis-docs/assets/*
