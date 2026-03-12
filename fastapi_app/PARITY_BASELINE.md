# Parity Baseline Matrix (Phase 0)

## Source of Truth

1. app_files/modules/server.py
2. app_files/modules/data_processing.py
3. app_files/modules/topic_modeling.py
4. app_files/modules/visualization.py
5. app_files/modules/config.py
6. app_files/modules/app_core.py
7. app_files/modules/ui.py

## Defaults and Validation Bounds (Locked)

1. min_topic_size: default 4, minimum 2
2. ngram_min: default 1, range 1 to 3
3. ngram_max: default 2, range 1 to 3, must be >= ngram_min
4. top_n_words: default 12, UI range up to 30
5. umap_n_neighbors: default 15, minimum 5
6. umap_n_components: default 5, minimum 2
7. umap_min_dist: default 0.1, range 0.0 to 1.0
8. chunking enable: default false
9. similarity_threshold: default 0.75, range 0.5 to 0.9
10. min_chunk_length: default 200, range 20 to 200
11. max_chunk_length: default 2000, range 500 to 5000

## Stage Model (Parity Mode)

1. INIT
2. LOADED
3. ATTACHMENTS_PROCESSED
4. CLEANED
5. MODELING_RUNNING
6. MODELED
7. ERROR
8. DELETED

## Artifact and Directory Baseline

Deterministic session directories:

1. outputs/{session_id}/temp
2. outputs/{session_id}/visualizations
3. outputs/{session_id}/reports

Required artifact names to preserve in migration:

1. df_initial.csv
2. df_initial_attach.csv
3. df_initial_attach_clean.csv
4. df_topics.csv
5. visualizations/*
6. topic_comparison/* when Topic-Human exists

## Accepted Parity Notes

1. Modeling remains allowed when data exists (not strictly CLEANED-gated) during parity phase.
2. Hardening and stricter workflow gates are deferred until after parity signoff.
