# Shiny to FastAPI Migration Plan (Parity First, AI Executable)

## 1. Goal

Migrate the current Python Shiny app in app_files to FastAPI while preserving behavior parity for:

1. Data upload and validation
2. Attachment processing
3. Data cleaning
4. Topic modeling
5. Visualization and report outputs
6. Session-scoped files and downloads
7. Status and progress reporting

This plan is designed to be executed by AI agents with limited context and must avoid unnecessary code.

## 2. Scope and Non-Goals (v1)

### In Scope

1. FastAPI backend with explicit HTTP contracts
2. Session-scoped orchestration and task polling
3. Minimal parity UI flow
4. Parity test coverage for behavior and artifacts

### Non-Goals

1. No algorithm redesign of NLP, BERTopic, chunking, or visualization logic
2. No distributed queue infrastructure
3. No WebSocket or SSE push updates
4. No full React redesign before parity acceptance
5. No broad service refactor beyond what is required for parity

## 3. Source of Truth

Implementation and behavior must align with:

1. app_files/modules/server.py
2. app_files/modules/data_processing.py
3. app_files/modules/topic_modeling.py
4. app_files/modules/visualization.py
5. app_files/modules/config.py
6. app_files/modules/app_core.py
7. app_files/modules/ui.py

## 4. Parity Principles

1. Wrap existing logic before rewriting it.
2. Keep routers thin and place orchestration in small services.
3. Add code only when required for stateless API operation.
4. Preserve current output filenames and directory semantics.
5. Preserve current workflow permissiveness in parity mode.
6. Defer hardening changes until after parity signoff.

## 5. Current Behavior Baseline to Preserve

### Workflow Baseline

1. Data is uploaded and validated.
2. Attachments can be processed if present.
3. Data can be cleaned.
4. Modeling can run when data exists.
5. Outputs are saved into session output directories.
6. Visualizations and comparison outputs are generated from model results.

### Key Defaults and Validation Constraints

Use these as API defaults and request validation bounds:

1. min_topic_size default 4, minimum 2
2. ngram_min default 1, range 1 to 3
3. ngram_max default 2, range 1 to 3, must be >= ngram_min
4. top_n_words default 12, UI range up to 30
5. umap_n_neighbors default 15, minimum 5
6. umap_n_components default 5, minimum 2
7. umap_min_dist default 0.1, range 0.0 to 1.0
8. chunking enable default false
9. similarity_threshold default 0.75, range 0.5 to 0.9
10. min_chunk_length default 200, range 20 to 200
11. max_chunk_length default 2000, range 500 to 5000

### Artifact and Directory Baseline

Keep deterministic session paths:

1. outputs/{session_id}/temp
2. outputs/{session_id}/visualizations
3. outputs/{session_id}/reports

Preserve current artifact naming unless a mandatory safety change is required:

1. df_initial.csv
2. df_initial_attach.csv
3. df_initial_attach_clean.csv
4. df_topics.csv
5. visualization outputs under visualizations
6. comparison outputs under topic_comparison when Topic-Human exists

## 6. Target Minimal Architecture

### Required Packages

1. fastapi_app/main.py
2. fastapi_app/api/routers
3. fastapi_app/core
4. fastapi_app/schemas
5. fastapi_app/services

### Router Set

1. health
2. sessions
3. data
4. model
5. tasks
6. visualizations
7. downloads

### Core Services (Minimal)

1. Session service
2. Orchestration service
3. Task registry service
4. Artifact registry service
5. Adapters to existing processing/modeling modules

Do not split into additional services unless blocked by duplication or coupling.

## 7. Explicit State Model (Parity Mode)

Use explicit stage metadata in session.json.

### Session Stages

1. INIT
2. LOADED
3. ATTACHMENTS_PROCESSED
4. CLEANED
5. MODELING_RUNNING
6. MODELED
7. ERROR
8. DELETED

### Transition Rules

1. INIT to LOADED on successful upload
2. LOADED to ATTACHMENTS_PROCESSED on successful attachment processing
3. LOADED or ATTACHMENTS_PROCESSED to CLEANED on successful cleaning
4. LOADED or ATTACHMENTS_PROCESSED or CLEANED to MODELING_RUNNING on model run request in parity mode
5. MODELING_RUNNING to MODELED on success
6. Any stage to ERROR on failure
7. Any active stage to DELETED on delete

Important parity note:
Do not enforce CLEANED-only modeling during parity phase because current app allows modeling when data exists. Strict gating can be added after parity as a hardening option.

## 8. API Contracts (v1)

### Session and Health

1. POST /api/sessions
2. DELETE /api/sessions/{session_id}
3. GET /api/health

### Data

1. POST /api/sessions/{session_id}/upload
2. POST /api/sessions/{session_id}/attachments/process
3. POST /api/sessions/{session_id}/clean
4. GET /api/sessions/{session_id}/files

### Modeling and Tasks

1. POST /api/sessions/{session_id}/model/run
2. GET /api/sessions/{session_id}/tasks/{task_id}
3. GET /api/sessions/{session_id}/model/summary
4. GET /api/sessions/{session_id}/model/results

### Visualizations and Downloads

1. GET /api/sessions/{session_id}/visualizations/topic
2. GET /api/sessions/{session_id}/visualizations/hierarchy
3. GET /api/sessions/{session_id}/visualizations/wordcloud
4. GET /api/sessions/{session_id}/visualizations/word-scores
5. GET /api/sessions/{session_id}/visualizations/alignment
6. GET /api/sessions/{session_id}/downloads/{artifact_name}

### Standard Response Expectations

1. Structured success payloads with session_id and stage where relevant
2. Structured error payloads with code, message, details, session_id, stage
3. Stable task payload with task_id, status, progress, message, created_at, updated_at, optional result, optional error

## 9. Async and Task Execution

1. Route handlers are async.
2. CPU-heavy modeling and embedding work runs off request thread.
3. Model run endpoint must return immediately with task_id.
4. Task status is polling based.
5. Duplicate model submission for same session while running returns conflict.
6. Restart behavior in v1 can mark in-flight tasks as failed and require rerun.

## 10. Phased Delivery Plan

### Phase 0: Parity Lock and Baseline Matrix

Deliverables:

1. Parity checklist covering validations, stage behavior, outputs, and errors
2. Frozen defaults and bounds from config and UI
3. Frozen artifact manifest and expected locations

Acceptance:

1. Baseline matrix approved before implementation starts

### Phase 1: Minimal FastAPI Foundation

Tasks:

1. Create app entry and router wiring
2. Implement settings and dependency wiring
3. Implement session create/delete and session.json persistence

Acceptance:

1. Health endpoint works
2. Session directory creation is deterministic and valid
3. Delete is idempotent

### Phase 2: Data Workflow Endpoints

Tasks:

1. Implement upload endpoint using existing processing code
2. Implement attachments endpoint using existing processing code
3. Implement clean endpoint using existing processing code
4. Implement files endpoint for artifact listing

Acceptance:

1. Required-column validation matches current behavior
2. File outputs match expected names and locations
3. Stage transitions persist correctly in session.json

### Phase 3: Modeling and Polling

Tasks:

1. Add task registry
2. Add model run enqueue endpoint
3. Run existing topic modeling logic in background execution
4. Add task status endpoint

Acceptance:

1. API remains responsive during modeling
2. Duplicate run guard works
3. Successful run produces expected modeling and visualization artifacts

### Phase 4: Visualization and Downloads

Tasks:

1. Add read-only visualization endpoints
2. Add key-based artifact download registry
3. Add comparison artifact exposure when available

Acceptance:

1. Known artifact keys resolve correctly
2. Unknown keys return validation errors
3. Missing files return not-found without unsafe path access

### Phase 5: Thin Parity UI

Tasks:

1. Implement minimal workflow UI for upload, process, clean, run, poll, and view outputs
2. Preserve seed-topic editing and defaults

Acceptance:

1. End-to-end flow is functionally equivalent for core workflow

### Phase 6: Verification and Cutover

Tasks:

1. Contract tests
2. Workflow and stage tests
3. Parity fixture tests
4. Concurrency and cleanup checks

Acceptance:

1. Parity checklist is fully green
2. No critical regressions against baseline behavior

## 11. AI-Agent Execution Rules

These rules are mandatory for limited-context execution.

### Ticket Size Rules

1. Each ticket touches at most 2 to 4 files.
2. Each ticket has explicit inputs, outputs, and acceptance checks.
3. Each ticket includes rollback notes.

### Implementation Rules

1. Prefer adapter wrappers around existing modules.
2. Avoid cross-cutting refactors before parity tests pass.
3. Keep API and orchestration changes isolated.
4. Run targeted tests after each ticket.

### Required Ticket Template

Each ticket must include:

1. Objective
2. Files to modify
3. Exact code changes
4. Acceptance checks
5. Risks
6. Rollback steps

## 12. Testing and Acceptance Matrix

### Contract Tests

1. Every endpoint validates request schema and response shape.
2. Every error path returns structured error payload.

### Workflow Tests

1. Stage transitions follow the defined model.
2. Duplicate model run while active is rejected.
3. Session delete is idempotent.

### Parity Tests

1. Required-column validation parity
2. Output file naming parity
3. Visualization availability parity
4. Topic comparison output parity when Topic-Human exists

### Concurrency and Cleanup Tests

1. Parallel sessions remain isolated
2. Background tasks do not block API responsiveness
3. Session cleanup removes all session-scoped artifacts

## 13. Risks and Mitigations

1. Reactive-to-request behavior drift
   Mitigation: explicit state and parity tests

2. CPU-heavy modeling blocks API
   Mitigation: background execution and polling

3. Session/file leakage
   Mitigation: session-scoped artifact registry and idempotent delete

4. Overengineering during migration
   Mitigation: minimal architecture and strict ticket sizing

5. Premature UI scope expansion
   Mitigation: thin parity UI before any redesign

## 14. Deliverables

1. FastAPI backend with validated contracts
2. Explicit session and task orchestration
3. Deterministic session artifact pipeline
4. Polling status endpoint for long-running modeling
5. Thin parity UI flow
6. Test suite for contracts, workflow, parity, and cleanup

## 15. Execution Rule

Implement sequentially by phase. Preserve parity first. Defer hardening and enhancements until after parity signoff.
